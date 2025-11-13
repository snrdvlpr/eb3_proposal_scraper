import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pdfplumber

from scripts.remote_llm import RemoteLLM


SYSTEM_PROMPT = """You are a data extraction assistant that converts PDF proposal pages into structured rate data.
Only return JSON that conforms exactly to this schema:
{
  "carrier": string | null,
  "plans": [
    {
      "plan_name": string | null,
      "plan_id": string | null,
      "rate_structure": one of [
        "unit_rate",
        "2_tier",
        "3_tier",
        "4_tier",
        "5_tier",
        "8_tier",
        "aca_age",
        "age_band_5",
        "age_band_10",
        "esc_5_year",
        "esc_10_year",
        "4_tier_5_year",
        "4_tier_10_year",
        "3_tier_age_band",
        "2_tier_age_band"
      ],
      "rates": object  // keys depend on structure, see below
    }
  ]
}

For the "rates" object use these keys:
- unit_rate: {"unit_rate": number}
- 2_tier: {"employee_only": number, "family": number}
- 3_tier: {"employee_only": number, "employee_plus_one": number, "employee_plus_two_or_more": number}
- 4_tier: {"employee_only": number, "employee_spouse": number, "employee_child": number, "family": number}
- 5_tier: {"employee_only": number, "employee_spouse": number, "employee_child": number, "employee_two_or_more_children": number, "family": number}
- 8_tier: use descriptive keys such as {"employee_only": number, "employee_spouse": number, "employee_child": number, "employee_two_children": number, "employee_three_or_more_children": number, "employee_spouse_child": number, "employee_spouse_two_children": number, "employee_spouse_three_or_more_children": number}
- aca_age: keys for ages "<15", "15", ..., "64+" where available
- age_band_5: keys like "<20", "20-24", ..., "80+"
- age_band_10: keys like "<20", "20-29", ..., "90+"
- esc_5_year and esc_10_year: nested object {"employee": {...band rates...}, "spouse": {...band rates...}, "children": number | null}
- 4_tier_5_year / 4_tier_10_year: nested object with keys "employee_only", "employee_spouse", "employee_child", "family", each holding the appropriate age-band dictionary.
- 3_tier_age_band / 2_tier_age_band: use separate dictionaries for employee vs dependent bands.

Return nulls when information is missing. Ignore marketing text, totals, enrollments, or exposure counts.
Return strictly valid JSON—no comments or trailing text. If nothing relevant is present, return {"carrier": null, "plans": []}.
"""

USER_PROMPT_TEMPLATE = """You will receive text extracted from proposal pages.
Chunk information:
- File: {pdf_name}
- Pages: {page_range}

Only consider information necessary to identify carrier names, plan identifiers, plan names, and rate values.
Text:
\"\"\"
{page_text}
\"\"\"

Respond with JSON that adheres to the required schema.
"""

MONEY_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class PlanEntry:
    plan_name: Optional[str]
    plan_id: Optional[str]
    rate_structure: str
    rates: Dict[str, Any]
    source_pages: List[int] = field(default_factory=list)


def chunk_pages(
    pdf_path: Path,
    pages_per_chunk: int = 2,
    max_chars: int = 6000,
) -> Iterable[Tuple[List[int], str]]:
    """
    Yield (page_numbers, text) tuples while limiting prompt size.
    """
    with pdfplumber.open(pdf_path) as pdf:
        buffer_pages: List[int] = []
        buffer_text: List[str] = []

        for idx, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            clean = " ".join(text.split())
            if not clean:
                continue

            candidate_pages = buffer_pages + [idx]
            candidate_text = "\n".join(buffer_text + [clean])

            if (
                len(candidate_pages) > pages_per_chunk
                or len(candidate_text) > max_chars
            ) and buffer_pages:
                yield buffer_pages, "\n".join(buffer_text)
                buffer_pages = [idx]
                buffer_text = [clean]
            else:
                buffer_pages = candidate_pages
                buffer_text = buffer_text + [clean]

        if buffer_pages:
            yield buffer_pages, "\n".join(buffer_text)


def extract_json(response: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Attempt to recover JSON substring
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def coerce_rate_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        match = MONEY_PATTERN.search(cleaned)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return value
        lowered = cleaned.lower()
        if lowered in {"na", "n/a", "not applicable"}:
            return None
    if isinstance(value, dict):
        return {k: coerce_rate_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [coerce_rate_value(v) for v in value]
    return value


def normalize_plan(raw: Dict[str, Any], source_pages: List[int]) -> Optional[PlanEntry]:
    plan = raw.get("plan") or raw
    rate_structure = plan.get("rate_structure")
    rates = plan.get("rates")

    if not isinstance(rate_structure, str) or not isinstance(rates, dict):
        return None

    normalized_rates = coerce_rate_value(rates)
    if not isinstance(normalized_rates, dict):
        return None

    return PlanEntry(
        plan_name=plan.get("plan_name"),
        plan_id=plan.get("plan_id"),
        rate_structure=rate_structure,
        rates=normalized_rates,
        source_pages=sorted(set(source_pages)),
    )


async def process_chunk(
    llm: RemoteLLM,
    pdf_name: str,
    page_numbers: List[int],
    page_text: str,
    retries: int = 2,
) -> Dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        pdf_name=pdf_name,
        page_range=", ".join(str(p) for p in page_numbers),
        page_text=page_text,
    )

    attempt = 0
    while attempt <= retries:
        response = await llm.chat(SYSTEM_PROMPT, user_prompt)
        data = extract_json(response)
        if data is not None:
            return data
        attempt += 1
        user_prompt += "\n\nReminder: respond with valid JSON only."

    return {"carrier": None, "plans": []}


async def extract_pdf_with_llm(
    pdf_path: Path,
    output_path: Path,
    pages_per_chunk: int = 2,
    max_chars: int = 6000,
) -> Dict[str, Any]:
    llm = RemoteLLM()
    carriers: List[str] = []
    plans: Dict[str, PlanEntry] = {}

    for page_numbers, text in chunk_pages(
        pdf_path, pages_per_chunk=pages_per_chunk, max_chars=max_chars
    ):
        result = await process_chunk(llm, pdf_path.name, page_numbers, text)

        carrier = result.get("carrier")
        if isinstance(carrier, str):
            carriers.append(carrier.strip())

        for plan_raw in result.get("plans", []):
            if not isinstance(plan_raw, dict):
                continue
            entry = normalize_plan(plan_raw, page_numbers)
            if not entry:
                continue
            key = entry.plan_id or entry.plan_name or f"page-{page_numbers}"
            if key not in plans:
                plans[key] = entry
            else:
                plans[key].source_pages = sorted(
                    set(plans[key].source_pages + entry.source_pages)
                )

    carrier_value = None
    if carriers:
        most_common = Counter(carriers).most_common(1)
        if most_common:
            carrier_value = most_common[0][0]

    output = {
        "file": str(pdf_path),
        "carrier": carrier_value,
        "plans": [
            {
                "plan_name": plan.plan_name,
                "plan_id": plan.plan_id,
                "rate_structure": plan.rate_structure,
                "rates": plan.rates,
                "source_pages": plan.source_pages,
            }
            for plan in plans.values()
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract plan rate data from a proposal PDF using Qwen LLM."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    parser.add_argument(
        "--out",
        help="Optional output JSON path (defaults to output/<pdf_name>.json)",
    )
    parser.add_argument(
        "--pages-per-chunk",
        type=int,
        default=2,
        help="Number of pages to group per LLM prompt",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Maximum characters per chunk",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_path = (
        Path(args.out).resolve()
        if args.out
        else Path("output") / f"{pdf_path.stem}.json"
    )

    asyncio.run(
        extract_pdf_with_llm(
            pdf_path,
            output_path,
            pages_per_chunk=args.pages_per_chunk,
            max_chars=args.max_chars,
        )
    )


if __name__ == "__main__":
    main()

