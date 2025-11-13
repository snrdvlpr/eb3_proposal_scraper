import argparse
import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pdfplumber

from scripts.remote_llm import RemoteLLM

# Configure logging
logger = logging.getLogger(__name__)


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


# Keywords that indicate rate-related content
RATE_KEYWORDS = re.compile(
    r"\b(?:rates?|premium|employee|spouse|child|family|tier|plan|coverage|deductible|"
    r"age[s]?|band[s]?|carrier|option|ppo|hmo|aca)\b",
    re.IGNORECASE,
)


def chunk_pages(
    pdf_path: Path,
    pages_per_chunk: int = 2,
    max_chars: int = 6000,
    filter_empty: bool = True,
) -> Iterable[Tuple[List[int], str]]:
    """
    Yield (page_numbers, text) tuples while limiting prompt size.
    Optimized to skip clearly non-relevant pages early.
    """
    logger.info(f"Starting to chunk pages from PDF: {pdf_path.name}")
    
    with pdfplumber.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)
        logger.info(f"PDF has {total_pages} total pages")
        
        buffer_pages: List[int] = []
        buffer_text: List[str] = []
        processed_pages = 0
        skipped_pages = 0

        for idx, page in enumerate(pdf_doc.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                skipped_pages += 1
                logger.debug(f"Page {idx}: Skipped (no text)")
                continue
            
            # Normalize whitespace efficiently
            clean = " ".join(text.split())
            if not clean:
                skipped_pages += 1
                logger.debug(f"Page {idx}: Skipped (empty after normalization)")
                continue

            # Early filtering: skip pages that clearly don't contain rate data
            if filter_empty and len(clean) < 100:
                skipped_pages += 1
                logger.debug(f"Page {idx}: Skipped (too short: {len(clean)} chars)")
                continue

            candidate_pages = buffer_pages + [idx]
            candidate_text = "\n".join(buffer_text + [clean])

            # Yield chunk if limits exceeded
            if (
                len(candidate_pages) > pages_per_chunk
                or len(candidate_text) > max_chars
            ) and buffer_pages:
                logger.info(
                    f"Chunk created: pages {buffer_pages[0]}-{buffer_pages[-1]} "
                    f"({len(buffer_pages)} pages, {len(buffer_text)} chars)"
                )
                yield buffer_pages, "\n".join(buffer_text)
                processed_pages += len(buffer_pages)
                buffer_pages = [idx]
                buffer_text = [clean]
            else:
                buffer_pages = candidate_pages
                buffer_text = buffer_text + [clean]

        if buffer_pages:
            processed_pages += len(buffer_pages)
            logger.info(
                f"Final chunk created: pages {buffer_pages[0]}-{buffer_pages[-1]} "
                f"({len(buffer_pages)} pages, {len(buffer_text)} chars)"
            )
            yield buffer_pages, "\n".join(buffer_text)

    logger.info(
        f"Chunking complete: {processed_pages} pages processed, "
        f"{skipped_pages} pages skipped, {total_pages} total pages"
    )


def extract_json(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response with robust error handling.
    Attempts to recover JSON even if wrapped in text.
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip()
    
    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block (might be wrapped in markdown or text)
    # Look for first { and matching }
    start = response.find("{")
    if start == -1:
        return None
    
    # Find matching closing brace
    brace_count = 0
    end = -1
    for i in range(start, len(response)):
        if response[i] == "{":
            brace_count += 1
        elif response[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    
    if end > start:
        try:
            json_str = response[start : end + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
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
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Process a single chunk of pages through the LLM.
    Returns extracted data or empty result on failure.
    """
    page_range_str = f"{page_numbers[0]}-{page_numbers[-1]}" if len(page_numbers) > 1 else str(page_numbers[0])
    logger.info(f"Processing chunk: pages {page_range_str} ({len(page_numbers)} pages, {len(page_text)} chars)")
    
    # Truncate text if too long to avoid token limit issues
    original_len = len(page_text)
    if len(page_text) > 8000:  # Conservative limit
        page_text = page_text[:8000] + "...[truncated]"
        logger.warning(f"Chunk truncated from {original_len} to {len(page_text)} chars")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        pdf_name=pdf_name,
        page_range=", ".join(str(p) for p in page_numbers),
        page_text=page_text,
    )

    attempt = 0
    last_error = None
    
    while attempt <= retries:
        try:
            logger.debug(f"LLM request for pages {page_range_str}, attempt {attempt + 1}/{retries + 1}")
            response = await llm.chat(
                SYSTEM_PROMPT,
                user_prompt,
                max_new_tokens=max_tokens,
            )
            logger.debug(f"LLM response received for pages {page_range_str}: {len(response)} chars")
            
            data = extract_json(response)
            if data is not None and isinstance(data, dict):
                plans_count = len(data.get("plans", []))
                carrier = data.get("carrier")
                logger.info(
                    f"Chunk processed successfully: pages {page_range_str} - "
                    f"found {plans_count} plans, carrier: {carrier or 'None'}"
                )
                return data
            
            # If JSON parsing failed, retry with reminder
            logger.warning(f"JSON parsing failed for pages {page_range_str}, attempt {attempt + 1}")
            attempt += 1
            if attempt <= retries:
                user_prompt += "\n\nReminder: respond with valid JSON only—no comments or extra text."
        except Exception as e:
            last_error = str(e)
            logger.error(f"Error processing chunk pages {page_range_str}, attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt <= retries:
                wait_time = 0.5 * attempt
                logger.debug(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)  # Exponential backoff

    # Return empty result on failure
    logger.warning(f"Chunk processing failed after {retries + 1} attempts: pages {page_range_str}")
    if last_error:
        logger.error(f"Last error: {last_error}")
    return {"carrier": None, "plans": []}


async def extract_pdf_with_llm(
    pdf_path: Path,
    output_path: Path,
    pages_per_chunk: int = 2,
    max_chars: int = 6000,
    max_concurrent: int = 3,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Extract rates from PDF using LLM with optimized parallel processing.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Path to write JSON output
        pages_per_chunk: Pages per LLM prompt
        max_chars: Max characters per chunk
        max_concurrent: Max concurrent LLM requests
        max_tokens: Max tokens per LLM response
    """
    logger.info(
        f"Starting extraction: {pdf_path.name} "
        f"(chunks: {pages_per_chunk} pages, max_chars: {max_chars}, "
        f"max_concurrent: {max_concurrent}, max_tokens: {max_tokens})"
    )
    
    llm = RemoteLLM()
    carriers: List[str] = []
    plans: Dict[str, PlanEntry] = {}

    try:
        # Create chunks list for parallel processing
        chunks = list(chunk_pages(pdf_path, pages_per_chunk=pages_per_chunk, max_chars=max_chars))
        
        if not chunks:
            logger.warning(f"No chunks created from PDF: {pdf_path.name}")
            # No content found in PDF
            output = {
                "file": str(pdf_path),
                "carrier": None,
                "plans": [],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            logger.info(f"Extraction complete: 0 plans found, output written to {output_path}")
            return output

        logger.info(f"Processing {len(chunks)} chunks with max_concurrent={max_concurrent}")

        # Process chunks with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(page_numbers: List[int], text: str) -> Dict[str, Any]:
            async with semaphore:
                return await process_chunk(
                    llm, pdf_path.name, page_numbers, text, max_tokens=max_tokens
                )

        # Process all chunks in parallel (with concurrency limit)
        tasks = [
            process_with_limit(page_numbers, text)
            for page_numbers, text in chunks
        ]
        logger.info(f"Started {len(tasks)} parallel processing tasks")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"All {len(results)} tasks completed")

        # Process results
        successful_chunks = 0
        failed_chunks = 0
        total_plans_found = 0
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failed_chunks += 1
                logger.error(f"Chunk {idx + 1} failed with exception: {result}")
                continue
                
            if not isinstance(result, dict):
                failed_chunks += 1
                logger.warning(f"Chunk {idx + 1} returned invalid result type: {type(result)}")
                continue

            successful_chunks += 1
            carrier = result.get("carrier")
            if isinstance(carrier, str) and carrier.strip():
                carriers.append(carrier.strip())

            chunk_plans = result.get("plans", [])
            total_plans_found += len(chunk_plans)
            
            for plan_raw in chunk_plans:
                if not isinstance(plan_raw, dict):
                    continue
                entry = normalize_plan(plan_raw, chunks[idx][0])
                if not entry:
                    continue
                key = entry.plan_id or entry.plan_name or f"page-{chunks[idx][0]}"
                if key not in plans:
                    plans[key] = entry
                else:
                    plans[key].source_pages = sorted(
                        set(plans[key].source_pages + entry.source_pages)
                    )

        logger.info(
            f"Results processing complete: {successful_chunks} successful chunks, "
            f"{failed_chunks} failed chunks, {total_plans_found} total plans found, "
            f"{len(plans)} unique plans after deduplication"
        )

        # Determine most common carrier
        carrier_value = None
        if carriers:
            most_common = Counter(carriers).most_common(1)
            if most_common:
                carrier_value = most_common[0][0]
                logger.info(f"Most common carrier: {carrier_value} (found {most_common[0][1]} times)")

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
        
        logger.info(
            f"Extraction complete: {len(plans)} unique plans extracted, "
            f"carrier: {carrier_value or 'None'}, output written to {output_path}"
        )
        
        return output

    finally:
        # Cleanup LLM session
        await llm.close()
        logger.debug("LLM session closed")


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

