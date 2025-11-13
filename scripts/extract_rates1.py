import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pdfplumber


TIER_HEADERS = [
    ("employee_only", re.compile(r"\bemployee\s*only\b", re.I)),
    ("employee_spouse", re.compile(r"\bemployee\s*\+\s*spouse\b", re.I)),
    ("employee_child", re.compile(r"\bemployee\s*\+\s*child\b", re.I)),
    ("family", re.compile(r"\bfamily\b", re.I)),
]

MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
PLAN_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-\*]*$", re.I)


def find_carrier_name(all_text: str) -> Optional[str]:
    candidates = [
        r"Blue Cross and Blue Shield of [A-Za-z ]+",
        r"UnitedHealthcare",
        r"Aetna",
        r"Cigna",
        r"Principal",
        r"Guardian",
        r"MetLife",
        r"BCBS[ A-Za-z]*",
    ]
    for pat in candidates:
        m = re.search(pat, all_text, re.I)
        if m:
            return m.group(0).strip()
    return None


def header_index_map(header_row: List[str]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for col_idx, cell in enumerate(header_row):
        text = (cell or "").replace("\n", " ").strip()
        for key, rx in TIER_HEADERS:
            if rx.search(text):
                idx[key] = col_idx
    return idx


def parse_money(cell: Any) -> Optional[float]:
    if cell is None:
        return None
    text = str(cell).strip()
    m = MONEY_RE.search(text)
    if not m:
        return None
    # remove $ and commas
    try:
        return float(m.group(0).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def extract_tables_from_page(page: pdfplumber.page.Page) -> List[List[List[Optional[str]]]]:
    try:
        return page.extract_tables() or []
    except Exception:
        return []


def looks_like_rate_table(table: List[List[Optional[str]]]) -> bool:
    if not table or not table[0]:
        return False
    header = [((c or "").replace("\n", " ").strip()) for c in table[0]]
    tiers = header_index_map(header)
    # Need at least 2 tier columns (Employee Only + something else) to consider
    return len(tiers) >= 2


def extract_rates_from_table(table: List[List[Optional[str]]]) -> List[Dict[str, Any]]:
    if not table:
        return []
    header = [((c or "").replace("\n", " ").strip()) for c in table[0]]
    tiers = header_index_map(header)
    if not tiers:
        return []

    results: List[Dict[str, Any]] = []
    for row in table[1:]:
        if not row or not row[0]:
            continue
        plan_id = str(row[0]).strip()
        # skip header/section rows
        if not PLAN_ID_RE.match(plan_id):
            continue

        rate_entry: Dict[str, Any] = {"plan_id": plan_id, "tiers": {}}
        has_any_rate = False
        for key, col_idx in tiers.items():
            if col_idx < len(row):
                value = parse_money(row[col_idx])
                if value is not None:
                    rate_entry["tiers"][key] = value
                    has_any_rate = True
        if has_any_rate:
            results.append(rate_entry)
    return results


def extract_pdf_rates(pdf_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "file": str(pdf_path),
        "carrier": None,
        "plans": [],  # type: ignore[list-item]
    }

    all_text_chunks: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                all_text_chunks.append(txt)

        # Carrier guess from full text
        carrier_guess = find_carrier_name("\n".join(all_text_chunks))
        if carrier_guess:
            out["carrier"] = carrier_guess

        # Extract tiered tables
        for page in pdf.pages:
            tables = extract_tables_from_page(page)
            for tbl in tables:
                if not looks_like_rate_table(tbl):
                    continue
                rates = extract_rates_from_table(tbl)
                if not rates:
                    continue
                out["plans"].extend(rates)  # type: ignore[arg-type]

    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract tiered rates from proposal PDF into JSON.")
    parser.add_argument("pdf", type=str, help="Path to proposal PDF")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    data = extract_pdf_rates(pdf_path)

    out_path = Path(args.out) if args.out else Path("output") / (pdf_path.stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Print a tiny summary
    print(f"Wrote: {out_path}")
    print(f"Carrier: {data.get('carrier')}")
    print(f"Plans extracted: {len(data.get('plans', []))}")


if __name__ == "__main__":
    main()

