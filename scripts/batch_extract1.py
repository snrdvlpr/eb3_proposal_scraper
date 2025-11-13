from pathlib import Path
import json
from typing import List
import sys

# Ensure project root on sys.path so we can import sibling modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the extractor without invoking as a subprocess
from scripts.extract_rates import extract_pdf_rates


def main() -> None:
    proposals_dir = Path("source/proposals")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files: List[Path] = sorted(proposals_dir.glob("*.pdf"))
    summary = []

    for pdf_path in pdf_files:
        data = extract_pdf_rates(pdf_path)
        out_path = output_dir / f"{pdf_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {out_path} (plans: {len(data.get('plans', []))})")
        summary.append({"file": str(pdf_path), "output": str(out_path), "plans": len(data.get("plans", []))})

    # Write a small manifest for convenience
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

