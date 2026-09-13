"""
main.py -- Complete submission pipeline for Buy or Wait?

Generates output.csv in the repository root by evaluating all 250 requests
in dataset/requests.csv with the deterministic decision engine.
"""

import csv
import os
import sys
from datetime import date
from typing import Dict, List

# Ensure code/ directory is on the path
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
sys.path.insert(0, _CODE_DIR)

from data_loader import DATASET_DIR, Dataset, Request
from decision_engine import DecisionResult, evaluate_request
from financial_engine import _round2

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

ALLOWED_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

ALLOWED_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}


def process_all_requests(dataset: Dataset) -> List[Dict[str, str]]:
    """
    Process all requests from dataset/requests.csv and return output rows.
    """
    requests_path = os.path.join(DATASET_DIR, "requests.csv")
    if not os.path.exists(requests_path):
        raise FileNotFoundError(f"requests.csv not found at: {requests_path}")

    # Read requests in exact original order
    ordered_request_ids = []
    with open(requests_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ordered_request_ids.append(row["request_id"])

    print(f"Loaded {len(ordered_request_ids)} requests from requests.csv")

    rows: List[Dict[str, str]] = []
    missing_image_requests = []
    error_requests = []

    for i, req_id in enumerate(ordered_request_ids, 1):
        req = dataset.get_request(req_id)
        if req is None:
            print(f"Warning: Request {req_id} not indexed in Dataset")
            continue

        result = evaluate_request(req_id, dataset)

        if result.error:
            if "Missing image amount" in result.error:
                missing_image_requests.append(req_id)
            else:
                error_requests.append((req_id, result.error))

        # Format amount_safe_to_pay (clean integer format if whole number)
        safe_val = result.amount_safe_to_pay
        if safe_val == int(safe_val):
            safe_str = str(int(safe_val))
        else:
            safe_str = str(_round2(safe_val))

        row_dict = {
            "request_id": result.request_id,
            "amount_safe_to_pay": safe_str,
            "affordability_status": result.affordability_status,
            "recommended_payment_method": result.recommended_payment_method,
            "payment_plan": result.payment_plan,
            "earliest_date_for_full_payment": result.earliest_date_for_full_payment,
            "spending_changes_needed": result.spending_changes_needed,
            "decision_explanation": result.decision_explanation,
        }
        rows.append(row_dict)

    print(f"Processed {len(rows)} requests.")
    if missing_image_requests:
        print(f"Note: {len(missing_image_requests)} requests handled safely via fallback (unextracted images): {', '.join(missing_image_requests)}")
    if error_requests:
        print(f"Note: {len(error_requests)} requests encountered other issues: {error_requests}")

    return rows


def write_output_csv(rows: List[Dict[str, str]], output_path: str) -> None:
    """Write rows to the destination CSV with exact column ordering."""
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {output_path}")


def validate_output(rows: List[Dict[str, str]], dataset: Dataset) -> bool:
    """Validate output according to all challenge constraints."""
    print("\n--- Running Output Validation ---")
    valid = True

    if len(rows) != 250:
        print(f"ERROR: Expected 250 rows, got {len(rows)}")
        valid = False
    else:
        print(f"PASS: Exactly 250 rows generated.")

    seen_ids = set()
    for row in rows:
        req_id = row["request_id"]
        if req_id in seen_ids:
            print(f"ERROR: Duplicate request_id: {req_id}")
            valid = False
        seen_ids.add(req_id)

        # Status validation
        st = row["affordability_status"]
        if st not in ALLOWED_STATUSES:
            print(f"ERROR: Invalid affordability_status '{st}' in {req_id}")
            valid = False

        # Method validation
        met = row["recommended_payment_method"]
        if met not in ALLOWED_METHODS:
            print(f"ERROR: Invalid recommended_payment_method '{met}' in {req_id}")
            valid = False

        # Safe to pay range check
        req = dataset.get_request(req_id)
        if req:
            try:
                safe_float = float(row["amount_safe_to_pay"])
                if safe_float < 0 or safe_float > req.requested_amount + 1e-4:
                    print(f"ERROR: amount_safe_to_pay {safe_float} out of range [0, {req.requested_amount}] for {req_id}")
                    valid = False
            except ValueError:
                print(f"ERROR: non-numeric amount_safe_to_pay in {req_id}")
                valid = False

        # Explanation check
        if not row["decision_explanation"]:
            print(f"ERROR: Missing decision_explanation for {req_id}")
            valid = False

    if valid:
        print("PASS: All validation invariants passed successfully.")
    return valid


def main() -> None:
    print("=" * 60)
    print("Buy or Wait? -- Final Output Generator")
    print("=" * 60)

    print("\n[1/3] Loading dataset...")
    ds = Dataset()
    ds.load_all()
    print("Dataset successfully loaded.")

    print("\n[2/3] Evaluating requests...")
    rows = process_all_requests(ds)

    output_root_path = os.path.join(_REPO_ROOT, "output.csv")
    print(f"\n[3/3] Saving to {output_root_path}...")
    write_output_csv(rows, output_root_path)

    is_valid = validate_output(rows, ds)
    if not is_valid:
        print("\nWARNING: Some validation checks failed. Please inspect logs.")
        sys.exit(1)

    print("\nSUCCESS: output.csv is ready for submission.")


if __name__ == "__main__":
    main()
