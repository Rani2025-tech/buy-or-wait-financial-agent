# Buy or Wait? — AI-Powered Financial Decision Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-184%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](#)
[![Deterministic](https://img.shields.io/badge/Engine-Deterministic%20%26%20Offline-orange.svg)](#)

A deterministic financial affordability decision agent built for **HackerRank Orchestrate (September 2026)**. The system determines whether a user can safely afford a requested purchase or payment, recommending personalized payment plans, safe dates, or flexible spending adjustments while strictly safeguarding essential commitments and a minimum buffer balance.

---

## 1. Overview

**Buy or Wait?** evaluates whether a user can safely proceed with a requested expense based on their current balance, 90-day cash-flow forecast, recurring commitments, pending obligations, seller payment options, and unstructured supporting evidence (messages and images).

For every purchase or payment request, the agent produces a structured, actionable decision: whether to pay in full immediately, split into partial payments, use a provider installment plan, wait for future income, or decline the expense.

---

## 2. Problem

When evaluating a major purchase, checking the current available bank balance alone is insufficient:

* A user with a high current balance may have large scheduled rent payments, debt obligations, or pending debits due within days.
* A user with a low balance today may have confirmed incoming salary settling within the week.
* Recurring essential spending (groceries, utilities, healthcare) continuously drains available liquidity.
* Overlooking payment options (such as 0% interest provider installment schedules) may cause a user to postpone an affordable purchase unnecessarily.
* Violating the user's defined `minimum_balance_to_keep` leaves them vulnerable to financial emergencies.

---

## 3. Solution

The system implements a deterministic, multi-stage financial analysis pipeline:

1. **Data Ingestion**: Loads user profiles, historical/pending/scheduled financial events, seller payment options, fixed exchange rates, messages, and image references from structured CSV files.

2. **Financial State Reconstruction**: Reconstructs current liquid balances, segregates settled funds from pending debits, ignores non-cash/unrealized events, and enforces user priorities and protected spending categories.

3. **Evidence Integration**:

   * **Messages**: Parses structured financial updates (salary increases, temporary pay cuts, employment terminations, rent increases, confirmed client invoices) in both English and Indonesian.
   * **Images**: Resolves verified image-linked transaction amounts from receipts, invoices, bills, and payslips without requiring live OCR runtime dependencies.

4. **Foreign Exchange Engine**: Converts foreign-currency cash events and requests to the user's home currency using dated fixed exchange rates via direct, inverse, or multi-hop paths.

5. **90-Day Cash-Flow Forecasting**: Simulates daily opening, closing, and spendable balances over a 90-day forward horizon, modeling recurring salary cycles and essential spending.

6. **Plan Generation & Ranking**: Evaluates candidate plans (immediate payment, partial payments, installments, delayed payment, and flexible spending reductions/cancellations) and ranks them using conservative safety rules.

7. **Validation**: Enforces strict mathematical invariants, ensuring the projected balance never dips below the user's minimum buffer balance.

---

## 4. Key Features

* **90-Day Daily Cash-Flow Forecasting**: Day-by-day cash balance projection accounting for recurring salary, rent, utilities, and debt obligations.

* **Minimum-Balance Protection**: Enforces that projected spendable balance never falls below `minimum_balance_to_keep`.

* **Comprehensive Payment Modalities**:

  * `full_payment`: Safe one-time payment on request date or future safe date.
  * `partial_payment`: Exactly two payments (safe amount today + remainder on earliest full payment date) when permitted.
  * `installments`: Evaluates supplier options from `request_payment_options.csv` respecting `max_installment_months`.
  * `wait`: Identifies the exact earliest safe date for full payment after scheduled inflows.
  * `not_recommended`: Safely rejects requests that cannot be completed within deadlines or budget limits.

* **Flexible-Expense Adjustments**: Explores up to 3 targeted `stop:<event_id>` or `reduce_to:<event_id>:<amount>` actions on non-protected, flexible expenses to unlock affordability.

* **Bilingual Message Interpretation**: Detects and integrates salary revisions, contract terminations, rent adjustments, and confirmed client invoices from English and Indonesian messages.

* **Image Transaction Amount Resolution**: Resolves verified image-linked transaction figures from receipts, invoices, bills, and payslips.

* **Fixed FX Conversion**: Deterministic currency conversion utilizing dataset exchange rates with forward reference dates.

* **Zero Double-Counting**: Strict guards prevent duplicate projections between settled history, scheduled future events, and external message/image evidence.

---

## 5. How It Works

```text
               ┌────────────────────────────────────────────────────────┐
               │                     Dataset Input                      │
               │ (profiles, events, options, fx rates, messages, images)│
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │                      Data Loader                       │
               │   (Parses CSVs, types dataclasses, builds indexes)    │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │              Financial State Reconstruction            │
               │  (Filters settled vs pending, excludes non-cash/void) │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │                Message & Image Evidence                │
               │ (Resolves salary shifts, invoices, receipts, payslips) │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │              90-Day Forward Cash-Flow Engine           │
               │ (Simulates day-by-day spendable balances with FX rates)│
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │                    Decision Engine                      │
               │ (Generates & ranks: full, partial, installment, wait)  │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │                  Output Validation                      │
               │    (Verifies schema, non-negativity, safety bounds)    │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │                       output.csv                        │
               │                (Final submission output)                 │
               └────────────────────────────────────────────────────────┘
```

---

## 6. Decision Logic

### Affordability Statuses

| Status                 | Description                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `affordable_now`       | The user can pay the full requested amount immediately on the request date without breaching their minimum balance.                           |
| `affordable_with_plan` | The purchase is affordable via a multi-payment schedule (installments, two-part partial payment) or by applying approved spending changes.    |
| `affordable_later`     | The purchase cannot be safely made today, but waiting for confirmed future income enables a safe full payment by the desired completion date. |
| `not_affordable`       | The purchase cannot safely be completed by the desired deadline under any supported payment method or permitted spending reduction.           |

### Recommended Payment Methods

| Payment Method    | Description                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `full_payment`    | Single payment of the entire requested amount on the request date or earliest safe date.                         |
| `partial_payment` | Exactly two payments: `amount_safe_to_pay` on `request_date`, and remainder on `earliest_date_for_full_payment`. |
| `installments`    | Equal periodic installment schedule selected from supplier payment options.                                      |
| `wait`            | User is advised to wait until `earliest_date_for_full_payment` when liquidity is restored.                       |
| `not_recommended` | The expense is unsafe and not recommended.                                                                       |

---

## 7. Project Structure

```text
├── code/
│   ├── main.py                  # Primary entry point: loads dataset, evaluates 250 requests, writes output.csv
│   ├── data_loader.py           # Data ingestion and typed indexing for all dataset tables
│   ├── financial_engine.py      # 90-day cash flow simulation, FX converter, recurrence detection, image resolver
│   ├── decision_engine.py       # Candidate generation, affordability ranking, plan formatting, explanations
│   ├── message_parser.py        # Deterministic English & Indonesian regex parser for message evidence
│   └── evaluation/
│       ├── main.py              # Evaluation & scoring script
│       └── usage_report.md      # Model and token usage report
│
├── dataset/
│   ├── financial_profiles.csv       # User balance, minimum reserve, priorities, protected/flexible categories
│   ├── financial_events.csv         # Historical, scheduled, and pending financial transactions
│   ├── request_payment_options.csv  # Seller-provided installment and financing options
│   ├── exchange_rates.csv           # Fixed historical and future exchange rates
│   ├── requests.csv                 # 250 evaluation requests
│   ├── sample_requests.csv          # 25 public reference examples
│   ├── messages.csv                 # Unstructured text notifications and evidence
│   ├── images.csv                   # Image metadata and event mappings
│   └── media/images/                # PNG receipts, invoices, and payslips
│
├── tests/
│   ├── test_stage2.py           # Stage 2 data loader, cash flow, and financial engine tests
│   ├── test_stage3.py           # Stage 3 decision engine, affordability classification, and invariant tests
│   ├── test_messages.py         # Bilingual message parsing and financial context integration tests
│   └── test_images.py           # Image amount resolution, FX handling, and recurring boundary tests
│
├── output.csv                  # Generated prediction file for all 250 evaluation requests
├── problem_statement.md        # Official challenge specification
├── AGENTS.md                   # Agent harness rules and operational contracts
└── README.md                   # Project documentation
```

---

## 8. Testing & Validation

The test suite validates data loading, FX rate traversal, message interpretation, image integration, decision ranking, and mathematical safety invariants across all components.

```bash
pytest -q
```

**Test Results**:

```text
........................................................................ [ 39%]
........................................................................ [ 78%]
........................................                                 [100%]

184 passed in 2.85s
```

* **184 / 184 tests passing** (100% pass rate).
* Validated on all **250 evaluation requests** in `requests.csv`.
* The full pipeline validates the required schema, financial bounds, payment-plan constraints, and other decision invariants.

---

## 9. Illustrative Example

> **Note:** The following scenario is a fictional example for illustrative purposes.

* **User**: `user_demo` (Home Currency: `INR`)
* **Available Balance**: ₹120,000 | **Minimum Reserve**: ₹50,000
* **Request**: Laptop Purchase of **₹80,000** on **2026-10-01** (Completion Deadline: **2026-11-15**)
* **Upcoming Commitments**: Scheduled Rent of ₹60,000 due on **2026-10-05**; Salary of ₹90,000 settling on **2026-10-15**.

### Agent Evaluation

1. **Pay Full Today?**

   * Balance after purchase = ₹120,000 - ₹80,000 = ₹40,000.
   * On 2026-10-05, Rent of ₹60,000 brings balance to -₹20,000 (breaches ₹50,000 minimum reserve). → **Unsafe today**.

2. **Installments Option Available?**

   * Seller offers 3 monthly payments of ₹27,000 starting 2026-10-01.
   * Balance after payment 1 = ₹93,000; after Rent = ₹33,000 (below ₹50,000 buffer). → **Unsafe**.

3. **Wait for Inflow?**

   * On 2026-10-15, salary of ₹90,000 settles.
   * Projected balance on 2026-10-15 after Rent = ₹150,000.
   * Paying ₹80,000 on 2026-10-15 leaves ₹70,000 (≥ ₹50,000 reserve) for all 90 days. → **Safe**.

### Output Decision

* `affordability_status`: `affordable_later`
* `recommended_payment_method`: `wait`
* `earliest_date_for_full_payment`: `2026-10-15`
* `decision_explanation`: *"Pay INR 80,000 in full on 15 October 2026 after confirmed salary credit. Paying earlier would take the balance below the INR 50,000 minimum reserve due to scheduled rent."*

---

## 10. Tech Stack

* **Language**: Python 3.10+
* **Standard Libraries**: `dataclasses`, `datetime`, `collections`, `decimal`, `re`, `csv`
* **Testing**: `pytest`
* **Architecture**: 100% deterministic, offline, rule-based financial decision engine (zero external API, LLM, or cloud dependencies).

---

## 11. Running Locally

### Clone Repository

```bash
git clone https://github.com/Rani2025-tech/buy-or-wait-financial-agent.git
cd buy-or-wait-financial-agent
```

### Run Full Evaluation Pipeline

To process all 250 evaluation requests and regenerate `output.csv`:

```bash
python code/main.py
```

### Run Test Suite

```bash
pytest -v
```

---

## 12. Output Format

The system produces `output.csv` matching the required 8-column contract:

| Column                           | Type   | Description                                                                                                          |                |
| -------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------- | -------------- |
| `request_id`                     | String | Unique evaluation request identifier (`request_26` to `request_275`).                                                |                |
| `amount_safe_to_pay`             | Float  | Amount safe to pay immediately on `request_date` before spending changes (between 0 and requested amount inclusive). |                |
| `affordability_status`           | Enum   | `affordable_now`, `affordable_with_plan`, `affordable_later`, or `not_affordable`.                                   |                |
| `recommended_payment_method`     | Enum   | `full_payment`, `partial_payment`, `installments`, `wait`, or `not_recommended`.                                     |                |
| `payment_plan`                   | String | Chronological `YYYY-MM-DD:amount` entries separated by pipes (`                                                      | `), or `none`. |
| `earliest_date_for_full_payment` | String | First projected date (`YYYY-MM-DD`) for a safe single full payment, or blank.                                        |                |
| `spending_changes_needed`        | String | Up to 3 `stop:<event_id>` or `reduce_to:<event_id>:<amount>` actions, or `none`.                                     |                |
| `decision_explanation`           | String | Concise, grounded explanation justifying the recommendation.                                                         |                |

---

## 13. Design Philosophy

* **Conservatism First**: Balances are calculated conservatively. Pending debits are reserved immediately, whereas unconfirmed income, windfalls, and unrealized assets are never counted until settled.

* **Buffer Inviolability**: The user's `minimum_balance_to_keep` is treated as a hard safety boundary across all 90 projected days.

* **Traceable Decisions**: Recommendations are derived strictly from explicit financial data, deterministic rules, and validated calculations, ensuring full auditability without ungrounded heuristics.

---

## 14. Limitations

* **Dataset-Bound**: Operates strictly on provided offline financial records and fixed dated exchange rates. Does not connect to live open banking APIs, live stock feeds, or real-time forex streams.

* **Deterministic Evidence Resolution**: Relies on structured regex matching and verified image-linked transaction amount mappings rather than live OCR models.

---

## 15. Project Status

* **Dataset Coverage**: All 250 requests processed in `output.csv`.
* **Test Coverage**: 184 / 184 tests passing.
* **Project Status**: Completed and fully verified on the 250-request dataset.
