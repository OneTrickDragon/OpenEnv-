from __future__ import annotations
 
import base64
import contextlib
import io
import math
import random
import re
import sys
import textwrap
import traceback
from typing import Any
 
import numpy as np
import pandas as pd
 
from models import (
    Action, ActionType, Observation, Reward, State, TaskID,
)


MAX_STEPS       = 20
STEP_EXEC_LIMIT = 50   # max lines of code per exec action
 
ALLOWED_IMPORTS = {
    "pandas", "numpy", "re", "datetime", "difflib",
    "unicodedata", "collections", "itertools", "math", "string",
}
 
TASK_SPECS: dict[TaskID, str] = {
    TaskID.ECOMMERCE_EASY: textwrap.dedent("""
        TASK: E-commerce order cleanup  [EASY]
 
        You have a DataFrame called `df` with 500 e-commerce orders.
 
        KNOWN ISSUES to fix:
        1. `order_date` is stored as a plain string — convert to datetime64[ns].
        2. `price` has ~15% nulls — impute with the column median.
        3. `quantity` has some negative values — clip to 0.
        4. `revenue` contains mixed formats like "$12.50", "12,50 USD", "12.50"
           — strip symbols, convert to float (USD).
        5. `customer_id` has leading/trailing whitespace — strip it.
        6. `status` values should be one of: pending, shipped, delivered, cancelled
           — lowercase and strip any others; map "complete" → "delivered".
 
        SCHEMA (expected output dtypes):
          order_id      object
          order_date    datetime64[ns]
          customer_id   object
          product_id    object
          quantity      int64
          price         float64
          revenue       float64
          status        object
 
        Submit your cleaned DataFrame with action type='submit'.
        You have at most 20 exec steps before automatic submission.
    """).strip(),
 
    TaskID.PATIENT_RECORDS_MEDIUM: textwrap.dedent("""
        TASK: Patient records deduplication  [MEDIUM]
 
        You have a DataFrame called `df` with ~1200 patient records.
 
        KNOWN ISSUES to fix:
        1. DUPLICATES: ~20% of records are fuzzy duplicates (same patient,
           slightly different name formatting: "John Smith" / "J. Smith" / "SMITH, John").
           Keep the most-complete record (fewest nulls). Use patient_id as the
           canonical key — records sharing a patient_id are duplicates.
        2. `dob` appears in three formats: "1985-03-22", "03/22/1985", "22-Mar-1985"
           — normalise all to ISO-8601 (YYYY-MM-DD) string.
        3. `phone` uses mixed delimiters: "(555) 123-4567", "555.123.4567", "5551234567"
           — normalise to E.164-ish format: "+15551234567".
        4. `icd10_notes` is a free-text field containing ICD-10 codes embedded
           in sentences — extract the first valid ICD-10 code (pattern: letter +
           2 digits + optional dot + up to 4 alphanumerics, e.g. "J45.9") into a
           new column `icd10_code`. Set to None if no code found.
        5. `email` — lowercase all email addresses.
 
        SCHEMA (expected output):
          patient_id    object
          first_name    object
          last_name     object
          dob           object  (YYYY-MM-DD string)
          phone         object  (+1XXXXXXXXXX)
          email         object
          icd10_code    object  (nullable)
          [icd10_notes may be dropped]
 
        Submit with action type='submit'.
    """).strip(),
 
    TaskID.FINANCIAL_AUDIT_HARD: textwrap.dedent("""
        TASK: Financial transaction audit  [HARD]
 
        You have a DataFrame called `df` with 5000 financial transactions.
        You must apply the following named BUSINESS RULES. Do not simply
        delete bad rows — flag them with a `violation` column (comma-separated
        rule names, or empty string if clean).
 
        BUSINESS RULES:
        R1  REFUND_SIGN:    rows where txn_type='REFUND' must have amount < 0.
        R2  DEBIT_POSITIVE: rows where txn_type='DEBIT'  must have amount > 0.
        R3  DATE_ORDER:     transaction_date must be >= account_open_date for
                            that account_id.
        R4  FX_RECONCILE:   for non-USD rows, usd_amount must equal
                            amount * fx_rate (within 0.01 tolerance).
                            Correct usd_amount where it is wrong.
        R5  PARENT_REF:     REFUND rows must have a parent_txn_id that exists
                            in the txn_id column. Flag those that don't.
        R6  DUPE_TXN:       exact duplicate (txn_id, account_id, amount, date)
                            should have duplicate=True in a new bool column.
        R7  NULL_REQUIRED:  account_id, txn_id, amount, transaction_date must
                            never be null. Drop rows where they are.
        R8  AMOUNT_OUTLIER: flag rows where abs(amount) > 3 standard deviations
                            from the per-account mean with rule name OUTLIER.
        R9  CURRENCY_VALID: currency must be one of USD, EUR, GBP, JPY, CAD.
                            Flag others as INVALID_CCY.
        R10 ZERO_AMOUNT:    amount == 0 is never valid. Flag as ZERO_AMOUNT.
 
        SCHEMA (expected output adds these columns to original):
          violation   object  (comma-separated rule names, "" if clean)
          duplicate   bool
          usd_amount  float64 (corrected per R4)
 
        FX rates for this dataset (treat as ground truth):
          EUR→USD: 1.085,  GBP→USD: 1.265,  JPY→USD: 0.0067,  CAD→USD: 0.735
 
        Submit with action type='submit'.
    """).strip(),
}