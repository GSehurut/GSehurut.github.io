---
layout: post
title: "Data Engineering for Insurance Analytics: Principles That Survive Contact With Bordereaux"
date: 2024-03-22
description: "Idempotent loads, explicit grain, validation at the boundary, reconciliation and lineage: the data engineering habits that make actuarial and underwriting numbers trustworthy."
categories: [Data Engineering, Insurance]
tags: [Data Engineering, ETL, Data Quality, Bordereaux, Python, SQL]
---

Insurance data is rarely "big" in the way web data is big. It is *awkward*. It arrives as spreadsheets from coverholders, as extracts from policy administration systems, as bordereaux in a dozen slightly different layouts, and as manual adjustments that someone made for a good reason three years ago. The hard part is not volume; it is making sure the number an actuary or underwriter sees is right, explainable and reproducible.

These are the principles I have found matter most.

## 1. Define the grain before writing any code

The grain of a table is what one row represents. "One row per policy" sounds clear until you meet layered placements, endorsements, multiple sections, multiple currencies and several participating syndicates on the same risk. Many reporting errors come from joining two tables at different grains and silently duplicating premium.

Write the grain down for every table, for example: *one row per policy section, per layer, per transaction, per currency*. Then enforce it with a uniqueness test on those columns.

## 2. Make every load idempotent

An idempotent pipeline produces the same result whether it runs once or five times. This matters because pipelines fail halfway, get rerun by hand, and receive corrected files for a period that has already been loaded.

Practical patterns:

- Load by **partition** (for example by reporting period and source) and replace the whole partition, rather than appending.
- Use `MERGE` on a natural key instead of blind inserts.
- Keep raw files immutable and reprocess from them, so a bug fix can be replayed over history.

## 3. Validate at the boundary

Bad data is cheapest to catch the moment it arrives. A bordereau that is missing its inception dates, or has premium in the wrong currency, should fail loudly before it reaches a dashboard.

A lightweight validation step in pandas goes a long way:

```python
import pandas as pd

REQUIRED = ["policy_ref", "inception_date", "expiry_date", "currency", "gross_premium"]
VALID_CCY = {"GBP", "USD", "EUR", "CAD", "AUD", "JPY"}

def validate_bordereau(df: pd.DataFrame, source: str) -> pd.DataFrame:
    errors = []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")

    if df["policy_ref"].isna().any():
        errors.append("blank policy_ref")

    dupes = df.duplicated(subset=["policy_ref", "section"], keep=False)
    if dupes.any():
        errors.append(f"{dupes.sum()} duplicate policy/section rows")

    bad_dates = df["expiry_date"] <= df["inception_date"]
    if bad_dates.any():
        errors.append(f"{bad_dates.sum()} rows with expiry on or before inception")

    bad_ccy = ~df["currency"].isin(VALID_CCY)
    if bad_ccy.any():
        errors.append(f"unknown currencies: {sorted(df.loc[bad_ccy, 'currency'].unique())}")

    if errors:
        raise ValueError(f"{source}: " + "; ".join(errors))
    return df
```

Libraries such as pandera or Great Expectations formalise this with declarative schemas and readable reports, and are worth adopting once the number of checks grows. The principle is the same: reject or quarantine bad input with a clear message, never silently coerce it.

For delegated authority business, the Lloyd's Coverholder Reporting Standards define the expected fields for risk and claims bordereaux. Even where a coverholder does not follow them exactly, mapping incoming files to that standard gives you a stable internal shape.

## 4. Reconcile to something the business trusts

Validation checks that data is internally consistent. Reconciliation checks that it agrees with the outside world. Every important dataset should reconcile to a known control total: written premium to the finance ledger, paid claims to the claims system, policy counts to the underwriting system.

Automate the reconciliation, store the result alongside the data, and show it on the report. A dashboard that says "reconciles to ledger within 0.1%" earns far more trust than one that does not mention it.

## 5. Keep history explicitly

Actuarial work constantly asks "what did we know at the time?". Reserving uses data as at each quarter end; pricing reviews compare this year's assumptions with last year's. If your tables only hold the current state, those questions become impossible to answer.

Two common approaches:

- **Snapshots:** store a full copy of key tables at each reporting date. Simple and very easy to query.
- **Slowly changing dimensions (Type 2):** keep a row per version with `valid_from` and `valid_to` dates. More compact, but queries need care.

The data modelling post goes deeper on this, including the difference between when something happened and when we recorded it.

## 6. Separate business rules from plumbing

Rules such as "how to convert to settlement currency", "which rate of exchange to use" or "how to allocate premium across sections" should live in one well tested place, not be repeated in five SQL scripts and three notebooks. Put them in pure functions or in a single curated layer, write unit tests for them, and have everything else call that.

## 7. Make lineage visible

When an underwriter asks why a number changed, you need to be able to trace it back: which file, which load, which version of the transformation. At minimum, stamp every row with the source file, the load timestamp and the pipeline version. Platform tools can add richer lineage on top, but these three columns answer most questions.

## 8. Treat pipelines as software

Version control, code review, automated tests and CI/CD are not just for application developers. A pricing dataset that feeds underwriting decisions deserves the same discipline as a production web service. Reis and Housley (2022) describe this as treating data as a product with owners, contracts and service levels, which is a useful framing for actuarial data.

## A short checklist

- [ ] Grain written down and enforced with a uniqueness test
- [ ] Loads are idempotent and can be rerun safely
- [ ] Inputs validated with clear, actionable error messages
- [ ] Outputs reconciled to a trusted control total
- [ ] History kept so "as at" questions can be answered
- [ ] Business rules in one tested place
- [ ] Source, load time and version recorded on every row
- [ ] Code reviewed, tested and deployed through CI

## References

- Reis, J. and Housley, M. (2022). *Fundamentals of Data Engineering*. O'Reilly Media.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media.
- Kimball, R. and Ross, M. (2013). *The Data Warehouse Toolkit*, 3rd edition. Wiley.
- Lloyd's. *Coverholder Reporting Standards*. Lloyd's of London.
- pandera documentation. pandera.readthedocs.io
