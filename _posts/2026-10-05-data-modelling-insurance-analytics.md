---
layout: post
title: "Data Modelling for Insurance Analytics: Grain, Stars and Time"
date: 2026-10-05
description: "A dimensional model for premium and claims, why transaction facts beat snapshots of totals, slowly changing dimensions, and how to build reserving triangles from claim movements."
categories: [Data Engineering, Insurance]
tags: [Data Modelling, Dimensional Modelling, Star Schema, SQL, Reserving, Kimball]
---

Most problems I see in insurance reporting are not caused by bad SQL. They are caused by a data model that does not match the questions being asked. A good model makes the right answer the easy answer: premium cannot be double counted, history is available when you need it, and a reserving triangle is a simple query rather than a week of spreadsheet work.

This post walks through a dimensional model for premium and claims analytics, following the approach in Kimball and Ross's *The Data Warehouse Toolkit*, and adapted for specialty insurance.

## Start from the questions

Before drawing any tables, list the questions the model must answer. For a specialty portfolio they usually include:

- What premium have we written and earned, by line, year of account, broker and territory?
- How have incurred claims developed over time for each underwriting year?
- What is our exposure, at our share, to a given country, insured or scenario?
- How did any of the above look *as at* last quarter end?

Each question tells you something about the grain, the dimensions and the history you need.

## Facts and dimensions

A dimensional model separates **facts** (measurable events, usually numeric) from **dimensions** (the descriptive context you slice by).

A sensible core for specialty insurance:

**Facts**
- `fact_premium_transaction`: one row per premium movement, per policy section, per layer, per currency.
- `fact_claim_movement`: one row per claim payment or reserve change, per claim, per currency.
- `fact_exposure_snapshot`: one row per risk, per reporting date, with limits and sums insured at our share.

**Dimensions**
- `dim_policy` (with section and layer attributes), `dim_class` (line of business hierarchy), `dim_broker`, `dim_insured`, `dim_geography`, `dim_currency`, and a `dim_date` used in several roles (transaction date, inception date, loss date).

```
                 dim_date (as transaction / inception / loss date)
                        |
dim_broker --- fact_premium_transaction --- dim_policy --- dim_class
                        |
                  dim_currency
```

## Why transactions beat totals

It is tempting to store one row per policy with "total premium" and "total incurred" columns. It feels simpler, but it throws away the information you need most:

- You cannot see **when** amounts changed, so you cannot build development triangles.
- You cannot reproduce last quarter's figures once totals are overwritten.
- Corrections overwrite history rather than appearing as visible adjustments.

Storing **movements** (each payment, each reserve change, each additional premium) keeps all of that. Totals at any date are then just a sum of movements up to that date:

```sql
-- Incurred to date, as at any chosen valuation date
SELECT
    c.claim_id,
    SUM(m.paid_amount)    AS paid_to_date,
    SUM(m.reserve_change) AS outstanding,
    SUM(m.paid_amount + m.reserve_change) AS incurred
FROM fact_claim_movement AS m
JOIN dim_claim AS c ON c.claim_key = m.claim_key
WHERE m.transaction_date <= @valuation_date
GROUP BY c.claim_id;
```

## Building a triangle from movements

With claim movements in place, a reserving triangle is a straightforward aggregation. Development period is the gap between the start of the underwriting year and the transaction date:

```sql
WITH movements AS (
    SELECT
        p.underwriting_year,
        -- development quarter, 1 = first quarter of the underwriting year
        (YEAR(m.transaction_date) - p.underwriting_year) * 4
            + DATEPART(QUARTER, m.transaction_date) AS dev_quarter,
        m.paid_amount + m.reserve_change           AS incurred_movement
    FROM fact_claim_movement AS m
    JOIN dim_policy AS p ON p.policy_key = m.policy_key
    WHERE m.transaction_date <= @valuation_date
)
SELECT
    underwriting_year,
    dev_quarter,
    SUM(SUM(incurred_movement)) OVER (
        PARTITION BY underwriting_year ORDER BY dev_quarter
        ROWS UNBOUNDED PRECEDING
    ) AS cumulative_incurred
FROM movements
GROUP BY underwriting_year, dev_quarter
ORDER BY underwriting_year, dev_quarter;
```

Pivot the result and you have the cumulative incurred triangle that chain ladder methods, including Mack's (1993) distribution free model for reserve uncertainty, start from. Because the triangle is derived from movements rather than stored, you can rebuild it at any past valuation date simply by changing `@valuation_date`.

(The date functions above are T-SQL; Spark SQL uses `year()` and `quarter()` but the logic is identical.)

## Slowly changing dimensions

Dimension attributes change: a broker is acquired, a policy is reassigned to a different class, a country's risk rating changes. The question is whether reports should show the attribute as it is *now* or as it was *at the time*.

Kimball's slowly changing dimension types give you the vocabulary:

- **Type 1:** overwrite. Reports always use the current value. Fine for corrections.
- **Type 2:** add a new row with `valid_from`, `valid_to` and `is_current`. Reports can use the value at the time of each fact.

```sql
-- Type 2 lookup: the broker attributes that applied on each transaction date
SELECT f.*, b.broker_group
FROM fact_premium_transaction AS f
JOIN dim_broker AS b
  ON  b.broker_id = f.broker_id
  AND f.transaction_date >= b.valid_from
  AND f.transaction_date <  b.valid_to;
```

For most insurance hierarchies (class of business, broker group, territory), Type 2 is the safer default, because restating history without warning is exactly what makes figures fail to reconcile between quarters.

## Two kinds of time

Insurance data has two independent timelines:

- **Valid time:** when something was true in the real world (the loss occurred on 3 March).
- **Transaction time:** when we recorded it (we were told on 20 April and booked it on 22 April).

Most of the time, storing transaction dates on movements is enough. But some questions, like "what did we *believe* the exposure was on 31 December, before late bordereaux arrived?", need both. Snodgrass's work on time oriented databases describes this **bitemporal** approach in detail. You do not need it everywhere, but it is worth recognising when a question needs it.

## Conformed dimensions

A dimension is **conformed** when every fact table uses the same version of it. If premium and claims both join to the same `dim_class`, a loss ratio by class is just two aggregations side by side. If premium uses one class mapping and claims another, every loss ratio becomes a reconciliation exercise. Conformed dimensions are the cheapest way to make figures agree across reports.

## Practical guidelines

1. **Write the grain of every fact table** in its description, and test uniqueness on it.
2. **Store movements, derive totals.** History comes for free.
3. **Use surrogate keys** for dimensions so Type 2 history works and source system changes do not break joins.
4. **Conform dimensions** across premium, claims and exposure.
5. **Keep amounts in original currency** on facts, with conversion applied in views or a semantic layer.
6. **Model for the questions,** and revisit the model when the questions change.

## References

- Kimball, R. and Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling*, 3rd edition. Wiley. (Chapter 16 covers insurance.)
- Snodgrass, R. T. (1999). *Developing Time-Oriented Database Applications in SQL*. Morgan Kaufmann.
- Mack, T. (1993). Distribution-free calculation of the standard error of chain ladder reserve estimates. *ASTIN Bulletin*, 23(2), 213 to 225.
- Friedland, J. (2010). *Estimating Unpaid Claims Using Basic Techniques*. Casualty Actuarial Society.
