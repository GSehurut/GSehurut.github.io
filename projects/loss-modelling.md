---
layout: project
title: "loss-modelling"
date: 2026-11-12
order: 2
status: Active
repo: GSehurut/loss-modelling
description: "Claims development triangles, chain ladder projections and Mack reserve uncertainty in plain, readable Python, validated against published benchmarks."
categories: [Reserving, Actuarial]
tags: [Python, pandas, Chain Ladder, Mack, Open Source]
---

## Why it exists

Reserving is central to every insurer's balance sheet, yet the methods are often locked in spreadsheets or large libraries that are hard to read. loss-modelling keeps each method as a short, referenced function, so actuaries, students and data engineers can see exactly how a reserve is calculated, and trust it because the tests reproduce published results.

## What it does today

- A `Triangle` built from incremental arrays or straight from **claim movements** (for example the output of a SQL query)
- **Chain ladder** with volume weighted development factors
- **Mack (1993)** standard errors for each origin year and the total reserve
- The **Taylor and Ashe (1983)** benchmark triangle, with tests that match Mack's published figures exactly

```python
from loss_modelling import mack, taylor_ashe

result = mack(taylor_ashe())
print(f"{result.total_reserve:,.0f} +/- {result.total_std_error:,.0f}")
# 18,680,856 +/- 2,447,095
```

## Where it is going

- Bornhuetter-Ferguson and Cape Cod
- Tail factors and factor exclusions
- Bootstrap ODP for full reserve distributions
- Helpers for computing development periods from dates

The full list is in the [roadmap](https://github.com/GSehurut/loss-modelling/blob/main/ROADMAP.md).

## Get involved

Good first contributions include alternative averaging methods, more public benchmark triangles and paid versus incurred views. Read the [contributing guide](https://github.com/GSehurut/loss-modelling/blob/main/CONTRIBUTING.md) to get started.

Background: [Data modelling for insurance analytics]({% post_url 2026-10-05-data-modelling-insurance-analytics %}), which shows how to build these triangles from claim movements.
