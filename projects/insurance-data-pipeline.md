---
layout: project
title: "insurance-data-pipeline"
date: 2026-11-12
order: 3
status: Active
repo: GSehurut/insurance-data-pipeline
description: "Map, validate and standardise insurance bordereaux: canonical schemas, per-source column mappings, row level issue reports and synthetic test data."
categories: [Data Engineering, Insurance]
tags: [Python, pandas, Data Quality, Bordereaux, Open Source]
---

## Why it exists

Every coverholder sends bordereaux in its own layout, with different column names, date formats and quirks. Most teams handle this with one-off scripts per source, which break quietly when a file changes. This project provides one consistent way to map, validate and report on incoming data, so problems are caught at the boundary and sent back to the source with a clear explanation.

## What it does today

- A **canonical risk bordereau schema** with an explicit grain
- **YAML column mappings** per source, tolerant of case and whitespace differences
- **Row level validation** that reports every problem: types, required values, currencies, countries, date order and duplicates
- **Currency conversion** with explicit rates that fails loudly on missing currencies
- **Synthetic bordereaux** with realistic injected errors, for testing and teaching
- A command line tool: `idp validate` and `idp generate`

```sh
idp generate bordereau.csv --rows 300 --errors 0.1 --seed 7
idp validate bordereau.csv --mapping examples/coverholder_mapping.yml --issues issues.csv
```

## Where it is going

- Claims and premium transaction schemas
- Mapping to Lloyd's Coverholder Reporting Standards field names
- Reconciliation to control totals
- A Microsoft Fabric bronze to silver example with idempotent Delta merges

The full list is in the [roadmap](https://github.com/GSehurut/insurance-data-pipeline/blob/main/ROADMAP.md).

## Get involved

Anyone who has wrestled with a bordereau has something to offer. Good first contributions include a validation report in HTML or Markdown, reconciliation helpers and more realistic synthetic error patterns. Read the [contributing guide](https://github.com/GSehurut/insurance-data-pipeline/blob/main/CONTRIBUTING.md) to get started.

Background: [Data engineering for insurance analytics]({% post_url 2024-03-22-data-engineering-insurance %}), [Working with data in the London market]({% post_url 2026-09-14-working-with-london-market-data %}) and [Working with Microsoft Fabric]({% post_url 2026-10-19-working-with-microsoft-fabric %}).
