---
layout: post
title: "Working With Data in the London Insurance Market: A Field Guide"
date: 2026-09-14
description: "Lines, layers, line sizes, bordereaux, currencies and dates: the concepts that trip up analysts new to London market data, and how to handle them."
categories: [Insurance, Data Engineering]
tags: [London Market, Lloyd's, Bordereaux, Data Quality, Specialty Insurance]
---

If you come to the London market from another data role, the first few months can be disorienting. The data looks like ordinary transactional data, but almost every field carries a subtlety that changes the numbers. This post is the guide I would have liked when I started: the concepts that matter most, and the habits that keep analysis honest.

## How the market shapes the data

The London market, centred on Lloyd's and the company market around it, is built around **subscription**. A large risk is rarely written by a single insurer. A broker places it with a lead underwriter, who sets the terms, and then with following markets who each take a share. The same risk might be split into **layers** (for example the first 10m, then 15m excess of 10m), each with its own panel of insurers.

That structure has direct consequences for data:

- **Line size and share.** Your company's exposure is its *share* of each layer, not the full limit. Always know whether a figure is at 100% or at your share.
- **Signed versus written lines.** Underwriters write a line, but if the placement is oversubscribed the line is signed down. Premium and exposure should use signed lines once they are known.
- **One risk, many rows.** A policy can have multiple sections, layers, currencies and participating entities. Getting the grain wrong is the most common cause of double counted premium.

## Open market versus delegated authority

Business comes in through two broad routes, and the data looks quite different for each.

**Open market** business is placed risk by risk through brokers. Data usually comes from your own policy administration system and from market messaging.

**Delegated authority** business is written by coverholders or managing general agents under a binding authority. You typically learn about individual risks and claims through **bordereaux**: periodic spreadsheets listing the risks bound and claims handled. Bordereaux quality varies enormously, which is why the Lloyd's Coverholder Reporting Standards exist to define the expected fields. Mapping every incoming bordereau into that standard, even approximately, gives you a stable internal shape to build on.

## The market is modernising its data

For years, much of the market's data moved as documents and spreadsheets. Lloyd's Blueprint Two programme aims to change that, with the **Core Data Record (CDR)** defining a standard set of structured data captured at placement, so that downstream processes like accounting, settlement and regulatory reporting can use the same data. The ACORD data standards have long been the reference for insurance messaging and are a useful vocabulary for internal models too.

For analytics teams, the direction of travel is clear: more structured data at the point of placement, and more expectation that internal models line up with market standards. Designing your internal data model with those standards in mind will save painful remapping later.

## Currencies and rates of exchange

London market business is written in many currencies. A single portfolio report might include USD, GBP, EUR, CAD and JPY premium. Decide, and document, three things:

1. **Original currency** is the currency the contract is written in. Keep it.
2. **Settlement currency** is the currency the money actually moves in, which may differ.
3. **Reporting currency** is what you convert to for analysis, and *which rate* you use: a plan rate fixed for the year, the rate on the transaction date, or the closing rate at the reporting date.

Mixing rate bases is a classic source of "the numbers don't match finance". Store original amounts, apply conversion in one tested place, and label every report with its rate basis.

## Dates: there are always more than you think

A single claim or policy has several dates that answer different questions:

| Date | Question it answers |
| --- | --- |
| Inception / expiry | When is the policy on risk? |
| Underwriting year (year of account) | Which year's book does it belong to? |
| Loss date | When did the event happen? |
| Reported date | When did we hear about it? |
| Transaction / booked date | When was the movement recorded? |
| As at date | What did the data look like at a point in time? |

Reserving triangles, for example, are built from underwriting or accident year against development period, using data *as at* successive valuation dates. If you cannot reconstruct what the data looked like at a past date, you cannot rebuild last quarter's triangle to check it. That is why keeping history explicitly (snapshots or effective dated records) is not optional.

## Sign conventions and transaction types

Premium and claims arrive as a stream of transactions: original premium, additional and return premiums, adjustments, payments, reserve movements and recoveries. Agree a sign convention (for example, inwards premium positive, return premium negative) and enforce it at ingestion. A surprising number of reconciliation breaks come from one source treating returns as negative and another as positive with a type flag.

## Practical habits that help

- **State the basis on every number.** 100% or our share; gross or net; original or converted; written or earned; as at which date.
- **Reconcile to finance early and often.** If your premium does not tie to the ledger, nothing built on top will be trusted.
- **Keep reference data clean.** Brokers, coverholders, insureds and countries need consistent identifiers, or aggregation and exposure reporting fall apart.
- **Talk to underwriters and operations.** Most data quirks have a business reason. Someone in the room knows why that field is blank for one class.
- **Write things down.** A short data dictionary with grain, basis and known issues is worth more than another dashboard.

## Closing thought

London market data rewards curiosity. Every odd value usually reflects something real about how a risk was placed, written or settled. The analysts who do best are the ones who learn the business behind the fields, and then build pipelines that make those rules explicit and repeatable.

## References

- Lloyd's. *Blueprint Two* and the *Core Data Record*. lloyds.com
- Lloyd's. *Coverholder Reporting Standards*. Lloyd's of London.
- ACORD. *ACORD Data Standards*. acord.org
- Kimball, R. and Ross, M. (2013). *The Data Warehouse Toolkit*, 3rd edition, chapter on insurance. Wiley.
