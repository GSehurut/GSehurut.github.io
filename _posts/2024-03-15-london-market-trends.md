---
layout: post
title: "Emerging Risks in the London Specialty Market: What the Data Can and Can't Tell Us"
date: 2024-03-15
description: "Cyber, war and political violence, secondary natural perils and accumulation: how emerging risks show up in London market data, and why analysts should be humble about history."
categories: [Insurance, Risk]
tags: [London Market, Specialty Insurance, Emerging Risk, Cyber, Exposure Management]
---

The London market exists to insure risks that are large, unusual or hard to price anywhere else. That makes it the place where emerging risks show up first, and it also makes it the place where historical data is least reliable as a guide to the future. This post looks at four themes that matter for specialty portfolios and, more importantly, at what each one means for the people building the data and models behind underwriting decisions.

## Why "emerging" is a data problem

Most actuarial techniques assume the past is a reasonable guide to the future once you adjust for inflation, exposure and mix. Emerging risks break that assumption in three ways:

1. **Short or no history.** Cyber has only a couple of decades of meaningful loss data, and the threat changes faster than the data accumulates.
2. **Changing contract wordings.** When the market rewrites exclusions, past losses stop being comparable with future exposure, even if the underlying hazard is unchanged.
3. **Accumulation over independence.** Many of these risks are dangerous because one event hits many policies at once. Models built on independent claims understate the tail.

The practical consequence is that exposure data, scenario analysis and clear documentation of assumptions become more valuable than another decimal place of precision on a fitted distribution.

## 1. Cyber and systemic accumulation

Cyber is the textbook accumulation risk: a single vulnerability in widely used software or a cloud outage can trigger losses across thousands of insureds in different territories and classes. The market response has been as much about wording as pricing. Lloyd's Market Bulletin Y5381 (August 2022) required managing agents to include clauses excluding losses from state-backed cyber attacks in standalone cyber policies from March 2023, and the Lloyd's Market Association published model clauses (LMA5564 to LMA5567) to support that.

For an analyst, this has concrete implications:

- **Capture the wording version at policy level.** If you cannot tell which exclusion a policy carries, you cannot compare its exposure with last year's book.
- **Model scenarios, not just distributions.** Realistic Disaster Scenarios for cyber (for example a major cloud provider outage) are a better communication tool for underwriters and boards than a fitted severity curve.
- **Track non-affirmative ("silent") cyber** in property and casualty lines, where cyber exposure exists without being explicitly priced.

## 2. War, political violence and political risk

The aircraft lessor claims arising from aircraft stranded in Russia after 2022 were a reminder that political risk and aviation war exposures can crystallise suddenly and at very large scale, and that the question of *which* policy responds (all risks or war risks, contingent or possessed) can matter as much as the size of the loss.

What this means for data:

- **Exposure by country and counterparty must be current.** Quarterly snapshots are often too slow when a sanctions regime changes in a week.
- **Aggregation needs a clear hierarchy.** A single obligor or government can sit behind many policies through different brokers and coverholders. Without consistent entity resolution, the same exposure is counted under several names or missed entirely.
- **Scenario definitions should be versioned** so that a figure reported to a risk committee can be reproduced months later.

## 3. Natural catastrophe: the rise of secondary perils

Peak perils like US hurricane and Japanese earthquake dominate catastrophe model output, but much of the insured loss in recent years has come from so-called secondary perils: severe convective storms, flooding and wildfire. The Swiss Re Institute's annual *sigma* reports on natural catastrophes have repeatedly highlighted this trend, alongside the effect of rising exposure values in hazard-prone areas.

For a specialty book this matters in a few ways:

- Secondary perils are often less well represented in vendor catastrophe models, so **model adjustments and loadings need to be explicit and documented**.
- **Geocoding quality drives everything.** A risk located to postcode centroid rather than building coordinates can land on the wrong side of a flood zone boundary.
- **Exposure growth** (more insured value in the same place) can look like a trend in frequency if you only analyse loss counts.

## 4. Supply chains and contingent exposures

Marine cargo, contingent business interruption and trade credit share a common feature: the loss often occurs far from the insured. A port closure, a canal blockage or a regional conflict can generate claims across cargo, hull, war and political risk policies simultaneously. Capturing *where* goods are and *which routes* they take is hard, and most portfolios rely on declared values, voyage-level bordereaux or storage location schedules of mixed quality.

## What good looks like for analytics teams

Across all four themes, the same habits make a portfolio easier to understand:

| Habit | Why it matters |
| --- | --- |
| Policy-level wording and exclusion flags | Makes year-on-year exposure comparisons valid |
| Clean entity and geography reference data | Prevents double counting and missed aggregations |
| Versioned scenario definitions | Makes committee figures reproducible |
| Separate "exposure growth" from "hazard change" | Avoids reading inflation and growth as trend |
| Documented model adjustments | Keeps expert judgement visible and reviewable |

None of these are glamorous, and none require machine learning. They do require good data engineering and a close working relationship between analysts, exposure managers and underwriters.

## Closing thought

The temptation with emerging risks is to reach for a more sophisticated model. In my experience, the bigger gains come from knowing exactly what you have written, where it is, and which wording applies. Sophisticated models built on uncertain exposure data produce confident looking numbers that are not worth very much.

## References

- Lloyd's (2022). *Market Bulletin Y5381: State backed cyber-attack exclusions*. Lloyd's of London.
- Lloyd's Market Association (2021). *Cyber war and cyber operation exclusion clauses, LMA5564 to LMA5567*.
- Swiss Re Institute. *sigma: Natural catastrophes* (annual series). Swiss Re.
- Lloyd's. *Realistic Disaster Scenarios: Scenario specification* (published annually). Lloyd's of London.
- Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern Actuarial Risk Theory: Using R*, 2nd edition. Springer.
