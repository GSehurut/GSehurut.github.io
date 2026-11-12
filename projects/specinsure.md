---
layout: project
title: "specinsure"
date: 2026-11-12
order: 1
status: Active
repo: GSehurut/specinsure
description: "A Python toolkit for simulating specialty insurance losses and the reinsurance structures that protect them, using transparent Monte Carlo methods."
categories: [Reinsurance, Risk Modelling]
tags: [Python, NumPy, SciPy, Monte Carlo, Open Source]
---

## Why it exists

Reinsurance decisions, such as where to set a retention or whether a layer is worth its price, rest on simulation. Commercial tools do this well but are closed and expensive, and most open examples online apply contract terms incorrectly. specinsure aims to be a small, readable and well tested reference that anyone can learn from, check and extend.

## What it does today

- Frequency and severity simulation into a **year loss table**, using any `scipy.stats` distribution
- **Excess of loss** treaties (per occurrence or annual aggregate, with an optional aggregate limit) and **quota share**
- Annual terms applied **year by year** across the simulation, which is where many implementations go wrong
- **Risk metrics:** average annual loss, VaR, TVaR, return period losses, AEP and OEP curves

```python
from scipy import stats
from specinsure import ExcessOfLossTreaty, simulate, metrics

ylt = simulate(stats.nbinom(n=12, p=0.5), stats.lognorm(s=1.2, scale=1_000_000),
               n_years=50_000, seed=42)
ceded = ylt.ceded(ExcessOfLossTreaty(attachment=2_000_000, limit=5_000_000))
net_annual = ylt.annual_totals(ylt.losses - ceded)
print(metrics.return_period_loss(net_annual, 200))
```

## Where it is going

- Reinstatements, surplus treaties and stacked programmes
- Parameter uncertainty, inflation trends and copula dependence between lines
- Plot helpers and worked examples comparing reinsurance structures

The full list is in the [roadmap](https://github.com/GSehurut/specinsure/blob/main/ROADMAP.md).

## Get involved

You don't need to be an actuary. Good first contributions include examples, plotting helpers, and the standard error for average annual loss. Read the [contributing guide](https://github.com/GSehurut/specinsure/blob/main/CONTRIBUTING.md), then pick an issue.

Background: [Monte Carlo loss simulation for specialty insurance]({% post_url 2024-03-20-loss-simulation-monte-carlo %}) and [Copulas for insurance dependence]({% post_url 2024-03-21-copula-dependency-modeling %}).
