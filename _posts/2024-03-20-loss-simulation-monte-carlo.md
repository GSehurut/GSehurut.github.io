---
layout: post
title: "Monte Carlo Loss Simulation for Specialty Insurance, From First Principles"
date: 2024-03-20
description: "Build a frequency and severity simulation in NumPy, apply an excess of loss layer, read OEP and AEP curves correctly, and know how many simulations you actually need."
categories: [Actuarial, Risk Modelling]
tags: [Monte Carlo, Collective Risk Model, Reinsurance, Python, NumPy]
---

Monte Carlo simulation is the workhorse behind capital models, reinsurance pricing and catastrophe analytics. The idea is simple: if we can describe how many losses happen and how big each one is, we can generate thousands of possible years and read off whatever statistic we need. The details are where most mistakes happen, so this post builds a small but correct simulation and points out the traps along the way.

## The collective risk model

The standard starting point is the collective risk model. Annual aggregate loss is

**S = X<sub>1</sub> + X<sub>2</sub> + ... + X<sub>N</sub>**

where *N* is the number of losses in the year and *X<sub>1</sub>, X<sub>2</sub>, ...* are individual loss sizes, assumed independent of each other and of *N*. This is covered in depth in Klugman, Panjer and Willmot's *Loss Models*, and it is the right mental model even when the production version is far more elaborate.

Two modelling choices matter most:

- **Frequency.** Poisson is the default, but specialty data is usually *overdispersed* (variance greater than mean) because the underlying rate itself varies from year to year. A negative binomial captures that.
- **Severity.** Large commercial losses are heavy tailed. A lognormal is a reasonable body; a Pareto or generalised Pareto is often better above a threshold.

## A minimal, vectorised simulation

The code below simulates 100,000 years. It avoids a Python loop over years by drawing every loss at once and then summing them back into their years with `np.add.at`.

```python
import numpy as np

rng = np.random.default_rng(seed=2024)  # always seed, so results are reproducible

n_years = 100_000

# Frequency: negative binomial with mean 12 and variance 24 (overdispersed)
mean_n, var_n = 12.0, 24.0
p = mean_n / var_n
r = mean_n * p / (1 - p)
counts = rng.negative_binomial(r, p, size=n_years)

# Severity: lognormal body with a mean of about 1m
mu, sigma = np.log(1_000_000) - 0.5 * 1.2**2, 1.2
losses = rng.lognormal(mu, sigma, size=counts.sum())

# Map every loss back to the year it belongs to
year_index = np.repeat(np.arange(n_years), counts)

# Gross annual aggregate loss
gross = np.zeros(n_years)
np.add.at(gross, year_index, losses)
```

## Applying an excess of loss layer

A per-risk excess of loss layer "5m xs 2m" pays the part of each loss between 2m and 7m. Applying it *per loss before aggregating* is the step people most often get wrong; applying it to the annual total gives an aggregate stop loss instead, which is a different contract.

```python
attachment, limit = 2_000_000, 5_000_000

ceded_per_loss = np.clip(losses - attachment, 0, limit)

ceded = np.zeros(n_years)
np.add.at(ceded, year_index, ceded_per_loss)

net = gross - ceded
```

If the layer has an annual aggregate limit or reinstatements, apply those to `ceded` per year after this step.

## Reading the results: AEP versus OEP

Two exceedance curves come out of a simulation like this, and they answer different questions:

- **AEP (aggregate exceedance probability):** the probability that *total* annual loss exceeds a level. Relevant for capital and aggregate covers.
- **OEP (occurrence exceedance probability):** the probability that the *largest single* loss in a year exceeds a level. Relevant for per-occurrence reinsurance.

```python
# Largest single loss per year (years with no losses have a maximum of 0)
max_loss = np.zeros(n_years)
np.maximum.at(max_loss, year_index, losses)

def return_period_loss(annual_values, return_period):
    """Loss level exceeded on average once every `return_period` years."""
    return np.quantile(annual_values, 1 - 1 / return_period)

def tvar(annual_values, level):
    """Average of outcomes beyond the `level` quantile (Tail Value at Risk)."""
    threshold = np.quantile(annual_values, level)
    return annual_values[annual_values >= threshold].mean()

for rp in (10, 50, 100, 200):
    print(
        f"1-in-{rp:<3}  AEP gross {return_period_loss(gross, rp)/1e6:6.1f}m"
        f"   OEP gross {return_period_loss(max_loss, rp)/1e6:6.1f}m"
        f"   AEP net {return_period_loss(net, rp)/1e6:6.1f}m"
    )

print(f"Average annual loss (gross): {gross.mean()/1e6:.2f}m")
print(f"99.5% TVaR (net):            {tvar(net, 0.995)/1e6:.2f}m")
```

The 1 in 200 AEP is the 99.5th percentile of annual loss, which is the confidence level used in Solvency II style capital calculations. TVaR averages the whole tail beyond that point and is a coherent risk measure in the sense of Artzner et al. (1999), which VaR is not.

## How many simulations is enough?

Simulation error is often ignored, but it is easy to estimate. For a mean, the standard error falls with the square root of the number of simulations:

```python
se_aal = gross.std(ddof=1) / np.sqrt(n_years)
print(f"AAL = {gross.mean()/1e6:.3f}m +/- {1.96*se_aal/1e6:.3f}m (95% interval)")
```

Tail quantiles converge much more slowly than the mean because only a handful of simulations land beyond them. With 100,000 years, only 500 sit beyond the 1 in 200 point. A quick, honest check is to rerun with several seeds and look at the spread of the quantile you care about. If a 1 in 200 figure moves by more than your materiality threshold between seeds, you need more simulations or a variance reduction technique; Glasserman (2003) covers importance sampling and control variates in detail.

## Common pitfalls

- **Not seeding the generator.** Results that change on every run cannot be reviewed or reconciled. Use `np.random.default_rng(seed)` and record the seed.
- **Using the legacy NumPy API.** `np.random.seed` and global state make parallel runs hard to reproduce. The `Generator` API is the current recommendation in the NumPy documentation.
- **Applying per-risk terms to aggregates.** As above, this silently changes the contract.
- **Forgetting dependence.** The collective model assumes independence. Catastrophe events, inflation and the underwriting cycle all create dependence between losses and between lines; the next post on copulas looks at one way to handle that.
- **Reporting false precision.** If your parameters come from ten years of data, the parameter uncertainty dwarfs the simulation error. Say so.

## Where this goes next

This simple model already supports useful work: comparing reinsurance structures, sizing retentions and explaining tail risk to underwriters. Extensions worth adding, roughly in order of value, are parameter uncertainty (simulate the parameters as well as the losses), inflation and exposure trends, reinstatement terms, and dependence between lines.

## References

- Klugman, S. A., Panjer, H. H. and Willmot, G. E. (2019). *Loss Models: From Data to Decisions*, 5th edition. Wiley.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Artzner, P., Delbaen, F., Eber, J.-M. and Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203 to 228.
- Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern Actuarial Risk Theory: Using R*, 2nd edition. Springer.
- NumPy documentation. *Random Generator*. numpy.org/doc/stable/reference/random/generator.html
