---
layout: post
title: "Copulas for Insurance Dependence: A Practical Guide and Its Pitfalls"
date: 2024-03-21
description: "Why correlation is not enough for tail risk, how Gaussian and t copulas differ, how to simulate and calibrate them in Python, and the warnings every analyst should know."
categories: [Actuarial, Risk Modelling]
tags: [Copulas, Dependence, Tail Risk, Python, SciPy]
---

The previous post simulated losses as if every line of business were independent. In reality, a hurricane hits property and marine cargo together, a recession hits credit and political risk together, and inflation pushes up claims everywhere at once. Copulas are a standard way to add that dependence to a simulation without giving up the marginal distributions you have already fitted for each line.

## Why correlation alone is not enough

Linear (Pearson) correlation is a single number, and many very different joint behaviours share the same correlation. Embrechts, McNeil and Straumann (2002) set out the problems clearly:

- Correlation only fully describes dependence for elliptical distributions such as the multivariate normal.
- It is not invariant to non-linear transformations, so the correlation between losses differs from the correlation between their logarithms.
- For a given pair of marginals, not every correlation between -1 and 1 is achievable.
- Most importantly for insurance, two models with the same correlation can have completely different behaviour in the **tail**, which is exactly where capital and reinsurance decisions are made.

## What a copula is

Sklar's theorem (1959) says that any joint distribution can be split into its marginal distributions and a **copula** that describes how they move together. In practice that gives a clean workflow:

1. Fit each line's loss distribution on its own (the marginals).
2. Choose and calibrate a copula for the dependence.
3. Simulate correlated uniforms from the copula and push them through each marginal's inverse CDF.

Rank based measures such as Kendall's tau and Spearman's rho depend only on the copula, not the marginals, which makes them the natural tools for calibration.

## Gaussian versus Student t: the tail matters

The two copulas you meet most often are the Gaussian and the Student t. With the same correlation parameter they look similar in the middle of the distribution, but they behave very differently in the extremes.

The **coefficient of upper tail dependence** measures the probability that one variable is extreme given that the other is extreme, in the limit. For the Gaussian copula it is **zero** for any correlation below 1: extremes eventually become independent. For the t copula it is positive and grows as the degrees of freedom fall. McNeil, Frey and Embrechts (2015) give the formulas and derivations.

That single fact explains why a Gaussian copula can understate the chance of two lines having a bad year together.

## Simulating both in Python

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=7)
n = 200_000
rho = 0.5
corr = np.array([[1.0, rho], [rho, 1.0]])
chol = np.linalg.cholesky(corr)

# Gaussian copula: correlated normals mapped to uniforms
z = rng.standard_normal((n, 2)) @ chol.T
u_gauss = stats.norm.cdf(z)

# Student t copula with 4 degrees of freedom:
# same correlated normals, divided by a shared chi-square mixing variable
nu = 4
w = rng.chisquare(nu, size=(n, 1)) / nu
t_samples = z / np.sqrt(w)
u_t = stats.t.cdf(t_samples, df=nu)

def joint_exceedance(u, q):
    """Probability that both variables exceed their q-quantile."""
    return np.mean((u[:, 0] > q) & (u[:, 1] > q))

for q in (0.9, 0.99, 0.995):
    print(
        f"q={q}: independent {(1-q)**2:.5f}  "
        f"gaussian {joint_exceedance(u_gauss, q):.5f}  "
        f"t(4) {joint_exceedance(u_t, q):.5f}"
    )
```

Run it and you will see the three columns agree reasonably at the 90th percentile, then diverge sharply at 99.5%, with the t copula producing several times more joint extremes than the Gaussian. Same correlation, very different capital.

## Turning uniforms into losses

Once you have dependent uniforms, apply each line's fitted marginal:

```python
# Marginals fitted separately for each line (illustrative parameters)
property_cat = stats.lognorm(s=1.4, scale=8e6)
marine_cargo = stats.genpareto(c=0.3, scale=2e6)

losses_property = property_cat.ppf(u_t[:, 0])
losses_marine = marine_cargo.ppf(u_t[:, 1])
portfolio = losses_property + losses_marine

print(f"1-in-200 portfolio loss: {np.quantile(portfolio, 0.995)/1e6:.1f}m")
```

Swap `u_t` for `u_gauss` and compare. The difference is a direct measure of how much your choice of copula is driving the tail.

## Calibrating the dependence

For elliptical copulas there is a convenient link between Kendall's tau and the correlation parameter: rho = sin(pi * tau / 2). Calibrating via rank correlation is more robust than using Pearson correlation on raw losses:

```python
def rho_from_kendall(x, y):
    tau, _ = stats.kendalltau(x, y)
    return np.sin(np.pi * tau / 2)
```

The degrees of freedom of a t copula are harder to estimate and usually need maximum likelihood on pseudo-observations (ranks scaled into the unit interval). With the short histories typical of specialty lines, expect the estimate to be uncertain and treat it as a judgement to be documented and stress tested, not a fact.

## Pitfalls to take seriously

- **Data is thin in the tail.** You are trying to estimate joint extreme behaviour from a handful of joint extreme years. Expert judgement and scenarios should inform the parameters, and sensitivity testing should show how much they matter.
- **The Gaussian copula's reputation.** Li's (2000) Gaussian copula model for default correlation became infamous after the financial crisis, largely because its lack of tail dependence and the over-confidence in its calibration were not appreciated. Mikosch (2006) is a sharp critique of how copulas are sometimes oversold.
- **Dependence is not constant.** Lines can look independent in normal years and move together in stress. A single copula fitted to all years can hide this.
- **Higher dimensions.** A single t copula imposes the same tail behaviour on every pair of lines. Vine copulas (Joe, 2014) allow more flexible pairwise structures at the cost of more parameters to justify.

## Takeaways

Copulas are a useful tool for one specific job: combining well understood marginals with a defensible dependence structure. They are not a substitute for thinking about *why* lines are dependent. In practice the most valuable output is often the comparison itself: showing underwriters and management how much the tail moves when dependence assumptions change.

## References

- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229 to 231.
- Embrechts, P., McNeil, A. and Straumann, D. (2002). Correlation and dependence in risk management: properties and pitfalls. In *Risk Management: Value at Risk and Beyond*, Cambridge University Press.
- McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*, revised edition. Princeton University Press.
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd edition. Springer.
- Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman and Hall/CRC.
- Li, D. X. (2000). On default correlation: a copula function approach. *Journal of Fixed Income*, 9(4), 43 to 54.
- Mikosch, T. (2006). Copulas: tales and facts. *Extremes*, 9, 3 to 20.
