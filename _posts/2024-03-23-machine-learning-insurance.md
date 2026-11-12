---
layout: post
title: "Machine Learning in Insurance Pricing: Where It Helps, Where GLMs Still Win"
date: 2024-03-23
description: "A practical view of GLMs and gradient boosting for insurance pricing: Tweedie losses, exposure offsets, explainability, and why specialty lines need a different approach."
categories: [Machine Learning, Actuarial]
tags: [Machine Learning, GLM, Gradient Boosting, Pricing, Explainability, Python]
---

Machine learning in insurance is often described as a revolution. In practice it is more of an evolution: generalised linear models (GLMs) remain the backbone of rating in many markets, and the most useful role for more flexible models is often to *improve* and *challenge* a GLM rather than replace it. This post looks at what each approach does well, how to set up a fair comparison, and why specialty lines need extra care.

## Why GLMs are still everywhere

GLMs became the industry standard for personal lines pricing because they fit the problem well:

- **Right distributions.** Poisson for claim counts, gamma for severity and Tweedie for pure premium match the shape of insurance data.
- **Exposure handled properly.** An offset of log exposure means a six month policy is expected to have half the claims of an annual one.
- **Multiplicative structure.** A log link produces rating factors that underwriters, regulators and rating engines understand directly.
- **Transparency.** Every coefficient can be inspected, challenged and constrained.

Goldburd, Khare, Tevet and Guller's CAS monograph is the standard practical reference.

## Where gradient boosting adds value

Tree based ensembles such as gradient boosting machines (Chen and Guestrin's XGBoost being the best known) are good at finding interactions and non-linear effects that are tedious to specify by hand in a GLM. Henckaerts et al. (2021) showed that boosted trees can produce competitive insurance tariffs and offered ways to extract interpretable insights from them.

Useful applications include:

- **Finding interactions** to add back into a GLM.
- **Challenger models** that show where the production GLM is mispricing.
- **Risk selection and triage**, where the output informs a decision rather than becoming the price directly.
- **Claims analytics**, such as predicting large claim development or flagging files for review.

## Setting up a fair comparison

The comparison only means something if both models are trained on the same target, with the same exposure handling and the same validation scheme. A minimal example with scikit-learn:

```python
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split

# X: rating factors (numeric or one-hot encoded), y: incurred loss, exposure: years at risk
X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
    X, y, exposure, test_size=0.25, random_state=0
)

# Model pure premium (loss per unit exposure), weighting each row by its exposure
glm = TweedieRegressor(power=1.5, link="log", alpha=1e-4, max_iter=1000)
glm.fit(X_train, y_train / e_train, sample_weight=e_train)

gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=500, learning_rate=0.05)
gbm.fit(X_train, y_train / e_train, sample_weight=e_train)

for name, model in [("GLM", glm), ("GBM", gbm)]:
    pred = model.predict(X_test)
    dev = mean_tweedie_deviance(y_test / e_test, pred, sample_weight=e_test, power=1.5)
    print(f"{name}: Tweedie deviance {dev:.4f}")
```

Two things to note. First, use a deviance that matches the data (Tweedie or Poisson), not mean squared error, which is dominated by a few large claims. Second, look beyond one summary metric: **lift charts** and **double lift charts** comparing the two models by predicted decile are what pricing teams actually use to decide whether a model is better.

## Explainability is not optional

Pricing decisions affect customers and are scrutinised by regulators. In the UK, the FCA's general insurance pricing practices rules (PS21/5) are a reminder that how prices are set matters, not just whether they are accurate. Any model used in pricing needs to be explainable to underwriters, compliance and, where relevant, customers.

Tools that help:

- **Partial dependence and accumulated local effects** plots show how predictions change with a single factor.
- **SHAP values** (Lundberg and Lee, 2017) attribute each prediction to its inputs and can be aggregated into global importance.
- **Monotonic constraints**, supported by most boosting libraries, enforce business logic such as "higher sum insured should not reduce the price".

## Specialty lines are different

Most published work on machine learning pricing uses personal lines data with hundreds of thousands of policies. Specialty and London market business looks very different:

- **Small data.** A marine or political risk book may have a few thousand policies and very few large losses. Flexible models overfit easily.
- **Heterogeneous risks.** Every risk is partly bespoke, and much of the information sits in underwriter judgement and submission documents rather than structured fields.
- **Heavy tails.** Results are driven by rare, large losses that a model fitted to the body of the distribution will not capture.
- **Rating structures.** Pricing is often built from exposure based rates, increased limit factors and expert adjustments rather than a single statistical model.

In this setting, the most valuable "machine learning" is often modest: well engineered features, credibility weighting between portfolio experience and market benchmarks, GLMs with sensible constraints, and good tooling that lets underwriters see the drivers behind a technical price. Wüthrich and Merz (2023) cover the statistical foundations that link classical actuarial models with modern machine learning, and it is a good reminder that the two traditions have more in common than the hype suggests.

## Practical advice

1. **Start with a well built GLM.** It is the benchmark everything else must beat.
2. **Use the right loss function** for counts, severity or pure premium.
3. **Handle exposure explicitly** through offsets or weights.
4. **Validate out of time**, not just out of sample. Insurance data drifts with the cycle.
5. **Compare with lift charts**, not just a single metric.
6. **Explain everything** you put in front of an underwriter or regulator.
7. **Respect small data.** In specialty lines, credibility and judgement are features, not failures.

## References

- Goldburd, M., Khare, A., Tevet, D. and Guller, D. (2020). *Generalized Linear Models for Insurance Rating*, 2nd edition. CAS Monograph No. 5. Casualty Actuarial Society.
- Wüthrich, M. V. and Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer (open access).
- Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD Conference*, 785 to 794.
- Henckaerts, R., Côté, M.-P., Antonio, K. and Verbelen, R. (2021). Boosting insights in insurance tariff plans with tree-based machine learning methods. *North American Actuarial Journal*, 25(2), 255 to 285.
- Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
- Financial Conduct Authority (2021). *PS21/5: General insurance pricing practices*.
- scikit-learn documentation. *Tweedie regression on insurance claims*. scikit-learn.org
