---
layout: post
title: "Building Analytics Services for Actuaries and Underwriters With FastAPI"
date: 2026-11-02
description: "Why actuarial logic belongs in a service, how to structure one with FastAPI and Pydantic, test it properly, version it, and put it in front of underwriters through Excel."
categories: [Software Engineering, Data Engineering]
tags: [FastAPI, Python, Pydantic, APIs, Excel Add-ins, Testing]
---

In many insurance teams, the same calculation lives in five places: a pricing spreadsheet, a reserving notebook, a Power BI measure, a rating tool and somebody's personal macro. Each copy drifts a little. When an assumption changes, someone has to remember to update all of them, and usually one gets missed.

Putting that logic behind a small, well tested web service solves this. There is one implementation, one set of tests, and every consumer (dashboards, notebooks, rating tools, even Excel) calls the same thing. This post shows how to build such a service with FastAPI, using a layer loss cost calculation as the example.

## Why a service?

- **One source of truth.** The calculation is written, reviewed and tested once.
- **Consistency across tools.** A notebook, a dashboard and an Excel sheet all get the same answer.
- **Change control.** Updating an assumption is a reviewed code change with a version number, not an edit to a shared spreadsheet.
- **Auditability.** Requests and responses can be logged, so you can show exactly what was calculated and when.

FastAPI is a good fit for analytics teams already working in Python: it uses type hints for validation through Pydantic, generates interactive documentation automatically, and is fast enough for almost any internal workload.

## Keep the maths pure

The most important design decision is to keep the business logic in plain functions with no knowledge of HTTP. That makes it easy to test, reuse in notebooks, and review.

Here is the expected loss to an excess of loss layer when individual loss sizes follow a lognormal distribution. It uses the closed form limited expected value of the lognormal (see Klugman, Panjer and Willmot, *Loss Models*):

```python
# pricing/layers.py
import math
from scipy.stats import norm

def lognormal_lev(limit: float, mu: float, sigma: float) -> float:
    """Limited expected value E[min(X, limit)] for X ~ Lognormal(mu, sigma)."""
    if limit <= 0:
        return 0.0
    log_l = math.log(limit)
    return (
        math.exp(mu + sigma**2 / 2) * norm.cdf((log_l - mu - sigma**2) / sigma)
        + limit * (1 - norm.cdf((log_l - mu) / sigma))
    )

def layer_expected_loss(
    attachment: float, limit: float, frequency: float, mu: float, sigma: float
) -> float:
    """Expected annual loss to a per-risk layer `limit` xs `attachment`."""
    per_loss = lognormal_lev(attachment + limit, mu, sigma) - lognormal_lev(attachment, mu, sigma)
    return frequency * per_loss
```

## Validate at the edge with Pydantic

Request and response models define the contract. Pydantic rejects bad input with a clear error before it reaches the maths, which is exactly where validation belongs:

```python
# api/schemas.py
from pydantic import BaseModel, Field

class LayerRequest(BaseModel):
    attachment: float = Field(ge=0, description="Layer attachment point")
    limit: float = Field(gt=0, description="Layer limit")
    frequency: float = Field(ge=0, description="Expected ground-up losses per year")
    mu: float = Field(description="Lognormal mu (log scale)")
    sigma: float = Field(gt=0, description="Lognormal sigma (log scale)")

class LayerResponse(BaseModel):
    expected_loss: float
    rate_on_line: float
    model_version: str
```

## A thin API layer

The endpoint does nothing except translate between HTTP and the pure function:

```python
# api/main.py
from fastapi import FastAPI
from api.schemas import LayerRequest, LayerResponse
from pricing.layers import layer_expected_loss

MODEL_VERSION = "2026.11.0"

app = FastAPI(title="Pricing Analytics API", version=MODEL_VERSION)

@app.post("/v1/layers/expected-loss", response_model=LayerResponse)
def expected_loss(req: LayerRequest) -> LayerResponse:
    el = layer_expected_loss(req.attachment, req.limit, req.frequency, req.mu, req.sigma)
    return LayerResponse(
        expected_loss=el,
        rate_on_line=el / req.limit,
        model_version=MODEL_VERSION,
    )

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}
```

Run it with `uvicorn api.main:app --reload` and open `/docs` for interactive documentation that underwriters and analysts can try directly.

Two details are worth copying:

- **Version in the path** (`/v1/...`). When the contract has to change, add `/v2` and retire `/v1` gracefully instead of breaking every consumer at once.
- **Model version in every response.** Anyone looking at a stored result can tell which version of the logic produced it.

## Test both layers

Test the maths against known values and properties, and test the API contract separately:

```python
# tests/test_layers.py
import math
import pytest
from fastapi.testclient import TestClient
from api.main import app
from pricing.layers import lognormal_lev, layer_expected_loss

def test_lev_tends_to_mean_for_large_limits():
    mu, sigma = 13.0, 1.2
    assert lognormal_lev(1e15, mu, sigma) == pytest.approx(math.exp(mu + sigma**2 / 2), rel=1e-6)

def test_stacked_layers_add_up():
    args = dict(frequency=10, mu=13.0, sigma=1.2)
    lower = layer_expected_loss(1e6, 4e6, **args)
    upper = layer_expected_loss(5e6, 5e6, **args)
    combined = layer_expected_loss(1e6, 9e6, **args)
    assert lower + upper == pytest.approx(combined)

client = TestClient(app)

def test_rejects_negative_attachment():
    r = client.post("/v1/layers/expected-loss", json={
        "attachment": -1, "limit": 1e6, "frequency": 5, "mu": 13, "sigma": 1.2,
    })
    assert r.status_code == 422

def test_returns_version():
    r = client.post("/v1/layers/expected-loss", json={
        "attachment": 1e6, "limit": 4e6, "frequency": 5, "mu": 13, "sigma": 1.2,
    })
    assert r.status_code == 200
    assert r.json()["model_version"]
```

Property tests like "stacked layers add up to the combined layer" catch subtle errors that a single hard coded expected value would miss. Run the suite in CI on every pull request.

## Meeting underwriters where they work: Excel

However good a web app is, many underwriters and actuaries live in Excel. Office Add-ins support **custom functions** written in JavaScript, which can call your API and return the result into a cell like any built in function:

```javascript
/**
 * Expected annual loss to a layer.
 * @customfunction
 * @param {number} attachment
 * @param {number} limit
 * @param {number} frequency
 * @param {number} mu
 * @param {number} sigma
 * @returns {Promise<number>}
 */
async function LAYERLOSS(attachment, limit, frequency, mu, sigma) {
  const response = await fetch("https://pricing-api.internal/v1/layers/expected-loss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ attachment, limit, frequency, mu, sigma }),
  });
  if (!response.ok) {
    throw new CustomFunctions.Error(
      CustomFunctions.ErrorCode.invalidValue,
      `Pricing API error ${response.status}`
    );
  }
  return (await response.json()).expected_loss;
}
```

Now `=LAYERLOSS(1000000, 4000000, 5, 13, 1.2)` in a cell calls the same tested logic as every other tool. In production you would add authentication (for example Microsoft Entra ID) so only authorised users can call the API.

## Production considerations

- **Configuration from the environment.** Follow the Twelve-Factor App guidance: no credentials or environment specific URLs in code.
- **Authentication and authorisation.** Internal does not mean open. Use your organisation's identity provider.
- **Logging and monitoring.** Log request IDs, timings and model versions; alert on error rates.
- **Containerise and deploy through CI/CD,** with automated tests as a gate.
- **Keep it small.** A focused service with a clear contract is easier to trust than a sprawling one. Newman (2021) is a good guide to drawing sensible service boundaries.

## Takeaways

A service is not about technology for its own sake. It is about making sure that a pricing or reserving calculation exists once, is tested, is versioned, and gives the same answer wherever it is used. FastAPI and Pydantic make that achievable for any analytics team comfortable with Python.

## References

- FastAPI documentation. fastapi.tiangolo.com
- Pydantic documentation. docs.pydantic.dev
- Microsoft Learn. *Create custom functions in Excel* (Office Add-ins). learn.microsoft.com/office/dev/add-ins
- Wiggins, A. *The Twelve-Factor App*. 12factor.net
- Newman, S. (2021). *Building Microservices*, 2nd edition. O'Reilly Media.
- Klugman, S. A., Panjer, H. H. and Willmot, G. E. (2019). *Loss Models: From Data to Decisions*, 5th edition. Wiley.
