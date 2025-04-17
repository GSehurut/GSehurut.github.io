---
layout: project
title: "Exceedance Probability Curves for Catastrophe Risk"
date: 2024-03-20
categories: [Risk Modeling, Catastrophe]
tags: [Exceedance Probability, Catastrophe Modeling, Python, Risk Analytics]
image: /pictures/flowers.jpg
description: "Implementation of exceedance probability curves for catastrophe risk assessment using Python and historical data."
---

# Exceedance Probability Curves for Catastrophe Risk

## Project Overview

This project implements exceedance probability (EP) curves for catastrophe risk assessment, a crucial tool in understanding and managing catastrophic risks in insurance portfolios. EP curves show the probability that a certain level of loss will be exceeded in a given time period.

## Technical Implementation

The implementation uses Python with key libraries for statistical analysis and visualization:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, genextreme
import seaborn as sns

class ExceedanceProbability:
    def __init__(self, historical_losses):
        self.losses = np.sort(historical_losses)
        self.n = len(historical_losses)
        
    def calculate_ep_curve(self):
        """Calculate exceedance probabilities for each loss level"""
        ranks = np.arange(1, self.n + 1)
        exceedance_probs = 1 - ranks / (self.n + 1)
        return self.losses, exceedance_probs
    
    def fit_extreme_value_distribution(self):
        """Fit Gumbel distribution to the tail of the loss distribution"""
        # Use top 20% of losses for extreme value fitting
        tail_losses = self.losses[int(0.8 * self.n):]
        params = genextreme.fit(tail_losses)
        return params
    
    def calculate_aal(self):
        """Calculate Average Annual Loss"""
        return np.mean(self.losses)
    
    def calculate_pml(self, return_period):
        """Calculate Probable Maximum Loss for given return period"""
        ep_curve = self.calculate_ep_curve()
        losses, probs = ep_curve
        # Interpolate to find loss at target probability
        target_prob = 1 / return_period
        pml = np.interp(target_prob, probs[::-1], losses[::-1])
        return pml
    
    def plot_ep_curve(self, save_path=None):
        """Plot the exceedance probability curve"""
        losses, probs = self.calculate_ep_curve()
        
        plt.figure(figsize=(12, 6))
        plt.semilogy(losses, probs, 'b-', label='Historical EP Curve')
        
        # Add return period labels
        return_periods = [10, 50, 100, 250, 500]
        for rp in return_periods:
            pml = self.calculate_pml(rp)
            plt.axvline(x=pml, color='r', linestyle='--', alpha=0.3)
            plt.text(pml, 0.1, f'{rp}yr', rotation=90, va='bottom')
        
        plt.xlabel('Loss Amount')
        plt.ylabel('Exceedance Probability')
        plt.title('Exceedance Probability Curve')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Example usage with synthetic data
np.random.seed(42)
historical_losses = np.concatenate([
    np.random.lognormal(10, 1, 900),  # Small to medium losses
    np.random.lognormal(12, 1.5, 100)  # Large losses
])

ep = ExceedanceProbability(historical_losses)

# Calculate key metrics
aal = ep.calculate_aal()
pml_100yr = ep.calculate_pml(100)
pml_250yr = ep.calculate_pml(250)

print(f"Average Annual Loss: {aal:,.2f}")
print(f"100-year PML: {pml_100yr:,.2f}")
print(f"250-year PML: {pml_250yr:,.2f}")

# Plot EP curve
ep.plot_ep_curve('ep_curve.png')
```

## Key Features

1. **Historical Loss Analysis**
   - Processing and analysis of historical loss data
   - Calculation of exceedance probabilities
   - Extreme value distribution fitting

2. **Risk Metrics**
   - Average Annual Loss (AAL)
   - Probable Maximum Loss (PML) for various return periods
   - Exceedance probability curves

3. **Visualization**
   - Interactive EP curve plotting
   - Return period markers
   - Custom styling and annotations

## Practical Applications

This implementation can be used for:

1. **Portfolio Risk Assessment**
   - Understanding the probability of different loss levels
   - Identifying key risk drivers
   - Setting risk appetite thresholds

2. **Capital Modeling**
   - Determining capital requirements at different confidence levels
   - Stress testing scenarios
   - Reinsurance optimization

3. **Risk Management**
   - Monitoring portfolio risk metrics
   - Early warning indicators
   - Risk mitigation strategies

## Technical Details

The implementation uses several advanced statistical techniques:

1. **Extreme Value Theory**
   - Gumbel distribution fitting
   - Tail risk estimation
   - Return period calculations

2. **Data Processing**
   - Loss data cleaning and validation
   - Outlier detection and treatment
   - Distribution fitting

3. **Visualization**
   - Logarithmic scaling for better visualization
   - Custom annotations and markers
   - Professional styling

## Future Enhancements

Planned improvements include:

1. **Machine Learning Integration**
   - Automated distribution selection
   - Parameter optimization
   - Anomaly detection

2. **Real-time Updates**
   - Streaming data integration
   - Automated reporting
   - Alert systems

3. **Advanced Analytics**
   - Copula-based dependency modeling
   - Climate change adjustments
   - Spatial risk analysis

## Conclusion

This project provides a robust framework for analyzing and visualizing catastrophe risk using exceedance probability curves. The implementation is flexible and can be adapted to various types of catastrophic risks and insurance portfolios.

For more information or to discuss potential applications, please feel free to reach out. 