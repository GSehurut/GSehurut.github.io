---
layout: post
title: "Monte Carlo Simulation for Loss Modelling in Specialty Insurance"
date: 2024-03-20
categories: [Actuarial, Risk Modelling]
tags: [Monte Carlo, Loss Simulation, Specialty Insurance, Python]
image: /pictures/coding.jpg
---

# Monte Carlo Simulation for Loss Modelling in Specialty Insurance

## Introduction

Monte Carlo simulation is a powerful technique used in actuarial science to model complex risk scenarios, particularly in specialty insurance lines. This post explores how we can implement Monte Carlo simulation to model loss distributions and calculate key risk metrics.

## The Problem

In specialty insurance, we often deal with complex risks where traditional analytical methods fall short. Consider a portfolio of marine cargo insurance with the following characteristics:

- Multiple perils (storm, collision, fire)
- Correlated risks
- Complex policy structures
- Limited historical data

## Implementation

Let's walk through a Python implementation of a Monte Carlo simulation for this scenario:

```python
import numpy as np
import pandas as pd
from scipy.stats import lognorm, multivariate_normal, poisson
import matplotlib.pyplot as plt
import seaborn as sns

class LossSimulator:
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
        
    def simulate_losses(self, mean_frequency, std_frequency, 
                       mean_severity, std_severity, 
                       correlation_matrix):
        # Simulate number of claims
        frequencies = np.random.normal(mean_frequency, std_frequency, self.n_simulations)
        frequencies = np.maximum(frequencies, 0)  # Ensure non-negative
        
        # Simulate claim severities
        severities = np.random.lognormal(mean_severity, std_severity, self.n_simulations)
        
        # Calculate total losses
        total_losses = frequencies * severities
        
        return total_losses
    
    def simulate_multiple_perils(self, peril_params):
        """Simulate losses for multiple perils with correlations"""
        n_perils = len(peril_params)
        
        # Generate correlated frequencies
        mean_freqs = [params['mean_frequency'] for params in peril_params]
        cov_matrix = np.array([params['freq_covariance'] for params in peril_params])
        
        frequencies = np.random.multivariate_normal(mean_freqs, cov_matrix, self.n_simulations)
        frequencies = np.maximum(frequencies, 0)
        
        # Simulate severities for each peril
        severities = []
        for params in peril_params:
            severity = np.random.lognormal(
                params['mean_severity'], 
                params['std_severity'], 
                self.n_simulations
            )
            severities.append(severity)
        
        severities = np.array(severities).T
        
        # Calculate total losses
        total_losses = np.sum(frequencies * severities, axis=1)
        
        return total_losses, frequencies, severities
    
    def calculate_metrics(self, losses):
        metrics = {
            'Mean Loss': np.mean(losses),
            'Standard Deviation': np.std(losses),
            'VaR 95%': np.percentile(losses, 95),
            'VaR 99%': np.percentile(losses, 99),
            'TVaR 95%': np.mean(losses[losses >= np.percentile(losses, 95)]),
            'TVaR 99%': np.mean(losses[losses >= np.percentile(losses, 99)]),
            'Skewness': self.calculate_skewness(losses),
            'Kurtosis': self.calculate_kurtosis(losses)
        }
        return metrics
    
    def calculate_skewness(self, data):
        """Calculate skewness of the data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis of the data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def plot_loss_distribution(self, losses, title="Simulated Loss Distribution"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(losses, bins=50, density=True, alpha=0.7, color='skyblue')
        ax1.set_title(f'{title} - Histogram')
        ax1.set_xlabel('Loss Amount')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy.stats import probplot
        probplot(losses, dist="norm", plot=ax2)
        ax2.set_title(f'{title} - Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def stress_test(self, losses, stress_factors):
        """Perform stress testing on simulated losses"""
        stressed_losses = {}
        
        for scenario, factor in stress_factors.items():
            stressed_losses[scenario] = losses * factor
        
        return stressed_losses

# Example usage
simulator = LossSimulator(n_simulations=10000)

# Single peril simulation
losses = simulator.simulate_losses(
    mean_frequency=10,
    std_frequency=2,
    mean_severity=100000,
    std_severity=0.5,
    correlation_matrix=np.array([[1, 0.3], [0.3, 1]])
)

metrics = simulator.calculate_metrics(losses)
print("Risk Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:,.2f}")

# Multiple perils simulation
peril_params = [
    {
        'mean_frequency': 5,
        'freq_covariance': [4, 1, 1],
        'mean_severity': 8,
        'std_severity': 0.3
    },
    {
        'mean_frequency': 3,
        'freq_covariance': [1, 9, 1],
        'mean_severity': 9,
        'std_severity': 0.4
    },
    {
        'mean_frequency': 2,
        'freq_covariance': [1, 1, 16],
        'mean_severity': 10,
        'std_severity': 0.5
    }
]

multi_losses, frequencies, severities = simulator.simulate_multiple_perils(peril_params)
multi_metrics = simulator.calculate_metrics(multi_losses)

print("\nMulti-Peril Risk Metrics:")
for metric, value in multi_metrics.items():
    print(f"{metric}: {value:,.2f}")

# Plot results
simulator.plot_loss_distribution(losses, "Single Peril Losses")
simulator.plot_loss_distribution(multi_losses, "Multi-Peril Losses")
```

## Advanced Features

### 1. Time-Dependent Simulation
For long-tail lines, we need to consider time dependencies:

```python
class TimeDependentSimulator(LossSimulator):
    def __init__(self, n_simulations=10000, time_horizon=10):
        super().__init__(n_simulations)
        self.time_horizon = time_horizon
    
    def simulate_time_dependent_losses(self, initial_params, trend_params):
        """Simulate losses with time-dependent parameters"""
        losses_by_year = []
        
        for year in range(self.time_horizon):
            # Adjust parameters based on trends
            current_freq = initial_params['mean_frequency'] * (1 + trend_params['freq_trend'] * year)
            current_severity = initial_params['mean_severity'] * (1 + trend_params['severity_trend'] * year)
            
            # Simulate losses for this year
            year_losses = self.simulate_losses(
                current_freq, initial_params['std_frequency'],
                current_severity, initial_params['std_severity'],
                initial_params['correlation_matrix']
            )
            
            losses_by_year.append(year_losses)
        
        return np.array(losses_by_year)
```

### 2. Reinsurance Modeling
Incorporating reinsurance structures:

```python
def apply_reinsurance(losses, reinsurance_structure):
    """Apply reinsurance structure to losses"""
    retained_losses = np.zeros_like(losses)
    ceded_losses = np.zeros_like(losses)
    
    for i, loss in enumerate(losses):
        if loss <= reinsurance_structure['retention']:
            retained_losses[i] = loss
        else:
            retained_losses[i] = reinsurance_structure['retention']
            ceded_losses[i] = loss - reinsurance_structure['retention']
    
    return retained_losses, ceded_losses
```

### 3. Scenario Analysis
```python
def scenario_analysis(simulator, base_params, scenarios):
    """Perform scenario analysis"""
    results = {}
    
    for scenario_name, scenario_params in scenarios.items():
        # Adjust base parameters for scenario
        adjusted_params = {**base_params, **scenario_params}
        
        # Simulate losses
        losses = simulator.simulate_losses(**adjusted_params)
        
        # Calculate metrics
        metrics = simulator.calculate_metrics(losses)
        results[scenario_name] = metrics
    
    return results
```

## Key Insights

1. **Distribution Fitting**: The choice of distribution for frequency and severity is crucial. In this example, we used:
   - Normal distribution for frequency (with non-negative constraint)
   - Lognormal distribution for severity

2. **Correlation Modeling**: The correlation matrix allows us to capture dependencies between different perils or lines of business.

3. **Risk Metrics**: The simulation provides several important metrics:
   - Value at Risk (VaR)
   - Tail Value at Risk (TVaR)
   - Mean and standard deviation of losses
   - Skewness and kurtosis

## Practical Applications

This simulation approach can be used for:

1. **Capital Modeling**: Determining required capital for different risk percentiles
2. **Pricing**: Understanding the full distribution of potential losses
3. **Risk Management**: Identifying key drivers of portfolio risk
4. **Reinsurance Strategy**: Optimizing reinsurance structures based on simulated outcomes

## Model Validation

### 1. Backtesting
```python
def backtest_model(historical_data, simulated_data):
    """Backtest the simulation model"""
    # Compare key statistics
    historical_stats = {
        'mean': np.mean(historical_data),
        'std': np.std(historical_data),
        'percentiles': np.percentile(historical_data, [50, 75, 90, 95, 99])
    }
    
    simulated_stats = {
        'mean': np.mean(simulated_data),
        'std': np.std(simulated_data),
        'percentiles': np.percentile(simulated_data, [50, 75, 90, 95, 99])
    }
    
    return historical_stats, simulated_stats
```

### 2. Sensitivity Analysis
```python
def sensitivity_analysis(simulator, base_params, param_ranges):
    """Perform sensitivity analysis on key parameters"""
    results = {}
    
    for param_name, param_range in param_ranges.items():
        param_results = []
        
        for value in param_range:
            # Update parameter
            test_params = base_params.copy()
            test_params[param_name] = value
            
            # Simulate losses
            losses = simulator.simulate_losses(**test_params)
            metrics = simulator.calculate_metrics(losses)
            
            param_results.append({
                'value': value,
                'metrics': metrics
            })
        
        results[param_name] = param_results
    
    return results
```

## Next Steps

Future enhancements could include:

1. Incorporating more complex dependencies using copulas
2. Adding time-dependent factors for long-tail lines
3. Implementing machine learning for parameter estimation
4. Creating interactive dashboards for risk visualization

## Conclusion

Monte Carlo simulation provides a flexible framework for modeling complex insurance risks. By understanding the full distribution of potential outcomes, we can make more informed decisions about capital allocation, pricing, and risk management.

### Key Takeaways

1. **Flexibility**: Monte Carlo simulation can handle complex, non-linear relationships
2. **Comprehensive Analysis**: Provides full distribution of outcomes, not just point estimates
3. **Scenario Testing**: Enables testing of various scenarios and stress conditions
4. **Risk Metrics**: Generates multiple risk metrics for comprehensive analysis

### Best Practices

1. **Model Validation**: Always validate models against historical data
2. **Sensitivity Analysis**: Understand the impact of parameter uncertainty
3. **Documentation**: Maintain clear documentation of assumptions and methodology
4. **Regular Updates**: Update models as new data becomes available

Would you like to explore any specific aspect of this implementation in more detail? Feel free to reach out with questions or suggestions for future topics. 