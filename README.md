# VaR_Analysis

Project that estimates VaR of a portfolio using 3 different methods and performs stress-testing

A Streamlit-based web app for analyzing financial portfolio risks using:

### Value at Risk (VaR): Historical, Parametric, and Monte Carlo methods.

### Stress Testing: Hypothetical scenarios (e.g., Market Crash, Tech Selloff).

### Backtesting: Evaluates VaR model accuracy against historical data.

## Key Features

**Upload Portfolio**: CSV format with historical prices.

**VaR Analysis**: Calculate risk for individual assets and the entire portfolio.

**Stress Testing**: Analyze portfolio under extreme market conditions.

**Backtesting**: Check model accuracy by comparing actual vs. predicted losses.

**Visualization**: Portfolio returns distribution with VaR estimates.

## Installation

### Prerequisites

Python 3.8+

Streamlit

## Steps

1. Clone Repo
   
   ```python
   git clone https://github.com/Arch2775/portfolio-var-analysis.git
   
   cd portfolio-var-analysis```

3. Run App
   
   ```python
   streamlit run app.py```

## Usage

1.Upload a CSV file with historical prices.

2.Adjust VaR settings (confidence level, holding period).

3.View results for VaR, stress tests, and backtesting.

## File Structure

├── app.py               # Main app

├── portfolio.csv        # Sample data

├── requirements.txt     # Dependencies

├── README.md            # Documentation




