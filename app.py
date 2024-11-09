import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import yfinance as yf

# Set Streamlit page configuration
st.set_page_config(page_title="Portfolio VaR Analysis", layout="wide")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Concept"])

if page == "Home":
    # Title and Description
    st.title("Portfolio Value at Risk (VaR) Analysis")
    st.write("""
    This app allows you to upload your portfolio data and perform various risk analysis techniques, including Value at Risk (VaR), Stress Testing, and Backtesting.
    You can explore Historical VaR, Parametric VaR, and Monte Carlo VaR methods, along with stress testing scenarios.
    """)

    # Upload CSV File
    uploaded_file = st.file_uploader("Upload your portfolio CSV file  [Adj. Close price of assets]", type=["csv"])

    if uploaded_file:
        # Load the portfolio data
        portfolio_df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        portfolio_df.set_index('Date', inplace=True)
        st.subheader("Portfolio Data")
        st.dataframe(portfolio_df.head())

        # Plot the historical prices
        st.subheader("Portfolio Price Trends")
        fig, ax = plt.subplots(figsize=(10, 5))
        portfolio_df.plot(ax=ax)
        plt.title("Historical Prices of Portfolio Assets")
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(fig)

        # Select stocks for VaR calculation
        selected_stocks = st.multiselect("Select Stocks for VaR Calculation", options=portfolio_df.columns.tolist(), default=portfolio_df.columns.tolist())

        # Filter data based on selected stocks
        selected_data = portfolio_df[selected_stocks]

        # VaR Calculation Parameters
        st.sidebar.header("VaR Parameters")
        confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=90, max_value=99, value=95)
        var_days = st.sidebar.number_input("VaR Days (Holding Period)", min_value=1, max_value=30, value=1)


        # Calculate Returns
        returns = selected_data.pct_change().dropna()

        # Calculate Portfolio Returns
        portfolio_returns = returns.mean(axis=1)

        # VaR Calculations
        st.subheader("VaR Calculations")

        # 1. Historical VaR
        st.write("### 1. Historical VaR")
        st.write("""
        **Explanation**: Historical VaR is calculated using actual historical returns. It represents the worst loss expected at a certain confidence level over a given time period, assuming the future resembles the past.
        """)
        historical_var_individual = np.percentile(returns, 100 - confidence_level, axis=0) * np.sqrt(var_days)
        historical_var_portfolio = np.percentile(portfolio_returns, 100 - confidence_level) * np.sqrt(var_days)
        st.write(f"Historical VaR at {confidence_level}% confidence level (Individual Assets):")
        st.dataframe(pd.DataFrame(historical_var_individual, index=returns.columns, columns=["Historical VaR"]))
        st.write(f"Historical VaR at {confidence_level}% confidence level (Portfolio): {historical_var_portfolio:.4f}")

        # Plot Historical VaR
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax)
        ax.axvline(historical_var_portfolio, color='red', linestyle='--', label=f'Historical VaR ({confidence_level}%)')
        ax.legend()
        plt.title("Historical VaR - Portfolio Returns Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        st.pyplot(fig)

        # 2. Parametric VaR (assuming normal distribution)
        st.write("### 2. Parametric VaR (Normal Distribution)")
        st.write("""
        **Explanation**: Parametric VaR assumes that asset returns are normally distributed. It uses the mean and standard deviation of historical returns to calculate the risk.
        """)
        mean_returns = returns.mean()
        std_returns = returns.std()
        parametric_var_individual = norm.ppf(1 - confidence_level / 100) * std_returns * np.sqrt(var_days)
        parametric_var_portfolio = norm.ppf(1 - confidence_level / 100) * portfolio_returns.std() * np.sqrt(var_days)
        st.write(f"Parametric VaR at {confidence_level}% confidence level (Individual Assets):")
        st.dataframe(pd.DataFrame(parametric_var_individual, index=returns.columns, columns=["Parametric VaR"]))
        st.write(f"Parametric VaR at {confidence_level}% confidence level (Portfolio): {parametric_var_portfolio:.4f}")

        # Plot Parametric VaR
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax)
        ax.axvline(parametric_var_portfolio, color='blue', linestyle='--', label=f'Parametric VaR ({confidence_level}%)')
        ax.legend()
        plt.title("Parametric VaR - Portfolio Returns Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        st.pyplot(fig)

        # 3. Monte Carlo VaR

        st.write("### 3. Monte Carlo VaR")
        st.write("""
        **Explanation**: Monte Carlo VaR uses simulations to generate a wide range of possible outcomes based on historical mean and standard deviation of returns. It is useful for capturing non-linear risk factors.
        """)
        num_simulations = st.sidebar.slider("Number of Simulations for Monte Carlo", min_value=100, max_value=10000, value=1000)

        # Monte Carlo simulation for individual assets
        simulated_vars = []
        simulated_portfolio_returns = []

        for stock in selected_stocks:
            simulated_returns = np.random.normal(mean_returns[stock], std_returns[stock], (num_simulations, var_days))
            simulated_portfolio = np.cumprod(1 + simulated_returns, axis=1)
            simulated_var = np.percentile(simulated_portfolio[:, -1] - 1, 100 - confidence_level)
            simulated_vars.append(simulated_var)

        # Monte Carlo simulation for the portfolio
        portfolio_simulations = np.random.normal(portfolio_returns.mean(), portfolio_returns.std(), (num_simulations, var_days))
        simulated_portfolio_cumprod = np.cumprod(1 + portfolio_simulations, axis=1)
        mc_portfolio_var = np.percentile(simulated_portfolio_cumprod[:, -1] - 1, 100 - confidence_level)

        # Convert results to DataFrame
        mc_var_df = pd.DataFrame(simulated_vars, index=selected_stocks, columns=["Monte Carlo VaR"])
        st.write(f"Monte Carlo VaR at {confidence_level}% confidence level (Individual Assets):")
        st.dataframe(mc_var_df)
        st.write(f"Monte Carlo VaR at {confidence_level}% confidence level (Portfolio): {mc_portfolio_var:.4f}")

        # Plot 1: Monte Carlo VaR Only
        fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
        sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax_mc)
        ax_mc.axvline(mc_portfolio_var, color='green', linestyle='--', label=f'Monte Carlo VaR ({confidence_level}%)')
        ax_mc.legend()
        plt.title("Monte Carlo VaR - Portfolio Returns Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        st.pyplot(fig_mc)

        # Plot 2: Combined VaR Plot
        st.write("### Combined VaR Comparison - Portfolio Returns Distribution")
        fig_combined, ax_combined = plt.subplots(figsize=(10, 5))
        sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax_combined)
        ax_combined.axvline(historical_var_portfolio, color='red', linestyle='--', label=f'Historical VaR ({confidence_level}%)')
        ax_combined.axvline(parametric_var_portfolio, color='blue', linestyle='--', label=f'Parametric VaR ({confidence_level}%)')
        ax_combined.axvline(mc_portfolio_var, color='green', linestyle='--', label=f'Monte Carlo VaR ({confidence_level}%)')
        ax_combined.legend()
        plt.title("Combined VaR Comparison - Portfolio Returns Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        st.pyplot(fig_combined)





        # Stress Testing
        st.subheader("Stress Testing")
        st.write("""
        **Explanation**: Stress Testing evaluates the impact of extreme market conditions on your portfolio. It helps understand how your portfolio would perform under adverse scenarios.
        """)
        shock_factor = st.sidebar.slider("Shock Factor (%)", min_value=5, max_value=50, value=20)
        stressed_returns = returns - (shock_factor / 100)
        stressed_portfolio_returns = stressed_returns.mean(axis=1)
        stressed_var = np.percentile(stressed_portfolio_returns, 100 - confidence_level) * np.sqrt(var_days)
        st.write(f"Stressed VaR at {confidence_level}% confidence level (Portfolio): {stressed_var:.4f}")

        # Monte Carlo Simulation - Portfolio Stress Testing Plot
        # st.write("Monte Carlo Simulation - Portfolio Stress Testing")

        # # Number of simulations
        # num_simulations = st.sidebar.slider("Number of Simulations for Stress Testing", min_value=100, max_value=5000, value=1000)
        # time_horizon = st.sidebar.number_input("Time Horizon (days)", min_value=50, max_value=500, value=250)

        # # Generate Monte Carlo Simulations
        # simulated_portfolio_values = []
        # np.random.seed(42)  # For reproducibility
        # initial_portfolio_value = 1  # Normalize initial value to 1

        # for _ in range(num_simulations):
        #     daily_returns = np.random.normal(portfolio_returns.mean(), portfolio_returns.std(), time_horizon)
        #     cumulative_returns = np.cumprod(1 + daily_returns) * initial_portfolio_value
        #     simulated_portfolio_values.append(cumulative_returns)

        # # Convert simulations to DataFrame
        # simulated_df = pd.DataFrame(simulated_portfolio_values).T

        # # Plot the Monte Carlo simulation with colored lines
        # fig, ax = plt.subplots(figsize=(10, 6))

        # # Use a colormap to assign different colors to each line
        # cmap = plt.get_cmap('viridis')
        # colors = cmap(np.linspace(0, 1, num_simulations))

        # # Plot each simulation with a unique color
        # for i in range(num_simulations):
        #     ax.plot(simulated_df[i], color=colors[i], alpha=0.5)

        # # Plot the mean portfolio value
        # ax.plot(simulated_df.mean(axis=1), color='blue', label='Mean portfolio performance', linewidth=2)

        # plt.title("Monte Carlo Simulation - Portfolio Stress Testing")
        # plt.xlabel("Time")
        # plt.ylabel("Portfolio Value")
        # plt.legend()
        # st.pyplot(fig)

        # Backtesting
        st.subheader("Backtesting")
        st.write("""
        **Explanation**: Backtesting compares the predicted VaR with actual portfolio returns to assess the accuracy of the VaR model. If the actual loss exceeds the VaR, it indicates a failure.
        """)
        actual_losses = portfolio_returns[portfolio_returns < -historical_var_portfolio]
        st.write(f"Number of VaR breaches: {len(actual_losses)} out of {len(portfolio_returns)} observations.")

elif page == "Concept":
    st.title("Understanding Value at Risk (VaR) and Stress Testing Methods")

    st.markdown("""
    ### 1. Historical VaR
    **Formula**:
    """)
    st.latex(r"""
    \text{Historical VaR} = \text{Percentile}(\text{Portfolio Returns}, (100 - \text{Confidence Level})\%)
    """)
    st.markdown("""
    **Explanation**:  
    - Historical VaR is calculated using actual past returns of the portfolio.
    - It ranks historical returns from worst to best and identifies the loss at a specified percentile 
      (e.g., 5% for a 95% confidence level).
    - No assumptions are made about the distribution of returns, making it a non-parametric approach.
    
    ---

    ### 2. Parametric VaR (Normal Distribution)
    **Formula**:
    """)
    st.latex(r"""
    \text{Parametric VaR} = (\mu - z_{\alpha} \cdot \sigma) \times \sqrt{T}
    """)
    st.markdown("""
    - **μ**: Mean of the portfolio returns  
    - **σ**: Standard deviation of portfolio returns  
    - **zₐ**: Z-score for the chosen confidence level (e.g., -1.645 for 95%)  
    - **T**: Holding period (e.g., 1 day)
    
    **Explanation**:  
    - Assumes that portfolio returns follow a normal distribution.
    - Uses the mean (μ) and standard deviation (σ) of historical returns to calculate risk.
    - Often referred to as the "Variance-Covariance" approach.
    
    ---

    ### 3. Monte Carlo VaR
    **Formula**:
    """)
    st.latex(r"""
    \text{Monte Carlo VaR} = \text{Percentile}(\text{Simulated Returns}, (100 - \text{Confidence Level})\%)
    """)
    st.markdown("""
    **Explanation**:  
    - Uses simulations to generate possible future returns based on historical mean and volatility.
    - More flexible than the parametric method as it can capture non-linear risks and non-normal distributions.
    - Suitable for portfolios with options or non-linear instruments.
    
    ---

    ### 4. Stress Testing
    **Formula**:
    """)
    st.latex(r"""
    \text{Stressed Returns} = \text{Portfolio Returns} - (\text{Shock Factor} \times \text{Scenario Weights})
    """)
    st.markdown("""
    **Explanation**:  
    - Stress testing applies extreme but plausible shocks to portfolio returns based on hypothetical scenarios.
    - Helps assess the impact of adverse market conditions on the portfolio's performance.
    
    ---

    ### How to Choose a Method?
    - **Historical VaR**: Best for stable portfolios with ample historical data.
    - **Parametric VaR**: Suitable for portfolios assuming normally distributed returns.
    - **Monte Carlo VaR**: Ideal for portfolios with non-linear risk factors or limited historical data.
    - **Stress Testing**: Useful for evaluating risk under extreme market scenarios.
    """)


else:
    st.warning("Please upload a CSV file to proceed with the analysis.")

# Footer
st.markdown("---")
st.write("Developed by Archishman VB")
