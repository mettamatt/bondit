# Bondit Portfolio Management Application

![Bondit Logo](logo.png)

## Overview

**Bondit** is a Python-based portfolio management tool designed to help long-term investors manage their bond allocations based on key economic indicators. By analyzing data from sources like the Federal Reserve Economic Data (FRED), Bondit adjusts investment allocations to optimize for wealth preservation, modest growth, and protection against inflation and other economic risks.

## Features

- **Economic Data Analysis**: Fetches and processes a variety of economic indicators from FRED.
- **Dynamic Portfolio Management**: Manages investment allocations across various bond asset types with enforced minimum and maximum constraints.
- **Decision Engine**: Applies data-driven decision rules to adjust portfolio allocations based on current economic conditions.
- **Rebalancing Reports**: Generates detailed reports outlining portfolio adjustments for transparency and record-keeping.

## Installation

### Prerequisites

- **Python 3.7** or higher
- **FRED API Key**: Obtain one from [FRED API Access](https://fred.stlouisfed.org/docs/api/fred/)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/bondit-portfolio-management.git
   cd bondit-portfolio-management
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Configuration**

   - **Copy the Example Configuration File**

     ```bash
     cp config/config.example.yaml config/config.yaml
     ```

   - **Edit `config/config.yaml`**

     Update the file with your FRED API key and desired file paths:

     ```yaml
     storage:
       fred_file_path: "data/fred_data.json"
       recommendation_file_path: "reports/rebalancing_report.txt"

     api:
       fred_api_key: "YOUR_FRED_API_KEY"
     ```

## Usage

Run the application by executing the `main.py` script:

```bash
python src/main.py
```

The application will perform the following steps:

1. **Initialize Loggers**: Sets up logging for the application and report generation.
2. **Load Configuration**: Reads and validates the configuration settings.
3. **Data Collection**: Fetches economic indicators from FRED.
4. **Portfolio Initialization**: Creates a new portfolio or loads an existing one.
5. **Decision Engine Execution**: Applies decision rules based on economic indicators to adjust the portfolio.
6. **Rebalancing Report Generation**: Produces a report detailing the portfolio adjustments.

## Configuration

### Economic Indicators

The application uses a set of predefined economic indicators defined in `src/config.py`. Each indicator includes:

- **series_id**: Identifier for the data series in FRED.
- **name**: Human-readable name of the indicator.
- **internal_key**: Key used internally within the application.
- **indicator_type**: Frequency of the data (e.g., 'MONTHLY', 'DAILY', 'QUARTERLY').
- **time_frame_weights**: Weights assigned to different time frames for analysis.
- **calculation_method**: Method used to calculate changes (e.g., 'percentage_change', 'z_score').
- **thresholds**: Upper and lower threshold values for generating trend signals.
- **rule_weight**: Weight assigned to the rule associated with this indicator, influencing its impact on portfolio adjustments.

You can customize these indicators or add new ones by modifying the `INDICATORS` list in `src/config.py`.

### Portfolio Constraints

Portfolio constraints, such as minimum and maximum allocations for each asset type, are defined in `src/portfolio.py`. You can adjust these constraints to fit your investment strategy.

**Default Asset Types and Constraints:**

- **Government Bonds - Long-Term**: Min 10%, Max 50%
- **Corporate Bonds - Long-Term**: Min 5%, Max 30%
- **Government Bonds - Intermediate-Term**: Min 10%, Max 40%
- **Corporate Bonds - Intermediate-Term**: Min 5%, Max 25%
- **Short-Term Bonds**: Min 5%, Max 20%
- **TIPS**: Min 2%, Max 10%
- **Nominal Bonds**: Min 0%, Max 30%

### Key Modules

- **`src/main.py`**: Entry point of the application, orchestrating the workflow.
- **`src/portfolio.py`**: Defines the `Portfolio` class for managing investment allocations and enforcing constraints.
- **`src/config.py`**: Contains configurations for economic indicators.
- **`src/decision_engine.py`**: Applies decision rules to adjust the portfolio based on economic indicators.
- **`src/utils.py`**: Utility functions for initialization, data fetching, and other common tasks.
- **`src/data_fetcher.py`**: Handles data retrieval from FRED.
- **`src/data_storage.py`**: Manages storage of fetched data.
- **`src/decision_rules/decision_rules.py`**: Contains the logic for decision rules applied by the Decision Engine.
- **`src/indicators.py`**: Defines the `EconomicIndicator` class and related functions.

## Rebalancing Reports

After applying the decision rules, the application generates a rebalancing report that includes:

- **Adjustments Made**: Details of changes to each asset type.
- **Reasons for Adjustments**: Based on economic indicators and decision rules.
- **Final Allocation Percentages**: The updated portfolio allocation.

### Sample Rebalancing Report

Below is a sample rebalancing report generated by the application:

```
Rebalancing Report Date: 2024-09-22

Rebalancing Scenario
---------------------

Portfolio Adjustments:
  - **Interest Rate Adjustment:**
    - **Analysis**: Federal Funds Rate z-score is 1.10.
    - **Action**: No adjustment made.
    - **Rationale**: No significant change in interest rates.

  - **CPI Adjustment:** Increased 'TIPS' by 9.00%.

  - **Inflation Adjustment:**
    - **Analysis**: High inflation detected with composite change 2.54%.
    - **Action**: Increased TIPS allocation by 9.00%.
    - **Rationale**: High inflation reduces real returns on nominal bonds; TIPS provide inflation protection.

  - **Breakeven Inflation Adjustment:** Decreased 'TIPS' by 8.00%.

  - **Breakeven Inflation Adjustment:** Increased 'Nominal Bonds' by 8.00%.

  - **Inflation Expectations Adjustment:**
    - **Analysis**: Falling inflation expectations detected with a change of -26.65%.
    - **Action**: Reduced TIPS allocation by 8.00% and increased Nominal Bonds by the same amount.
    - **Rationale**: Falling inflation expectations make nominal bonds more attractive; reducing TIPS allocation accordingly.

  - **Yield Curve Adjustment:**
    - **Analysis**: Yield spread change is 16.51%.
    - **Action**: No adjustment made.
    - **Rationale**: Yield spread is stable, indicating no significant change in economic outlook.

  - **Recession Probability Adjustment:**
    - **Adjustments**:
      - Decreased 'Government Bonds - Long-Term' by 3.00%.
      - Decreased 'Government Bonds - Intermediate-Term' by 3.00%.
      - Increased 'Corporate Bonds - Long-Term' by 3.00%.
      - Increased 'Corporate Bonds - Intermediate-Term' by 3.00%.
    - **Analysis**: Low recession probability detected at 1.96%, below lower threshold of 15.0%.
    - **Rationale**: Low recession probability reduces default risk on corporate bonds; increasing exposure can enhance returns.

  - **Credit Spread Adjustment:**
    - **Analysis**: Credit spread change is -14.18%.
    - **Action**: No adjustment made.
    - **Rationale**: Credit spreads are stable, indicating consistent corporate credit conditions.

  - **GDP Growth Adjustment:**
    - **Adjustments**:
      - Decreased 'Government Bonds - Long-Term' by 3.14%.
      - Decreased 'Corporate Bonds - Long-Term' by 3.14%.
      - Increased 'Short-Term Bonds' by 6.28%.
    - **Analysis**: GDP growth increased by 3.14%, exceeding the upper threshold of 3.0.
    - **Rationale**: Strong GDP growth suggests economic expansion; reducing duration mitigates interest rate risk.

  - **Employment Rate Adjustment:**
    - **Analysis**: Unemployment rate change is 0.29%.
    - **Action**: No adjustment made.
    - **Rationale**: Employment levels are stable, suggesting steady economic conditions.

**Updated Portfolio Allocations:**
  - Government Bonds - Long-Term: 22.72%
  - Corporate Bonds - Long-Term: 18.91%
  - Government Bonds - Intermediate-Term: 16.19%
  - Corporate Bonds - Intermediate-Term: 17.14%
  - Short-Term Bonds: 15.51%
  - TIPS: 1.90%
  - Nominal Bonds: 7.62%
```

### Interpreting the Report

The rebalancing report provides a detailed breakdown of the portfolio adjustments made based on the analysis of economic indicators. Here's how to interpret the report:

- **Date**: Indicates when the rebalancing took place.

- **Portfolio Adjustments**: Each adjustment corresponds to an economic indicator or a set of indicators.

  - **Analysis**: Provides insights into the current state of the economic indicator.

  - **Action**: Describes the adjustment made to the portfolio (e.g., increasing or decreasing allocations).

  - **Rationale**: Explains why the adjustment was made, linking economic conditions to investment strategy.

**Example Highlights:**

- **Inflation Adjustment**: Due to high inflation detected (composite change of 2.54%), the allocation to TIPS was increased by 9.00% to protect against inflation eroding real returns.

- **Inflation Expectations Adjustment**: Despite high current inflation, falling inflation expectations (change of -26.65%) led to a reduction in TIPS allocation by 8.00% and an increase in Nominal Bonds by the same amount, as nominal bonds become more attractive when future inflation is expected to decrease.

- **GDP Growth Adjustment**: Strong GDP growth (increase of 3.14%) resulted in a reduction of long-term bond allocations and an increase in short-term bonds to mitigate interest rate risk associated with economic expansion.

- **Recession Probability Adjustment**: Low recession probability (1.96%) prompted an increase in corporate bond allocations due to reduced default risk, enhancing potential returns.

**Updated Portfolio Allocations**:

This section provides the final allocation percentages after all adjustments have been made. It reflects how the portfolio has been rebalanced in response to the analyzed economic indicators.
