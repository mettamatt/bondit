# Bondit Portfolio Management Application

![Bondit Logo](logo.png)

## Overview

**Bondit** is a Python-based portfolio management tool designed to help long-term investors optimize their bond allocations based on key economic indicators. By analyzing data from sources like the Federal Reserve Economic Data (FRED), Bondit dynamically adjusts investment allocations to enhance wealth preservation, achieve modest growth, and safeguard against inflation and other economic risks.

## Features

- **Economic Data Analysis**: Fetches and processes a variety of economic indicators from FRED.
- **Dynamic Portfolio Management**: Manages investment allocations across various bond asset types with enforced minimum and maximum constraints.
- **Decision Engine**: Applies data-driven decision rules to adjust portfolio allocations based on current economic conditions.
- **Rebalancing Reports**: Generates detailed reports outlining portfolio adjustments for transparency and record-keeping.
- **Flexible Rebalancing Dates**: Allows specifying a rebalancing date for backtesting or historical analysis.

## Installation

### Prerequisites

- **Python 3.7** or higher
- **FRED API Key**: Obtain one from [FRED API Access](https://fred.stlouisfed.org/docs/api/fred/)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mettamatt/bondit.git
   cd bondit
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

   - **Copy the Configuration Template File**

     ```bash
     cp config/config.template.yaml config/config.yaml
     ```

   - **Edit `config/config.yaml`**

     Update the file with your FRED API key, desired file paths, and initial portfolio allocations:

     ```yaml
     api:
       fred_api_key: "YOUR_FRED_API_KEY_HERE"
       # The API key for accessing the FRED (Federal Reserve Economic Data) API.
       # This key is required to fetch economic data from the FRED service.
       # Replace the placeholder value with your actual API key obtained from FRED.

     storage:
       fred_file_path: "./data/fred_data.json"
       # The file path to the JSON file where the economic data retrieved from the FRED API
       # will be stored and cached. Adjust this path if you want to store the data in a different location
       # or with a different file name.

     allocations:
       VBIRX:
         initial: 40.0
         min: 10.0
         max: 70.0
         # Vanguard Short-Term Bond Index Fund: Provides exposure to short-term investment-grade bonds.
       VBLAX:
         initial: 40.0
         min: 10.0
         max: 70.0
         # Vanguard Long-Term Bond Index Fund: Provides exposure to long-term investment-grade bonds.
       VTAPX:
         initial: 20.0
         min: 0.0
         max: 30.0
         # Vanguard Short-Term Inflation-Protected Securities Index Fund: Protects against inflation by investing in TIPS.
     ```

     **Important:** Ensure that the sum of all `initial` allocation percentages equals **100%**. For example, in the above configuration:

     - VBIRX: 40.0%
     - VBLAX: 40.0%
     - VTAPX: 20.0%

     **Total:** 100.0%

     If the initial allocations do not sum to 100%, the application will raise an error during initialization.

## Usage

Run the application by executing the `main.py` script:

```bash
python src/main.py
```

### Optional: Specify a Rebalancing Date

You can specify a rebalancing date to analyze data up to a particular point in time. This is useful for backtesting or historical analysis.

```bash
python src/main.py --rebalancing_date YYYY-MM-DD
```

**Example:**

```bash
python src/main.py --rebalancing_date 2024-09-22
```

If no rebalancing date is provided, the application defaults to the current date.

### Workflow Steps

The application will perform the following steps:

1. **Initialize Loggers**: Sets up logging for the application and report generation.
2. **Load Configuration**: Reads and validates the configuration settings from `config/config.yaml`.
3. **Initialize Data Storage and Fetcher**: Sets up data storage mechanisms and initializes the data fetcher with the FRED API key.
4. **Data Collection**: Fetches economic indicators from FRED up to the specified rebalancing date.
5. **Portfolio Initialization**: Creates a new portfolio or loads an existing one based on the initial allocations.
6. **Decision Engine Execution**: Applies decision rules based on economic indicators to adjust the portfolio.
7. **Rebalancing Report Generation**: Produces a report detailing the portfolio adjustments.

## High-Level Economic Indicators and Decision Rules

Bondit leverages a set of high-level economic indicators to inform its decision-making process. Each indicator is associated with specific decision rules that dictate how the portfolio should be adjusted in response to changes in economic conditions.

### Economic Indicators

| Indicator Name                                                    | Series ID      | Description                                                     | Type      | Calculation Method            | Thresholds           | Time Frame Weights (Years: Weight)                  | Rule Weight |
|-------------------------------------------------------------------|----------------|-----------------------------------------------------------------|-----------|-------------------------------|----------------------|------------------------------------------------------|-------------|
| **Federal Funds Rate**                                            | FEDFUNDS       | Federal Funds Rate                                             | Monthly   | Z-Score                        | (1.0, -1.0)           | 1y: 0.30, 3y: 0.40, 5y: 0.30                        | 1.0         |
| **Consumer Price Index (CPI)**                                    | CPIAUCSL       | Composite CPI & PCE Inflation                                   | Monthly   | Year-Over-Year Change         | (2.5, 2.0)            | 1y: 1.00                                             | 0.9         |
| **Personal Consumption Expenditures Price Index (PCE)**           | PCEPI          | Composite CPI & PCE Inflation                                   | Monthly   | Year-Over-Year Change         | (2.5, 2.0)            | 1y: 1.00                                             | 0.9         |
| **5-Year Breakeven Inflation Rate**                               | T5YIE          | Breakeven Inflation                                            | Daily     | Basis Points Change           | (10, -10)             | 1y: 1.00                                             | 0.8         |
| **Yield Spread**                                                  | T10Y2Y         | Yield Spread                                                   | Daily     | Basis Points Change           | (50, -50)             | 1y: 0.20, 3y: 0.50, 5y: 0.30                        | 0.7         |
| **Recession Probabilities**                                       | RECPROUSM156N  | Recession Probability                                          | Monthly   | Current Value                 | (25.0, 15.0)          | 1y: 1.00                                             | 0.6         |
| **Moody's BAA Corporate Bond Yield Minus 10-Year Treasury Yield** | BAA10YM        | Credit Spread                                                  | Monthly   | Basis Points Change           | (50, -50)             | 1y: 0.25, 3y: 0.50, 5y: 0.25                        | 0.5         |
| **Real Gross Domestic Product (GDP)**                             | GDPC1          | GDP Growth Rate                                                | Quarterly | CAGR                           | (3.0, 2.0)            | 1y: 1.00                                             | 0.4         |
| **Unemployment Rate**                                             | UNRATE         | Unemployment Rate                                              | Monthly   | Absolute Change               | (0.5, -0.5)           | 1y: 1.00                                             | 0.3         |

### Time Frame Weights and Their Usage

Each economic indicator is analyzed across multiple time frames (in years), with each time frame assigned a specific weight. These **Time Frame Weights** determine the importance of each period in the overall analysis of the indicator. The weighted analysis allows Bondit to consider both short-term and long-term trends when making portfolio adjustment decisions.

**How Time Frame Weights are Used:**

1. **Multi-Time Frame Analysis:** For each indicator, data is analyzed over different time frames (e.g., 1-year, 3-year, 5-year periods). Each time frame provides insight into the indicator's behavior over varying durations.

2. **Weighted Aggregation:** The changes calculated for each time frame are multiplied by their respective weights. This weighted sum represents the overall influence of the indicator on the portfolio adjustments.

3. **Overall Trend Determination:** The aggregated weighted changes help in determining the overall trend (Rising, Falling, Stable) for each indicator, which in turn influences how the portfolio allocations are adjusted.

4. **Balancing Short-term and Long-term Insights:** By assigning different weights to various time frames, Bondit balances the immediate economic conditions with longer-term trends, ensuring a well-rounded investment strategy.

**Example:**

For the **Federal Funds Rate**, the time frame weights are:

- **1 Year:** 30%
- **3 Years:** 40%
- **5 Years:** 30%

This means that the 3-year analysis has the highest influence on the overall trend determination, ensuring that medium-term trends are given more consideration while still accounting for short-term and long-term movements.

### Decision Rules

Each economic indicator is associated with specific decision rules that determine how the portfolio should be adjusted. The **Rule Weight** signifies the importance of each indicator in the decision-making process.

| Rule Name                          | Economic Indicator                     | Description                                                                                   | Action Taken                                               |
|------------------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------|
| **Interest Rate Adjustment**       | Federal Funds Rate (`FEDFUNDS`)        | Adjusts allocations based on the Z-score of the Federal Funds Rate to mitigate interest rate risk.| Increase or decrease allocations to long-term bonds based on rate changes. |
| **Inflation Adjustment**           | CPI (`CPIAUCSL`), PCE (`PCEPI`)        | Adjusts allocations to TIPS based on composite Year-Over-Year changes in CPI and PCE.            | Increase TIPS allocations during high inflation; decrease during low inflation. |
| **Inflation Expectations Adjustment** | 5-Year Breakeven Inflation Rate (`T5YIE`) | Adjusts allocations based on basis points changes in long-term inflation expectations.           | Shift allocations between TIPS and nominal bonds based on inflation expectations. |
| **Yield Curve Adjustment**         | Yield Spread (`T10Y2Y`)                | Adjusts allocations based on changes in the yield spread to respond to economic outlook shifts. | Modify allocations to short-term vs. long-term bonds based on yield curve trends. |
| **Recession Probability Adjustment** | Recession Probabilities (`RECPROUSM156N`) | Adjusts allocations based on current recession probabilities to manage interest rate risk. | Increase allocations to long-term bonds when recession risk is high; increase short-term bonds when recession risk is low. |
| **Credit Spread Adjustment**       | Credit Spread (`BAA10YM`)              | Adjusts allocations based on changes in credit spreads to respond to credit market conditions.  | Shift allocations between short-term and long-term bonds based on credit spread changes. |
| **GDP Growth Adjustment**          | Real GDP (`GDPC1`)                      | Adjusts allocations based on GDP growth rates to align with economic expansion or contraction.  | Increase short-term bonds during strong GDP growth; decrease during weak growth. |
| **Employment Rate Adjustment**     | Unemployment Rate (`UNRATE`)            | Adjusts allocations based on changes in the unemployment rate to respond to labor market conditions.| Modify allocations to long-term vs. short-term bonds based on employment trends. |

### How They Influence Portfolio Adjustments

- **High Rule Weight**: Indicators like the Federal Funds Rate (`FEDFUNDS`) with higher rule weights have a more significant impact on portfolio adjustments.
- **Time Frame Weights**: Each indicator's analysis across multiple time frames is weighted to reflect the importance of short-term versus long-term trends.
- **Thresholds**: Each indicator has predefined thresholds that trigger specific actions when breached.
- **Action Taken**: Based on the analysis, the Decision Engine will increase or decrease allocations to specific assets to align the portfolio with current economic conditions.

## Rebalancing Reports

After applying the decision rules, Bondit generates a comprehensive rebalancing report that details the portfolio adjustments made based on the latest economic indicators.

### Sample Rebalancing Report

Below is a sample rebalancing report generated by the application:

```
## Rebalancing Report Date: 2024-10-02


### **Updated Portfolio Allocations**

| **Asset**             | **Allocation** |
|-----------------------|----------------|
| VBIRX                 | 51.25%        |
| VBLAX                 | 46.67%        |
| VTAPX                 | 2.08%        |

---

### **Economic Indicators Overview**

| **Economic Indicator**       | **Overall Trend** | **1-Year Signal** |
|------------------------------|--------------------|--------------------|
| Federal Funds Rate             | Stable             | Falling            |
| Consumer Price Index (CPI)     | Falling            | Falling            |
| Personal Consumption Expenditures Price Index (PCE) | Falling            | Falling            |
| 5-Year Breakeven Inflation Rate | Falling            | Falling            |
| Yield Spread                   | Stable             | Rising             |
| Recession Probabilities        | Falling            | Falling            |
| Moody's BAA Corporate Bond Yield Minus 10-Year Treasury Yield | Stable             | Stable             |
| Real Gross Domestic Product (GDP) | Stable             | Stable             |
| Unemployment Rate              | Stable             | Stable             |

---

### **Rebalancing Scenario**

#### **Portfolio Adjustments**

| **Adjustment Type**                | **Asset**    | **Action**                                     | **Amount** | **Rationale**                                                                                         |
|------------------------------------|--------------|------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------|
| **fed_funds**  | VBLAX | Decreased by 1.95%. | -1.95% | High interest rates can lead to falling bond prices; reducing duration mitigates risk. |
| **fed_funds**  | VBIRX | Increased by 1.95%. | +1.95% | High interest rates can lead to falling bond prices; reducing duration mitigates risk. |
| **cpi**  | VTAPX | Decreased by 1.92%. | -1.92% | Low inflation makes nominal bonds more attractive; reducing TIPS allocation accordingly. |
| **cpi**  | VBIRX | Increased by 0.96%. | +0.96% | Low inflation makes nominal bonds more attractive; reducing TIPS allocation accordingly. |
| **cpi**  | VBLAX | Increased by 0.96%. | +0.96% | Low inflation makes nominal bonds more attractive; reducing TIPS allocation accordingly. |
| **breakeven_inflation**  | VTAPX | Decreased by 16.00%. | -16.00% | Falling inflation expectations make nominal bonds more attractive; reducing TIPS allocation accordingly. |
| **breakeven_inflation**  | VBIRX | Increased by 8.00%. | +8.00% | Falling inflation expectations make nominal bonds more attractive; reducing TIPS allocation accordingly. |
| **breakeven_inflation**  | VBLAX | Increased by 8.00%. | +8.00% | Falling inflation expectations make nominal bonds more attractive; reducing TIPS allocation accordingly. |
| **recession_prob**  | VBLAX | Decreased by 0.34%. | -0.34% | Low recession probability suggests rising interest rates; reducing duration mitigates interest rate risk. |
| **recession_prob**  | VBIRX | Increased by 0.34%. | +0.34% | Low recession probability suggests rising interest rates; reducing duration mitigates interest rate risk. |

---

### **Notes**

- **Overall Trend:** Represents the most frequently occurring trend signal derived from the analysis of all economic indicators.
- **1-Year Trend Signal:** Reflects the trend signal specifically for each economic indicator over the past year.

---

**Report Generated by:** Bondit v1.0 
---
```

### Interpreting the Report

The rebalancing report provides a detailed breakdown of the portfolio adjustments made based on the analysis of economic indicators. Here's how to interpret the report:

- **Date**: Indicates when the rebalancing took place.

- **Updated Portfolio Allocations**: Shows the current allocation percentages of each asset in the portfolio after adjustments.

- **Economic Indicators Overview**: Summarizes the overall trend and 1-year signal for each economic indicator analyzed.

- **Rebalancing Scenario**: Details each portfolio adjustment, including the type of adjustment, the asset affected, the action taken, the amount of change, and the rationale behind the decision.

- **Notes**: Provides additional context on how trends are derived and their implications.

**Example Highlights:**

- **fed_funds**: Due to high interest rates detected (z-score > 1.0), the allocation to long-term bonds (VBLAX) was decreased by 1.95%, and the allocation to short-term bonds (VBIRX) was increased by 1.95% to mitigate interest rate risk.

- **cpi**: Low inflation detected (Year-Over-Year Change < 2.0%) led to a decrease in TIPS allocation (VTAPX) by 1.92% and an increase in nominal bonds (VBIRX and VBLAX) by 0.96% each.

- **breakeven_inflation**: Falling inflation expectations prompted a significant reduction in TIPS allocation (VTAPX) by 16.00% and corresponding increases in nominal bonds (VBIRX and VBLAX) by 8.00% each.

- **recession_prob**: Low recession probability resulted in a slight decrease in long-term bond allocations (VBLAX) by 0.34% and a corresponding increase in short-term bonds (VBIRX) by 0.34%.

**Updated Portfolio Allocations**:

This section provides the final allocation percentages after all adjustments have been made. It reflects how the portfolio has been rebalanced in response to the analyzed economic indicators.
