# Economic Indicator Analysis for Bond Portfolio Rebalancing

This document provides a comprehensive overview of the economic indicators used in the bond investment tool for infrequent portfolio rebalancing. It explains the financial analysis, mathematical calculations, decision rules, and how users can manipulate the system to suit their investment strategies. Tables are incorporated to present key information clearly and facilitate easier understanding.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Economic Indicators Overview](#economic-indicators-overview)
   - [Federal Funds Rate](#1-federal-funds-rate)
   - [Consumer Price Index (CPI)](#2-consumer-price-index-cpi)
   - [Personal Consumption Expenditures Price Index (PCE)](#3-personal-consumption-expenditures-price-index-pce)
   - [5-Year Breakeven Inflation Rate](#4-5-year-breakeven-inflation-rate)
   - [Yield Spread](#5-yield-spread)
   - [Recession Probabilities](#6-recession-probabilities)
   - [Credit Spread](#7-credit-spread)
   - [Real Gross Domestic Product (GDP)](#8-real-gross-domestic-product-gdp)
   - [Unemployment Rate](#9-unemployment-rate)
3. [Mathematical Calculations](#mathematical-calculations)
   - [Calculation Methods](#calculation-methods)
   - [Trend Signal Generation](#trend-signal-generation)
4. [Financial Analysis Process](#financial-analysis-process)
5. [Decision Rules](#decision-rules)
   - [Overview](#overview)
   - [List of Decision Rules](#list-of-decision-rules)
   - [Decision Rules Logic](#decision-rules-logic)
   - [Customizing Decision Rules](#customizing-decision-rules)
6. [Manipulating the Analysis](#manipulating-the-analysis)
   - [Adjusting Thresholds](#adjusting-thresholds)
   - [Changing Weights](#changing-weights)
   - [Adding New Indicators](#adding-new-indicators)
7. [Analysis Output Structure](#analysis-output-structure)
8. [Conclusion](#conclusion)

---

## Introduction

The bond investment tool is designed to assist investors in making informed decisions for portfolio rebalancing based on key economic indicators. By analyzing trends and changes in these indicators, the tool provides insights into market conditions that may affect bond investments. Understanding the economic indicators, the mathematical calculations behind them, and the decision rules applied is crucial for leveraging the tool effectively.

---

## Economic Indicators Overview

Below is a detailed explanation of each economic indicator used in the tool, including its significance in bond investment and how it's analyzed. Key configurations for each indicator are summarized in the table below for quick reference.

### Economic Indicators Configuration

| **Indicator**                      | **Series ID**    | **Calculation Method**      | **Time Frames (Years)** | **Weights**         | **Thresholds**          | **Rule Weight** |
|------------------------------------|------------------|-----------------------------|-------------------------|---------------------|-------------------------|------------------|
| **Federal Funds Rate**             | FEDFUNDS         | Z-score                     | 1, 3, 5                 | 30%, 40%, 30%       | ±1.0                    | 1.0              |
| **Consumer Price Index (CPI)**     | CPIAUCSL         | Year-over-year change       | 1                       | 1.0                 | 2.5%, 2.0%              | 0.9              |
| **Personal Consumption Expenditures (PCE)** | PCEPI          | Year-over-year change       | 1                       | 1.0                 | 2.5%, 2.0%              | 0.9              |
| **5-Year Breakeven Inflation Rate**| T5YIE            | Basis points change         | 1                       | 1.0                 | ±10 bps                 | 0.8              |
| **Yield Spread**                   | T10Y2Y           | Basis points change         | 1, 3, 5                 | 20%, 50%, 30%       | ±50 bps                 | 0.7              |
| **Recession Probabilities**        | RECPROUSM156N    | Current value               | 1                       | 1.0                 | 25%, 15%                | 0.6              |
| **Credit Spread**                  | BAA10YM          | Basis points change         | 1, 3, 5                 | 25%, 50%, 25%       | ±50 bps                 | 0.5              |
| **Real Gross Domestic Product (GDP)** | GDPC1           | Compound Annual Growth Rate | 1                       | 1.0                 | 3.0%, 2.0%              | 0.4              |
| **Unemployment Rate**              | UNRATE           | Absolute change             | 1                       | 1.0                 | ±0.5 percentage points  | 0.3              |

### 1. Federal Funds Rate

**Description**: The Federal Funds Rate is the interest rate at which depository institutions lend reserve balances to other depository institutions overnight.

**Importance in Bond Investment**:
- **Interest Rate Benchmark**: Serves as a benchmark for other interest rates, including those on bonds.
- **Monetary Policy Indicator**: Reflects the Federal Reserve's monetary policy stance, influencing bond yields and prices.

**Analysis in the Tool**:
- **Calculation Method**: Z-score
- **Time Frames**: 1, 3, and 5 years with weights of 30%, 40%, and 30%, respectively.
- **Thresholds**: Upper and lower z-score thresholds of 1.0 and -1.0.
- **Purpose**: Detects significant deviations from the mean, indicating potential shifts in monetary policy affecting bond yields.

### 2. Consumer Price Index (CPI)

**Description**: CPI measures the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services.

**Importance in Bond Investment**:
- **Inflation Gauge**: High inflation erodes bond returns; CPI helps investors anticipate inflation trends.
- **Interest Rate Influence**: Influences central bank decisions on interest rates, affecting bond prices.

**Analysis in the Tool**:
- **Calculation Method**: Year-over-year change
- **Time Frame**: Focused on 1-year data.
- **Thresholds**: Upper and lower inflation targets of 2.5% and 2.0%.
- **Purpose**: Assesses inflation levels to anticipate interest rate adjustments impacting bond investments.

### 3. Personal Consumption Expenditures Price Index (PCE)

**Description**: PCE measures the prices paid by consumers for goods and services and is considered by the Federal Reserve when setting monetary policy.

**Importance in Bond Investment**:
- **Preferred Inflation Measure**: Often preferred by the Federal Reserve over CPI.
- **Monetary Policy Indicator**: Helps predict changes in interest rates affecting bond yields.

**Analysis in the Tool**:
- **Calculation Method**: Year-over-year change
- **Time Frame**: Focused on 1-year data.
- **Thresholds**: Inflation targets of 2.5% and 2.0%.
- **Purpose**: Monitors inflation trends for bond yield and price implications.

### 4. 5-Year Breakeven Inflation Rate

**Description**: The breakeven inflation rate represents the difference between the nominal yield on a fixed-rate investment and the real yield on an inflation-linked investment of the same maturity.

**Importance in Bond Investment**:
- **Inflation Expectations**: Reflects market expectations of inflation over the next five years.
- **Investment Strategy**: Helps in deciding between nominal and inflation-protected bonds.

**Analysis in the Tool**:
- **Calculation Method**: Basis points change
- **Time Frame**: Recent data with a 1-year focus.
- **Thresholds**: Thresholds of ±10 basis points.
- **Purpose**: Detects shifts in inflation expectations influencing bond selection.

### 5. Yield Spread

**Description**: The yield spread is the difference between yields on differing debt instruments of varying maturities, credit ratings, issuer, or risk levels.

**Importance in Bond Investment**:
- **Economic Indicator**: A narrowing spread can signal economic downturns; a widening spread can indicate growth.
- **Risk Assessment**: Helps assess the risk-return trade-off in bond investments.

**Analysis in the Tool**:
- **Calculation Method**: Basis points change
- **Time Frames**: 1, 3, and 5 years with weights of 20%, 50%, and 30%, respectively.
- **Thresholds**: Thresholds of ±50 basis points.
- **Purpose**: Monitors changes in economic outlook affecting bond yields.

### 6. Recession Probabilities

**Description**: Recession probabilities estimate the likelihood of an economic recession occurring in the near future.

**Importance in Bond Investment**:
- **Market Sentiment**: Anticipating recessions can guide defensive investment strategies.
- **Bond Demand**: During recessions, bonds often become more attractive due to their relative safety.

**Analysis in the Tool**:
- **Calculation Method**: Current value
- **Time Frame**: Focused on recent data with a 1-year scope.
- **Thresholds**: Thresholds at 25% (upper) and 15% (lower).
- **Purpose**: Evaluates recession risk to adjust bond portfolio defensively.

### 7. Credit Spread

**Description**: The credit spread is the difference in yield between securities with different credit qualities.

**Importance in Bond Investment**:
- **Default Risk Indicator**: Wider spreads indicate higher perceived risk of default.
- **Investment Decision**: Guides allocation between corporate bonds and safer government bonds.

**Analysis in the Tool**:
- **Calculation Method**: Basis points change
- **Time Frames**: 1, 3, and 5 years with weights of 25%, 50%, and 25%, respectively.
- **Thresholds**: Thresholds of ±50 basis points.
- **Purpose**: Assesses credit market conditions impacting bond risk premiums.

### 8. Real Gross Domestic Product (GDP)

**Description**: Real GDP measures the value of economic output adjusted for price changes (inflation or deflation).

**Importance in Bond Investment**:
- **Economic Health**: Strong GDP growth can lead to higher interest rates.
- **Interest Rate Impact**: Influences bond yields and prices.

**Analysis in the Tool**:
- **Calculation Method**: Compound Annual Growth Rate (CAGR)
- **Time Frame**: Focused on 1-year data.
- **Thresholds**: Thresholds at 3.0% (upper) and 2.0% (lower).
- **Purpose**: Evaluates economic growth trends affecting bond yields.

### 9. Unemployment Rate

**Description**: The unemployment rate measures the percentage of the total labor force that is unemployed but actively seeking employment.

**Importance in Bond Investment**:
- **Economic Indicator**: High unemployment may lead to lower interest rates.
- **Policy Influence**: Affects central bank policies impacting bond markets.

**Analysis in the Tool**:
- **Calculation Method**: Absolute change
- **Time Frame**: Focused on recent changes within 1 year.
- **Thresholds**: Thresholds of ±0.5 percentage points.
- **Purpose**: Monitors labor market conditions influencing monetary policy and bond yields.

---

## Mathematical Calculations

Understanding the mathematical calculations behind each indicator is essential for interpreting the analysis results accurately. Tables below summarize the calculation methods and their applications.

### Calculation Methods

The tool uses various calculation methods tailored to each indicator's characteristics. Each method is summarized in the table below.

| **Calculation Method**         | **Formula**                                                                                         | **Usage**                                                                                 | **Indicators Using This Method**                      | **Example Calculation**                                                                                                                                      |
|--------------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Percentage Change**          | $\text{Percentage Change (\%)} = \left( \frac{\text{Current Value} - \text{Start Value}}{\text{Start Value}} \right) \times 100$ | Assessing growth or decline over a period.                                               | *Not directly used in provided indicators*            | If value increases from 100 to 110: $\text{Percentage Change} = \left( \frac{110 - 100}{100} \right) \times 100 = 10\%$                                      |
| **Basis Points Change**        | $\text{Basis Points Change} = (\text{Current Value} - \text{Start Value}) \times 100$          | Measures small changes in interest rates or spreads. 1 bps = 0.01%.                      | 5-Year Breakeven Inflation Rate, Yield Spread, Credit Spread | If yield spread increases from 1.50% to 2.00%: $\text{Basis Points Change} = (2.00\% - 1.50\%) \times 100 = 50 \text{ bps}$                              |
| **Z-Score**                    | $Z = \frac{\text{Current Value} - \mu}{\sigma}$                                               | Identifies how many standard deviations the current value is from the mean.               | Federal Funds Rate                                     | If current value is 2.5%, mean is 1.5%, and std dev is 0.5%: $Z = \frac{2.5\% - 1.5\%}{0.5\%} = 2.0$                                                      |
| **Absolute Change**            | $\text{Absolute Change} = \text{Current Value} - \text{Start Value}$                           | Measures the direct difference in values, suitable for rates or levels.                    | Unemployment Rate                                      | If unemployment rate decreases from 5.0% to 4.5%: $\text{Absolute Change} = 4.5\% - 5.0\% = -0.5\%$                                                          |
| **Year-over-Year Change**      | $\text{YoY Change (\%)} = \left( \frac{\text{Current Value} - \text{Value One Year Ago}}{\text{Value One Year Ago}} \right) \times 100$ | Compares the current value to the same period in the previous year; accounts for seasonality. | CPI, PCE                                                | If CPI is 250 this year and was 245 last year: $\text{YoY Change} = \left( \frac{250 - 245}{245} \right) \times 100 \approx 2.04\%$                           |
| **Compound Annual Growth Rate (CAGR)** | $\text{CAGR (\%)} = \left( \left( \frac{\text{Ending Value}}{\text{Beginning Value}} \right)^{\frac{1}{n}} - 1 \right) \times 100$ | Measures the mean annual growth rate over a period longer than one year.                     | Real GDP                                               | If GDP grows from \$18 trillion to \$20 trillion over 2 years: $\text{CAGR} = \left( \left( \frac{20}{18} \right)^{\frac{1}{2}} - 1 \right) \times 100 \approx 5.52\%$ |
| **Current Value**              | $\text{Current Value} = \text{Most Recent Data Point}$                                        | Uses the most recent data point directly; suitable when the current level is of immediate interest. | Recession Probabilities                                | If the latest recession probability is 20%, the current value is 20%.                                                                                          |

### Trend Signal Generation

The tool generates trend signals based on calculated changes and predefined thresholds. The process is summarized in the table below.

| **Trend Signal** | **Condition**                                      | **Description**                                        |
|------------------|----------------------------------------------------|--------------------------------------------------------|
| **Rising**       | Change > Upper Threshold                           | Indicates a significant upward trend in the indicator.  |
| **Falling**      | Change < Lower Threshold                           | Indicates a significant downward trend in the indicator.|
| **Stable**       | Lower Threshold ≤ Change ≤ Upper Threshold         | Indicates no significant change; the indicator is stable.|
| **No Data**      | Change is `None` or `NaN`                          | Indicates insufficient data to determine a trend.       |

**Example**:

- **CPI YoY Change**: 2.6%
- **Thresholds**: Upper = 2.5%, Lower = 2.0%
- **Signal**: Rising (since 2.6% > 2.5%)

---

## Financial Analysis Process

The tool performs the following steps for each indicator:

1. **Data Processing**:
   - Cleans and structures raw data into a time series.
   - Handles missing values and ensures data alignment.

2. **Time Frame Extraction**:
   - Extracts data for specified time frames (e.g., 1, 3, 5 years).
   - Applies weights to each time frame based on its importance.

3. **Statistical Calculations**:
   - Computes mean, median, standard deviation, and other statistics.
   - Calculates changes using the specified calculation method.

4. **Trend Signal Generation**:
   - Compares calculated changes against thresholds.
   - Assigns trend signals for each time frame.

5. **Weighted Analysis**:
   - Aggregates changes across time frames using their weights.
   - Determines an overall trend for the indicator.

6. **Decision Support**:
   - Uses the overall trend and weighted changes to inform bond investment decisions.
   - Adjusts portfolio allocation based on signals (e.g., increase long-term bonds if interest rates are falling).

---

## Decision Rules

Decision Rules are the core logic that translates the analysis of economic indicators into actionable portfolio adjustments. They systematically evaluate the results of the economic indicators and determine how the bond portfolio should be rebalanced to optimize performance and manage risk.

### Overview

The Decision Rules module analyzes the outputs from the economic indicators and applies a set of predefined rules to decide on portfolio adjustments. Each rule focuses on a specific indicator or a combination of indicators and determines whether to increase, decrease, or maintain certain bond allocations based on the analysis.

### List of Decision Rules

The tool implements the following decision rules:

1. **Interest Rate Adjustment** (`Federal Funds Rate`)
2. **Inflation Adjustment** (`CPI` and `PCE`)
3. **Inflation Expectations Adjustment** (`5-Year Breakeven Inflation Rate`)
4. **Yield Curve Adjustment** (`Yield Spread`)
5. **Recession Probability Adjustment** (`Recession Probabilities`)
6. **Credit Spread Adjustment** (`Credit Spread`)
7. **GDP Growth Adjustment** (`Real GDP`)
8. **Employment Rate Adjustment** (`Unemployment Rate`)

### Decision Rules Logic

Each decision rule follows a structured approach to evaluate economic conditions and adjust the portfolio accordingly. The process is summarized in the table below.

| **Step**                      | **Description**                                                                                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **1. Retrieve Indicator Data** | Extract relevant data points (e.g., z-score, change percentages) from the analyzed indicators.                     |
| **2. Compare Against Thresholds** | Determine if the indicator's current value exceeds predefined upper or lower thresholds.                           |
| **3. Determine Action**        | Decide whether to increase, reduce, or maintain allocations based on the comparison.                              |
| **4. Calculate Adjustment Amount** | Use rule weights and maximum adjustment limits to determine the percentage change in allocations.                  |
| **5. Apply Adjustments**       | Modify the portfolio's bond allocations based on the determined actions.                                         |
| **6. Log Decisions**           | Document the reasoning and actions taken for transparency and review.                                            |

#### Example: Interest Rate Adjustment

| **Component**        | **Details**                                                                                                   |
|----------------------|----------------------------------------------------------------------------------------------------------------|
| **Indicator**        | Federal Funds Rate                                                                                            |
| **Data Point**       | Z-score                                                                                                        |
| **Thresholds**       | Upper = 1.0, Lower = -1.0                                                                                      |
| **Logic**            | - **Rising**: If z-score > 1.0, reduce long-term bond allocations to mitigate interest rate risk.<br>- **Falling**: If z-score < -1.0, increase long-term bond allocations to capitalize on favorable interest rates.<br>- **Stable**: No adjustment. |
| **Action**           | - Reduce or increase allocations in specific bond categories based on the trend signal.                        |
| **Rationale**        | - **Rising**: High interest rates can lead to falling bond prices; reducing duration mitigates risk.<br>- **Falling**: Low interest rates can lead to rising bond prices; increasing duration can enhance returns. |
| **Adjustment Amount**| Calculated using `rule_weight` and `max_adjustment` parameters.                                               |

### Customizing Decision Rules

Users can tailor the Decision Rules to better fit their investment strategies and risk profiles by adjusting parameters such as thresholds, rule weights, and maximum adjustment limits.

#### Adjusting Thresholds

- **Purpose**: Modify the sensitivity of decision rules to changes in economic indicators.
- **How to Adjust**:
  - Locate the thresholds in `src/config.py` within each `IndicatorConfig`.
  - Update the `thresholds` tuple to reflect desired sensitivity.

**Example**:

```python
# Original thresholds for Federal Funds Rate
thresholds=(1.0, -1.0)

# Adjusted thresholds for higher sensitivity
thresholds=(1.5, -1.5)
```

#### Changing Rule Weights

- **Purpose**: Alter the influence each decision rule has on portfolio adjustments.
- **How to Adjust**:
  - Modify the `rule_weight` attribute in `IndicatorConfig` for the relevant indicator.

**Example**:

```python
# Original rule weight for Unemployment Rate
rule_weight=0.3

# Increased weight for greater impact
rule_weight=0.6
```

#### Adjusting Maximum Adjustment Limits

- **Purpose**: Control the maximum percentage by which the portfolio can be adjusted in response to a single rule.
- **How to Adjust**:
  - Update the `max_adjustment` parameter in the `DecisionEngine` configuration.

**Example**:

```python
# Original maximum adjustment
max_adjustment=5.0  # Represents 5%

# Increased maximum adjustment
max_adjustment=7.5  # Represents 7.5%
```

#### Adding or Removing Decision Rules

- **Purpose**: Incorporate new economic indicators or retire outdated ones to keep the tool aligned with current market conditions.
- **How to Adjust**:
  - Define new `IndicatorConfig` entries in `src/config.py`.
  - Implement corresponding decision rules in `src/decision_rules/decision_rules.py`.

**Example**:

```python
IndicatorConfig(
    series_id="NEW_INDICATOR",
    name="New Economic Indicator",
    internal_key="new_indicator",
    description="Description of the new indicator",
    indicator_type="MONTHLY",
    time_frame_weights={1: 0.5, 3: 0.3, 5: 0.2},
    calculation_method="percentage_change",
    thresholds=(1.0, -1.0),
    rule_weight=0.5,
)
```

---

## Manipulating the Analysis

Users can customize the analysis to align with their investment strategies by adjusting configurations.

### Adjusting Thresholds

**Purpose**: Modify sensitivity to changes in indicators.

**How to Adjust**:

- **Locate the IndicatorConfig** for the desired indicator in `src/config.py`.
- **Modify the `thresholds` tuple**:
  - Increase thresholds to reduce sensitivity (fewer signals).
  - Decrease thresholds to increase sensitivity (more signals).

**Example**:

```python
# Original thresholds for CPI
thresholds=(2.5, 2.0)

# Adjusted thresholds for higher sensitivity
thresholds=(2.2, 1.8)
```

### Changing Weights

**Purpose**: Alter the influence of time frames or indicators on the overall analysis.

**How to Adjust**:

- **Time Frame Weights**:
  - In `IndicatorConfig`, modify the `time_frame_weights` dictionary.
  - Increase the weight of shorter time frames for a focus on recent trends.

**Example**:

```python
# Original time frame weights for Yield Spread
time_frame_weights={1: 0.20, 3: 0.50, 5: 0.30}

# Adjusted weights emphasizing recent data
time_frame_weights={1: 0.50, 3: 0.30, 5: 0.20}
```

- **Rule Weights**:
  - Adjust `rule_weight` to change the indicator's impact on portfolio adjustments.

**Example**:

```python
# Original rule weight for Unemployment Rate
rule_weight=0.3

# Increased weight for greater impact
rule_weight=0.6
```

### Adding New Indicators

**Purpose**: Incorporate additional economic indicators into the analysis.

**How to Add**:

1. **Define a New IndicatorConfig** in `src/config.py`:

   - Assign a unique `series_id`, `name`, `internal_key`, and other attributes.
   - Specify `calculation_method`, `thresholds`, and `time_frame_weights`.

2. **Fetch Data for the New Indicator**:

   - Ensure data is available in the required format (list of dictionaries with `date` and `value`).

3. **Instantiate EconomicIndicator**:

   - Use the new `IndicatorConfig` and data to create an `EconomicIndicator` instance.

**Example**:

```python
IndicatorConfig(
    series_id="NEW_INDICATOR",
    name="New Economic Indicator",
    internal_key="new_indicator",
    description="Description of the new indicator",
    indicator_type="MONTHLY",
    time_frame_weights={1: 0.5, 3: 0.3, 5: 0.2},
    calculation_method="percentage_change",
    thresholds=(1.0, -1.0),
    rule_weight=0.5,
)
```

---

## Analysis Output Structure

Understanding the output structure of the `EconomicIndicator` class is essential for interpreting the analysis results and integrating them into your investment decision-making process. This section provides a detailed explanation of the return structure from the `analyze_indicator` method, including examples to illustrate the data format and contents.

### Return Structure Overview

The `analyze_indicator` method of the `EconomicIndicator` class returns a dictionary containing comprehensive analysis results for the specific economic indicator. The return structure includes the following key components:

- **Time Frame Analysis**: Results for each specified time frame.
- **Weighted Analysis**: Aggregated weighted changes across time frames.
- **Overall Trend**: The overall trend signal based on the analysis.
- **Statistics**: Basic statistical metrics calculated from the data.
- **Error Handling**: Error messages if data is insufficient or missing.

### Structure Format

| **Key**            | **Type**                     | **Description**                                                                                         |
|--------------------|------------------------------|---------------------------------------------------------------------------------------------------------|
| **`1y`**, **`3y`**, **`5y`** | `dict`                       | Analysis results for each time frame. Includes `current_value`, `change`, `cagr`, `signal`, `z_score`, `year_over_year_change`. |
| **`weighted`**     | `dict`                       | Contains `weighted_change`, the aggregated weighted change across time frames.                          |
| **`overall_trend`**| `str`                        | The final trend signal derived from the analysis (`"Rising"`, `"Falling"`, `"Stable"`).                |
| **`statistics`**   | `dict`                       | Basic statistical metrics such as `most_recent_value`, `average`, `median`, `min`, `max`, `std_dev`, `z_score`. |
| **`error`**        | `str` (optional)             | Describes any issues encountered during analysis (e.g., missing data).                                  |

### Detailed Explanation of Return Components

#### 1. Time Frame Analysis

For each time frame specified in `time_frame_weights`, the analysis results include:

| **Field**                 | **Description**                                                                             |
|---------------------------|---------------------------------------------------------------------------------------------|
| **`current_value`**       | The most recent data point within the time frame.                                          |
| **`change`**              | The calculated change based on the specified calculation method.                           |
| **`cagr`**                | Compound Annual Growth Rate (if applicable).                                               |
| **`signal`**              | The trend signal (`'Rising'`, `'Falling'`, `'Stable'`) based on thresholds.               |
| **`z_score`**             | Z-score of the most recent value (if applicable).                                          |
| **`year_over_year_change`** | Year-over-year change percentage (if applicable).                                        |

**Example**:

```python
"1y": {
    "current_value": 250.5,
    "change": 2.04,
    "cagr": None,
    "signal": "Rising",
    "z_score": None,
    "year_over_year_change": 2.04
}
```

#### 2. Weighted Analysis

Aggregates the changes from different time frames using their respective weights.

| **Field**           | **Description**                                             |
|---------------------|-------------------------------------------------------------|
| **`weighted_change`** | The sum of weighted changes divided by the total weight.    |

**Example**:

```python
"weighted": {
    "weighted_change": 1.83
}
```

#### 3. Overall Trend

Determines the overall trend based on the majority of trend signals across time frames.

| **Field**           | **Description**                                                                                 |
|---------------------|-------------------------------------------------------------------------------------------------|
| **`overall_trend`** | `'Rising'`, `'Falling'`, or `'Stable'` indicating the aggregated trend signal.                  |

**Example**:

```python
"overall_trend": "Rising"
```

#### 4. Statistics

Provides basic statistical metrics calculated from the entire data series.

| **Field**               | **Description**                                         |
|-------------------------|---------------------------------------------------------|
| **`most_recent_value`** | The latest value in the data series.                    |
| **`average`**           | Mean of the data series.                                |
| **`median`**            | Median value of the data series.                       |
| **`min`**               | Minimum value in the data series.                      |
| **`max`**               | Maximum value in the data series.                      |
| **`std_dev`**           | Standard deviation of the data series.                 |
| **`z_score`**           | Z-score of the most recent value relative to the series. |

**Example**:

```python
"statistics": {
    "most_recent_value": 250.5,
    "average": 245.3,
    "median": 246.0,
    "min": 240.1,
    "max": 250.5,
    "std_dev": 3.2,
    "z_score": 1.62
}
```

#### 5. Error Handling

If the analysis cannot be performed due to missing or insufficient data, an error message is included.

| **Field**   | **Description**                                                                 |
|-------------|---------------------------------------------------------------------------------|
| **`error`** | Describes the issue encountered during analysis (e.g., missing data).           |

**Example**:

```python
"error": "No data available for indicator 'CPI'"
```

### Sample Return Structure

Below is an example of the return structure from the `EconomicIndicator` class after analyzing the Consumer Price Index (CPI) indicator.

```python
{
    "1y": {
        "current_value": 250.5,
        "change": 2.04,
        "cagr": None,
        "signal": "Rising",
        "z_score": None,
        "year_over_year_change": 2.04
    },
    "weighted": {
        "weighted_change": 2.04
    },
    "overall_trend": "Rising",
    "statistics": {
        "most_recent_value": 250.5,
        "average": 245.3,
        "median": 246.0,
        "min": 240.1,
        "max": 250.5,
        "std_dev": 3.2,
        "z_score": 1.62
    }
}
```

**Explanation**:

- **Time Frame Results**:
  - **1-Year Analysis**:
    - **Change**: 2.04% YoY increase.
    - **Signal**: 'Rising' since the change exceeds the upper threshold of 2.5%.
- **Weighted Change**:
  - Calculated by applying the `time_frame_weights` to the changes from each time frame.
- **Overall Trend**:
  - 'Rising' due to the majority of time frames indicating an upward trend.
- **Statistics**:
  - Provides context on the data distribution and how the current value compares historically.

---

## Conclusion

This document outlines the framework and components of the economic indicator analysis used for bond portfolio rebalancing. By understanding each economic indicator, the associated mathematical calculations, and the decision rules, users can effectively utilize the bond investment tool to make informed portfolio adjustments. Customizing the analysis parameters allows investors to tailor the tool to their specific investment strategies and risk profiles, ensuring optimal performance and risk management in their bond investments.
