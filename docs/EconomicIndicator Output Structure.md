# EconomicIndicator Output Structure

Understanding the output structure of the `EconomicIndicator` class is essential for interpreting the analysis results and integrating them into your investment decision-making process. This section provides a detailed explanation of the return structure from the `analyze_indicator` method, including examples to illustrate the data format and contents.

---

## Return Structure Overview

The `analyze_indicator` method of the `EconomicIndicator` class returns a dictionary containing comprehensive analysis results for the specific economic indicator. The return structure includes the following key components:

- **Time Frame Analysis**: Results for each specified time frame.
- **Weighted Analysis**: Aggregated weighted changes across time frames.
- **Overall Trend**: The overall trend signal based on the analysis.
- **Statistics**: Basic statistical metrics calculated from the data.
- **Error Handling**: Error messages if data is insufficient or missing.

### Structure Format

```python
{
    "1y": { ... },
    "3y": { ... },
    "5y": { ... },
    "weighted": {
        "weighted_change": float or None
    },
    "overall_trend": "Rising" or "Falling" or "Stable",
    "statistics": { ... },
    "error": str (optional)
}
```

- **Time Frames (`"1y"`, `"3y"`, etc.)**: Each key corresponds to a time frame, containing analysis results specific to that period.
- **Weighted Analysis (`"weighted"`)**: Contains the aggregated weighted change.
- **Overall Trend (`"overall_trend"`)**: The final trend signal derived from the analysis.
- **Statistics (`"statistics"`)**: Includes statistical measures like mean, median, standard deviation, etc.
- **Error (`"error"`)**: Present if there's an issue with the data or analysis.

---

## Detailed Explanation of Return Components

### 1. Time Frame Analysis

For each time frame specified in `time_frame_weights`, the analysis results include:

- **`"current_value"`**: The most recent data point within the time frame.
- **`"change"`**: The calculated change based on the specified calculation method.
- **`"cagr"`**: Compound Annual Growth Rate (if applicable).
- **`"signal"`**: The trend signal ('Rising', 'Falling', 'Stable') based on thresholds.
- **`"z_score"`**: Z-score of the most recent value (if applicable).
- **`"year_over_year_change"`**: Year-over-year change percentage (if applicable).

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

### 2. Weighted Analysis

Aggregates the changes from different time frames using their respective weights.

- **`"weighted_change"`**: The sum of weighted changes divided by the total weight.

**Example**:

```python
"weighted": {
    "weighted_change": 1.83
}
```

### 3. Overall Trend

Determines the overall trend based on the majority of trend signals across time frames.

- **`"overall_trend"`**: 'Rising', 'Falling', or 'Stable'.

**Example**:

```python
"overall_trend": "Rising"
```

### 4. Statistics

Provides basic statistical metrics calculated from the entire data series.

- **`"most_recent_value"`**: The latest value in the data series.
- **`"average"`**: Mean of the data series.
- **`"median"`**: Median value.
- **`"min"`**: Minimum value.
- **`"max"`**: Maximum value.
- **`"std_dev"`**: Standard deviation.
- **`"z_score"`**: Z-score of the most recent value.

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

### 5. Error Handling

If the analysis cannot be performed due to missing or insufficient data, an error message is included.

- **`"error"`**: Describes the issue encountered during analysis.

**Example**:

```python
"error": "No data available for indicator 'CPI'"
```

---

## Sample Return Structure

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
    "3y": {
        "current_value": 250.5,
        "change": 5.10,
        "cagr": None,
        "signal": "Rising",
        "z_score": None,
        "year_over_year_change": None
    },
    "5y": {
        "current_value": 250.5,
        "change": 10.25,
        "cagr": None,
        "signal": "Rising",
        "z_score": None,
        "year_over_year_change": None
    },
    "weighted": {
        "weighted_change": 4.58
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
  - **3-Year and 5-Year Analyses**:
    - **Change**: Cumulative percentage changes over the periods.
    - **Signal**: 'Rising' based on the thresholds.
- **Weighted Change**:
  - Calculated by applying the `time_frame_weights` to the changes from each time frame.
- **Overall Trend**:
  - 'Rising' due to the majority of time frames indicating an upward trend.
- **Statistics**:
  - Provides context on the data distribution and how the current value compares historically.

---

## Integrating the Return Structure

The return structure is designed to be easily integrated into:

- **Reporting Tools**: Generate summaries or detailed reports for investors.
- **Decision-Making Processes**: Use the trend signals and weighted changes to inform portfolio rebalancing.
- **Dashboards**: Visualize the analysis results for quick insights.

**Example Usage**:

```python
# Instantiate the EconomicIndicator
cpi_indicator = EconomicIndicator(config=cpi_config, data=cpi_data)

# Perform analysis
analysis_results = cpi_indicator.analyze_indicator()

# Access overall trend
overall_trend = analysis_results.get("overall_trend")

# Access weighted change
weighted_change = analysis_results.get("weighted", {}).get("weighted_change")

# Access statistics
statistics = analysis_results.get("statistics")

# Use the results in your application
if overall_trend == "Rising":
    # Adjust bond portfolio accordingly
    pass
```

---

## Customization and Extension

You can customize the analysis and the return structure by:

- **Modifying the Analysis Methods**:
  - Extend the `EconomicIndicator` class to include additional calculations or data points.
- **Adjusting the Return Data**:
  - Include or exclude specific metrics based on your requirements.
- **Handling Errors and Exceptions**:
  - Implement additional error handling to manage data quality issues.

---

## Conclusion

The return structure from the `EconomicIndicator` class provides a comprehensive set of analysis results that are crucial for informed bond portfolio rebalancing. By understanding the contents and format of the returned data, you can effectively integrate the analysis into your investment strategies and decision-making tools.
