## Dataset Information

- **Dataset Name:** WA_Fn-UseC_-Telco-Customer-Churn.csv
- **Source:** IBM Sample Data (available on Kaggle)
- **Domain:** Telecommunications
- **Number of Records:** 7043
- **Number of Features:** 21 (20 input features + 1 target)
- **Target Variable:** Churn
- **Problem Type:** Binary Classification



## Initial Inspection

- `SeniorCitizen` and `tenure` are stored as integers (`int64`).
- `MonthlyCharges` is stored as a floating-point value (`float64`).
- Most remaining features are categorical (`object` datatype).
- `TotalCharges` was initially stored as an object because it contained blank values.

## Target Variable Distribution

- Churn = No : 5174 customers
- Churn = Yes : 1869 customers

### Observation

The dataset is moderately imbalanced, with approximately 73.5% of customers retained and 26.5% having churned.

## Hidden Missing Values

Although no explicit null values were detected using `isnull()`, the `TotalCharges` column contained 11 blank string values.
These blank strings prevented pandas from recognizing the column as numeric.


## Data Cleaning Decision

The `TotalCharges` column contained 11 missing values after converting blank strings to `NaN`.

Investigation showed that all affected customers had:

- tenure = 0
- Contract = Two year
- Churn = No

These customers had recently joined the telecom service and had not yet completed their first billing cycle.

Therefore, replacing the missing `TotalCharges` values with **0** was considered a business-appropriate solution, and no records were removed from the dataset.