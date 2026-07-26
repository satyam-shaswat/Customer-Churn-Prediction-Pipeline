# Dataset Source

- Dataset Name: WA_Fn-UseC_-Telco-Customer-Churn.csv
- Source:
- Domain:
- Number of Records:  21 rows
- Number of Features: the dataset has 7043 columns.
- Target Variable:
- Problem Type:



senior citizen , tenure == int64
MonthlyCharges == float64
all other columns == str


Churn
No     5174
Yes    1869


there are 11 rows where TotalCharges is just empty whitespace, out of 21 columns total.


| Feature | Category | Description | Possible Values | Business Importance | Initial Hypothesis |
| customer-id| -------- | ----------- | --------------- | ------------------- | ------------------ |




## Data Cleaning Decision

`TotalCharges` contained 11 missing values.

Upon investigation, all affected customers had:
- tenure = 0
- Contract = Two year
- Churn = No

These customers are new subscribers who have not completed their first billing cycle. Therefore, `TotalCharges` was imputed with 0 instead of removing the records.