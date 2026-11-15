# Insurance Charges Prediction (Linear Regression)

A machine learning project that predicts individual medical insurance charges from personal attributes such as age, BMI, smoking status and region.

## Dataset
`insurance.csv` contains 1,338 records with 7 columns:

| Column | Description |
|--------|-------------|
| age | Age of the policyholder |
| sex | male / female |
| bmi | Body mass index |
| children | Number of dependents covered |
| smoker | yes / no |
| region | northeast, northwest, southeast, southwest |
| charges | Medical insurance cost billed (target variable) |

## Workflow
The full analysis is in `insurance_charges_prediction.ipynb`.

1. **Exploratory data analysis**: shape, null checks, histograms, count plots, box plots and a correlation heatmap
2. **Data cleaning and preprocessing**: duplicate removal, label encoding of `sex` and `smoker`, one-hot encoding of `region`
3. **Feature engineering**: BMI categories (underweight, healthy, overweight, obese), one-hot encoded
4. **Feature scaling**: `StandardScaler` applied to age, bmi and children
5. **Feature selection**: Pearson correlation for numeric features and chi-square test (on quartile bins of charges) for categorical features
6. **Modelling**: Linear Regression with an 80/20 train/test split (`random_state=42`)

**Final features used:** age, isfemale, region_southeast, bmi, children, issmoker, bmicat_obese

## Results
| Metric | Value |
|--------|-------|
| R-squared | 0.804 |
| Adjusted R-squared | 0.799 |
| Mean Squared Error | 36,003,101.65 |

The model explains about 80% of the variation in insurance charges on the test set.

## Tech stack
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, SciPy, Jupyter

## How to run
```bash
git clone https://github.com/vanshika1234578/insurance-charges-ml.git
cd insurance-charges-ml
pip install -r requirements.txt
jupyter notebook insurance_charges_prediction.ipynb
```
Keep `insurance.csv` in the same folder as the notebook.
