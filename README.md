# activity1_rcaAD3326
# CIE-2 Activity – Unit 1 and Unit 2

## Included files
- `house_prices.csv` – sample data for Multiple Linear Regression.
- `multiple_linear_regression.py` – manual small-sample normal-equation calculation, multiple vs single-feature model, coefficients, MAE, RMSE, R², adjusted R², VIF, residual plot, and a new prediction.
- `student_pass.csv` – sample binary-classification data.
- `logistic_regression.py` – stratified split, Logistic Regression, predicted probabilities, confusion matrix and metrics at thresholds 0.3, 0.5 and 0.7, majority-class baseline, and a new prediction.
- `requirements.txt` – Python libraries required.

## How to run
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Run:
   - `python multiple_linear_regression.py`
   - `python logistic_regression.py`

You can also open the `.py` files in Jupyter/VS Code and run them.

## Notes
- The CSV datasets are small, illustrative classroom datasets created for this activity; they are not real market or student records.
- Exact scores can vary if you change the dataset, split settings, or software versions.
- For the regression coefficient interpretation, a coefficient is the expected change in predicted house price for a one-unit increase in that feature while the other included features are held constant.
- For Logistic Regression, the recommended threshold depends on whether false positives or false negatives are more costly. Explain your choice using the printed metrics.
