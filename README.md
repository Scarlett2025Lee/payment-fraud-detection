# Payment Fraud Detection & Risk Modeling

A credit-card fraud analysis project comparing Logistic Regression and XGBoost under severe class imbalance. The workflow covers exploratory analysis, cost-based threshold selection, model comparison, SHAP explanations, and false-positive/false-negative analysis.

## Data

The notebook uses the ULB credit-card fraud dataset: 284,807 transactions, 492 fraud cases (0.1727%), anonymized V1–V28 features, Time, Amount, and Class.

Download `creditcard.csv` from the [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The raw dataset is not bundled. `Time` is elapsed time from the first recorded transaction; the derived Hour variable is relative, not a verified local clock hour.

## Workflow

1. Check completeness, class balance, transaction amounts, and relative-time patterns.
2. Train scaled Logistic Regression, including a class-weighted exploration.
3. Use a stratified 60%/20%/20% train/validation/test split for the later model comparisons.
4. Compare weighted and unweighted XGBoost.
5. Select thresholds on validation predictions using an illustrative cost function: `500 × false negatives + 5 × false positives`.
6. Inspect global and case-specific SHAP explanations and analyze errors.

## Recorded notebook results

These figures come from the uploaded notebook's saved outputs; they were not regenerated during repository preparation. PR-AUC is computed with scikit-learn's `average_precision_score` (average precision).

| Model | Threshold | Precision | Recall | Average precision | False negatives | False positives | Simulated cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.010 | 0.3660 | 0.8776 | 0.7449 | 12 | 149 | 6,745 |
| Weighted XGBoost | 0.050 | 0.1789 | 0.8980 | 0.8607 | 10 | 404 | 7,020 |
| Unweighted XGBoost | 0.006 | 0.3177 | 0.8980 | 0.8764 | 10 | 189 | 5,945 |

Unweighted XGBoost reduced the illustrative cost by 11.9% versus the listed Logistic Regression operating point. These are simulated cost units, not demonstrated financial savings. Threshold grids differ between models, so the comparison also reflects those search choices.

## Open and run

- [Cleaned notebook](payment_fraud_detection_clean.ipynb): recommended starting point; skips the initial partial-file debugging and two failed exploratory cells.
- [Original notebook source](payment_fraud_detection_xgboost.ipynb): preserves all original source cells, including exploratory errors.

Both copies have execution outputs cleared for a compact repository. The original uploaded notebook remains the source of the recorded results above. No new training results were generated.

Use Google Colab or a Jupyter environment. Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the complete `creditcard.csv` at `/content/creditcard.csv` in Colab, or change the notebook's data path for local use. Run the cleaned notebook from the top. It should report 284,807 rows and zero missing values. The original source notebook contains an `archive.zip` extraction step from the learning session; the cleaned version starts after that debugging sequence.

## Validation and limitations

All cleaned code cells passed Python syntax parsing. Full training was not rerun because the dataset was not supplied. Dependencies are listed without historical version pins; package versions can affect reproduced results.

The earlier exploratory section selects thresholds on test predictions; the later section switches to validation-based selection. Repeated inspection of the same test set and comparison across models mean these results should be treated as exploratory, not a pristine final holdout assessment. A time-based split, fresh holdout, threshold sensitivity analysis, and realistic business loss estimates would strengthen a deployment study.

SHAP explains the model's behavior on anonymized features; it does not establish causal drivers of fraud. This is a portfolio analysis, not a production payment risk service.
