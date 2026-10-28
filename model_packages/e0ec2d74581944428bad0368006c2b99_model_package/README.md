# AutoML-Agent-Lab Model Package

Model ID: `e0ec2d74581944428bad0368006c2b99`

## Model information

- Model: Logistic Regression
- Task: classification
- Target: Loan_Status
- Features: 12

## Files

- `model.joblib` — saved model and preprocessing pipeline
- `metadata.json` — target, feature names, and model metadata
- `predict.py` — command-line prediction tool
- `requirements.txt` — Python dependencies

## Setup

Use Python 3.12 or a compatible Python version. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Predict from a CSV

The input CSV must contain all required feature columns. It must not need
the target column.

```bash
python predict.py --input your_data.csv --output predictions.csv
```

The output CSV preserves the input columns and adds `prediction`.
For compatible classifiers, class probabilities are included too.

## Important

Only use trusted model files. Loading a joblib model can execute code.
