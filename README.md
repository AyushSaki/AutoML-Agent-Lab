# AutoML-Agent-Lab

A free, local-first autonomous machine-learning platform built with **Python + Flask + Scikit-learn + XGBoost + Ollama**.

## Features

- CSV / Excel upload
- Dataset profiling
- Missing-value and duplicate analysis
- Automatic task detection
- Classification models
- Regression models
- Model comparison
- Basic evaluation metrics
- Local LLM explanations through Ollama
- Light, responsive Flask dashboard
- Pytest starter tests
- Docker-ready project structure

## Requirements

- macOS / Linux / Windows
- Python 3.12 recommended
- Anaconda/Miniconda optional
- Ollama optional for the AI explanation feature

## Setup on your Mac

```bash
cd AutoML-Agent-Lab

conda activate python312

pip install -r requirements.txt
```

Check Python:

```bash
python --version
```

Start the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Ollama

Install Ollama from the official Ollama website, then:

```bash
ollama pull mistral
ollama run mistral
```

The application expects:

```text
http://localhost:11434
```

You can change the model in `.env`:

```text
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
```

## Project workflow

```text
Dataset
   ↓
Flask Upload
   ↓
Dataset Profiling
   ↓
Target Selection
   ↓
Task Detection
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model
   ↓
Ollama Explanation
```

## Run tests

```bash
pytest -q
```

## Important

This is a strong starter architecture. For a production version, add:
- persistent experiment database
- MLflow tracking
- SHAP visualizations
- hyperparameter optimization
- model artifact registry
- background training jobs
- authentication
- dataset validation
- model download/deployment endpoints

## GitHub

After testing:

```bash
git init
git add .
git commit -m "Initial AutoML Agent Lab"
git branch -M main
git remote add origin https://github.com/AyushSaki/AutoML-Agent-Lab.git
git push -u origin main
```
