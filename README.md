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
## 🤖 Ollama Setup (Local AI)

Ollama allows supported language models to run locally on your computer. If the project's AI features are configured to use Ollama, follow these steps.

### 1. Install Ollama

Download Ollama for macOS from:

https://ollama.com/download

Install and open the application.

### 2. Verify the installation

```bash
ollama --version
```

### 3. Download the model

This project can use the Mistral model if its AI integration is configured for it.

```bash
ollama pull mistral
```

### 4. Start Ollama

Open a separate Terminal window and run:

```bash
ollama serve
```

Keep the Ollama service running. If the service is already running, you may see a message that the port is in use; in that case, you generally do not need to start a second server.

### 5. Test the model

```bash
ollama run mistral
```

Enter a test prompt, then type `/bye` to exit the interactive session.

### 6. Start AutoML-Agent-Lab

In another Terminal window:

```bash
cd ~/Downloads/AutoML-Agent-Lab
conda activate python312
python app.py
```

Open the local URL configured by Flask, normally:

http://127.0.0.1:5004/workspace

### Troubleshooting Ollama

Check whether the service responds:

```bash
curl http://localhost:11434/api/tags
```

If the request succeeds, Ollama returns information about locally available models.

If the Mistral model is missing, run:

```bash
ollama pull mistral
```

**Note:** Ollama is optional unless the project's implemented AI features require it. Running Ollama alone does not automatically connect it to the Flask application; the application must be configured to call the Ollama API.


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
