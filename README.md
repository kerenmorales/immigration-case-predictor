# Immigration Case Outcome Predictor

AI-powered tool for Canadian immigration lawyers to predict refugee case outcomes using transformer models trained on 59k IRB decisions.

## Features

- Predict case outcome probability (Granted / Rejected / Uncertain)
- DistilBERT transformer model for high accuracy
- Confidence scores and key factor analysis
- Docker-ready for deployment

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch, Transformers
- **Frontend**: React, Vite, Tailwind CSS
- **Model**: DistilBERT fine-tuned on AsyLex dataset
- **Deployment**: Docker, docker-compose

## Quick Start

```bash
# 1. Train the model
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python download_data.py
python train_transformer.py  # ~20 min GPU, ~2 hours CPU

# 2. Run backend
uvicorn main:app --reload

# 3. Run frontend (new terminal)
cd frontend
npm install && npm run dev
```

## Deploy

```bash
docker-compose up --build
```

See [DEPLOY.md](./DEPLOY.md) for cloud deployment options (Railway, AWS, GCP).

## Dataset

[AsyLex](https://huggingface.co/datasets/clairebarale/AsyLex): 59k Canadian refugee decisions (1996-2022) with labeled outcomes.

## Disclaimer

For research purposes only. Not a substitute for legal advice.
