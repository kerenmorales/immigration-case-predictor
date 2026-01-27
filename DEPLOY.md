# Deployment Guide

## Local Development

### Train the Model First (Required)

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset
python download_data.py

# Train transformer model (recommended, ~1-2 hours on CPU, ~20 min on GPU)
python train_transformer.py

# OR train simple model (faster, ~5 min)
python train_model.py
```

### Run Locally

```bash
# Backend
cd backend
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# Access at http://localhost (frontend) and http://localhost:8000 (API)
```

Note: You need to train the model first and mount the `models/` directory.

---

## Cloud Deployment Options

### Option 1: Railway (Easiest)

1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Create new project → Deploy from GitHub
4. Add both `backend` and `frontend` as services
5. Set environment variables:
   - Frontend: `VITE_API_URL=https://your-backend.railway.app`

### Option 2: AWS (Production)

**Backend on ECS/Fargate:**

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t immigration-predictor-api ./backend
docker tag immigration-predictor-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/immigration-predictor-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/immigration-predictor-api:latest
```

**Frontend on S3 + CloudFront:**

```bash
cd frontend
npm run build
aws s3 sync dist/ s3://your-bucket-name --delete
```

### Option 3: Google Cloud Run

```bash
# Backend
cd backend
gcloud run deploy immigration-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated

# Frontend
cd frontend
npm run build
# Deploy to Firebase Hosting or Cloud Storage
```

### Option 4: Hugging Face Spaces (Free, Good for Demo)

1. Create a new Space at huggingface.co/spaces
2. Choose "Docker" as SDK
3. Upload the backend code
4. The model can be loaded directly from HF Hub

---

## Model Storage

For production, store the trained model in:
- **AWS S3** - Download on container startup
- **Hugging Face Hub** - Push model with `model.push_to_hub("your-username/immigration-predictor")`
- **Google Cloud Storage** - Similar to S3

Example loading from HF Hub (modify `main.py`):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("your-username/immigration-predictor")
tokenizer = AutoTokenizer.from_pretrained("your-username/immigration-predictor")
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL (frontend) | `https://api.yourapp.com` |
| `MODEL_PATH` | Path to model (backend) | `models/transformer/final` |

---

## GPU Considerations

The transformer model runs on CPU by default. For faster inference:

- **AWS**: Use `g4dn.xlarge` instances or SageMaker
- **GCP**: Use Cloud Run with GPU or Vertex AI
- **Railway/Render**: CPU-only (still works, ~1-2s per prediction)

For CPU deployment, consider using ONNX for faster inference:

```bash
pip install optimum onnxruntime
optimum-cli export onnx --model models/transformer/final models/onnx/
```
