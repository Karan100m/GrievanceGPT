# GrievanceGPT

Local full-stack public grievance intelligence prototype.

## Stack
- Frontend: React 18, Vite 6, Recharts, Lucide, Axios
- Backend: FastAPI, PyTorch, Hugging Face Transformers, Sentence Transformers
- No database is required for this first local version.

## Required model/data assets
Copy these from your Google Drive project into this repository:
- `models/best_transformer_augmented/` (entire Hugging Face model directory)
- `models/complaint_embeddings.npy`
- `data/processed/complaints_corpus.csv`
- Optionally copy original CSV files into `data/raw/`.

The complaint corpus CSV must contain `grievance_id`, `text`, and `category` columns.
The Sentence-BERT model `all-MiniLM-L6-v2` downloads from Hugging Face the first time semantic search is used.

## macOS Apple Silicon setup

### Backend
```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

From the `backend` directory:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
API docs: http://127.0.0.1:8000/docs

### Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Notes
- Dashboard overview counts and chart distributions are illustrative placeholders because this version has no database. The AI inspection page uses live API results.
- Urgency scoring is a prototype heuristic, not an emergency dispatch system. Critical cases require human review.
- The training/holdout data is synthetic; reported evaluation scores should not be interpreted as real-world government deployment accuracy.
