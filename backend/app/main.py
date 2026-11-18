
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.classifier import predict_department, load_classifier
from app.services.urgency import calculate_urgency, recommend_action
from app.services.similarity import find_similar


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the main classification model once at startup
    print("Loading GrievanceGPT DistilBERT model...")

    load_classifier()

    app.state.classifier_ready = True

    print("DistilBERT loaded successfully!")
    print("GrievanceGPT API is ready.")

    yield


app = FastAPI(
    title="GrievanceGPT API",
    description="AI-powered public grievance classification and analysis",
    version="1.0.0",
    lifespan=lifespan
)


# Allow communication with the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class AnalyzeRequest(BaseModel):
    complaint: str = Field(min_length=5, max_length=5000)


@app.get("/")
def root():
    return {
        "message": "Welcome to GrievanceGPT API",
        "status": "running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "GrievanceGPT API",
        "classifier_loaded": getattr(
            app.state, "classifier_ready", False
        )
    }


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):

    complaint = payload.complaint.strip()

    if len(complaint) < 5:
        raise HTTPException(
            status_code=422,
            detail="Complaint must contain at least 5 characters."
        )

    try:
        # 1. Department classification
        department, confidence, top = predict_department(complaint)

        # 2. Urgency and sentiment analysis
        score, priority, sentiment, keywords = calculate_urgency(
            complaint
        )

        # 3. Semantic similarity search
        similar = find_similar(complaint)

        # 4. Recommended action
        action = recommend_action(priority)

        return {
            "complaint": complaint,
            "predicted_department": department,
            "classification_confidence": confidence,
            "top_predictions": top,
            "urgency_score": score,
            "priority": priority,
            "sentiment": round(sentiment, 4),
            "detected_keywords": keywords,
            "recommended_action": action,
            "similar_complaints": similar
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:
        print(f"Analysis error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(exc)}"
        )
