from fastapi import FastAPI, HTTPException
import pickle
import os

MODEL_PATH = "follow_recommendations.pkl"

app = FastAPI(
    title="Follow Recommendation API",
    version="1.0"
)


# Load model at startup
@app.on_event("startup")
def load_model():

    global follow_recs

    print("Loading follow recommendation model...")

    if not os.path.exists(MODEL_PATH):
        raise Exception("follow_recommendations.pkl not found. Run train_follow.py first.")

    with open(MODEL_PATH, "rb") as f:
        follow_recs = pickle.load(f)

    print(f"Model loaded successfully. Users: {len(follow_recs)}")


# Health check endpoint
@app.get("/")
def health_check():

    return {
        "status": "live",
        "service": "follow recommendation"
    }


# Follow recommendation endpoint
@app.get("/follow-recommendations/{user_id}")
def get_follow_recommendations(user_id: str, top_k: int = 20):

    user_id = str(user_id)

    if user_id not in follow_recs:

        return {
            "user_id": user_id,
            "recommendations": [],
            "count": 0
        }

    recs = follow_recs[user_id][:top_k]

    return {
        "user_id": user_id,
        "recommendations": recs,
        "count": len(recs)
    }
