from fastapi import FastAPI, HTTPException
import pickle
import os

MODEL_PATH = "model/ppr_results.pkl"

app = FastAPI()

@app.on_event("startup")
def startup():

    global ppr_results

    if not os.path.exists(MODEL_PATH):
        raise Exception("Model file not found")

    with open(MODEL_PATH, "rb") as f:
        ppr_results = pickle.load(f)

    print("Model loaded")


@app.get("/")
def home():
    return {"status": "live"}


@app.get("/recommendations/{user_id}")
def recommend(user_id: str, top_k: int = 20):

    if user_id not in ppr_results:
        return {"recommendations": []}

    return {
        "user_id": user_id,
        "recommendations": ppr_results[user_id][:top_k]
    }
