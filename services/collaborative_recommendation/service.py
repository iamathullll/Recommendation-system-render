import pickle
import os

MODEL_PATH = "services/collaborative_recommendation/model/ppr_results.pkl"

# =========================
# Load model safely
# =========================

print("Loading recommendation model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run train.py first."
    )

with open(MODEL_PATH, "rb") as f:
    ppr_results = pickle.load(f)

print(f"Model loaded successfully. Users in model: {len(ppr_results)}")


# =========================
# Recommendation function
# =========================

def get_recommendations(user_id, top_k=100, fallback=None):
    """
    Returns recommended post IDs for given user_id
    """

    user_id = str(user_id)

    # Personalized recommendations
    if user_id in ppr_results and ppr_results[user_id]:
        return ppr_results[user_id][:top_k]

    # Cold-start fallback
    if fallback is not None:
        return fallback[:top_k]

    return []


# =========================
# Optional test runner
# =========================

if __name__ == "__main__":

    print("\nTesting recommendation service...\n")

    test_user = input("Enter user_id: ")

    recs = get_recommendations(test_user, top_k=10)

    print("\nRecommendations:")
    print(recs)
