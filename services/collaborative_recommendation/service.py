import pickle
import os

MODEL_PATH = "services/collaborative_recommendation/model/ppr_results.pkl"

ppr_results = None


def load_model():
    global ppr_results

    if ppr_results is None:

        if not os.path.exists(MODEL_PATH):
            print("⚠ Collaborative model not found. It will be created at startup.")
            return None

        print("Loading collaborative recommendation model...")

        with open(MODEL_PATH, "rb") as f:
            ppr_results = pickle.load(f)

        print(f"Model loaded successfully. Users in model: {len(ppr_results)}")

    return ppr_results


def get_recommendations(user_id, top_k=100, fallback=None):
    """
    Returns recommended post IDs for given user_id
    """

    model = load_model()

    if model is None:
        return []

    user_id = str(user_id)

    # Personalized recommendations
    if user_id in model and model[user_id]:
        return model[user_id][:top_k]

    # Cold-start fallback
    if fallback is not None:
        return fallback[:top_k]

    return []


if __name__ == "__main__":

    print("\nTesting recommendation service...\n")

    test_user = input("Enter user_id: ")

    recs = get_recommendations(test_user, top_k=10)

    print("\nRecommendations:")
    print(recs)