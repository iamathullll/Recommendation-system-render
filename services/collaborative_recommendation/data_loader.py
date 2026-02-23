from pymongo import MongoClient
import pandas as pd
import numpy as np

from .config import (
    MONGO_URI,
    USER_DB,
    USER_COLLECTION,
    POST_DB,
    POST_COLLECTION,
    INTERACTION_DB,
    INTERACTION_COLLECTION
)

# ============================================================
# Interaction weights
# ============================================================

INTERACTION_TYPE_WEIGHT = {
    "view": 1.0,
    "like": 3.0,
    "comment": 4.0,
    "share": 5.0,
    "dislike": -3.0
}


# ============================================================
# Mongo Connection
# ============================================================

def connect_mongo():
    """
    Create MongoDB connection
    """
    if not MONGO_URI:
        raise ValueError("MONGO_URI not found in environment")

    client = MongoClient(MONGO_URI)
    return client


# ============================================================
# Load Users
# ============================================================

def load_users(client):

    print("Loading users...")

    collection = client[USER_DB][USER_COLLECTION]

    cursor = collection.find(
        {},
        {
            "_id": 1,
            "followers": 1
        }
    )

    users_df = pd.DataFrame(list(cursor))

    if users_df.empty:
        print("Warning: No users found")
        return pd.DataFrame(columns=["user_id", "followers"])

    users_df.rename(columns={"_id": "user_id"}, inplace=True)

    users_df["user_id"] = users_df["user_id"].astype(str)

    users_df["followers"] = users_df["followers"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    print(f"Users loaded: {len(users_df)}")

    return users_df


# ============================================================
# Load Posts
# ============================================================

def load_posts(client):

    print("Loading posts...")

    collection = client[POST_DB][POST_COLLECTION]

    cursor = collection.find(
        {},
        {
            "_id": 1,
            "userId": 1,
            "createdAt": 1
        }
    )

    posts_df = pd.DataFrame(list(cursor))

    if posts_df.empty:
        print("Warning: No posts found")
        return pd.DataFrame(columns=["post_id", "user_id", "created_at"])

    posts_df.rename(columns={
        "_id": "post_id",
        "userId": "user_id",
        "createdAt": "created_at"
    }, inplace=True)

    posts_df["post_id"] = posts_df["post_id"].astype(str)
    posts_df["user_id"] = posts_df["user_id"].astype(str)

    posts_df["created_at"] = pd.to_datetime(
        posts_df["created_at"],
        errors="coerce"
    )

    posts_df.dropna(subset=["created_at"], inplace=True)

    print(f"Posts loaded: {len(posts_df)}")

    return posts_df


# ============================================================
# Load Interactions
# ============================================================

def load_interactions(client):

    print("Loading interactions...")

    collection = client[INTERACTION_DB][INTERACTION_COLLECTION]

    cursor = collection.find(
        {},
        {
            "user_id": 1,
            "all_detailed_interactions": 1
        }
    )

    rows = []

    doc_count = 0
    interaction_count = 0

    for doc in cursor:

        doc_count += 1

        user_obj_id = doc.get("user_id")

        if not user_obj_id:
            continue

        user_id = str(user_obj_id)

        interactions = doc.get("all_detailed_interactions", [])

        if not isinstance(interactions, list):
            continue

        for interaction in interactions:

            post_obj_id = interaction.get("post_id")
            interaction_type = interaction.get("type")

            if not post_obj_id or not interaction_type:
                continue

            weight = INTERACTION_TYPE_WEIGHT.get(
                interaction_type,
                0
            )

            rows.append({
                "user_id": user_id,
                "post_id": str(post_obj_id),
                "interaction_score": weight
            })

            interaction_count += 1

    print(f"Documents scanned: {doc_count}")
    print(f"Interactions extracted: {interaction_count}")

    clean_df = pd.DataFrame(rows)

    if clean_df.empty:
        print("Warning: No interactions found")
        return pd.DataFrame(columns=["user_id", "post_id", "interaction_score"])

    # ============================================================
    # Aggregate interactions per user-post pair
    # ============================================================

    clean_df = (
        clean_df.groupby(["user_id", "post_id"], as_index=False)
        .interaction_score.sum()
    )

    # ============================================================
    # SIGNED LOG SCALING (Option 2)
    # ============================================================

    clean_df["interaction_score"] = (
        np.sign(clean_df["interaction_score"]) *
        np.log1p(np.abs(clean_df["interaction_score"]))
    )

    print(f"Final interactions: {len(clean_df)}")

    return clean_df

# ============================================================
# Load All Data
# ============================================================

def load_all_data():

    print("Connecting to MongoDB...")

    client = connect_mongo()

    try:

        users_df = load_users(client)
        posts_df = load_posts(client)
        interactions_df = load_interactions(client)

        print("All data loaded successfully")

        return users_df, posts_df, interactions_df

    finally:

        client.close()
        print("MongoDB connection closed")
