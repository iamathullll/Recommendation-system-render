from recommendation_system.services.collaborative_recommendation.service import get_recommendations
from recommendation_system.services.content_recommendation.recommendation import (
    recommend_for_user_full,
    item_vectors,
    postid_to_idx,
    user_activity_col
)

import numpy as np
import faiss


def rerank_posts(user_id, top_k=100, alpha=0.5):

    # 1️⃣ Get candidates
    collab_posts = get_recommendations(user_id, top_k=top_k) or []

    content_result = recommend_for_user_full(user_id, top_k=top_k)
    content_posts = (
        content_result.get("final_posts", [])
        if isinstance(content_result, dict)
        else []
    )

    candidate_posts = []
    seen = set()

    for post in collab_posts + content_posts:
        if post not in seen:
            candidate_posts.append(post)
            seen.add(post)

    # 2️⃣ Build user profile
    doc = user_activity_col.find_one({"userId": user_id})
    if not doc:
        return candidate_posts[:top_k]

    liked_posts = [
        str(item.get("postId"))
        for item in doc.get("likes", [])
        if str(item.get("postId")) in postid_to_idx
    ]

    if not liked_posts:
        return candidate_posts[:top_k]

    idxs = [postid_to_idx[pid] for pid in liked_posts]

    user_profile = item_vectors[idxs].mean(axis=0, keepdims=True)
    faiss.normalize_L2(user_profile)

    # 3️⃣ Score all candidates
    scores = {}

    for post in candidate_posts:

        if post not in postid_to_idx:
            continue

        idx = postid_to_idx[post]
        item_vec = item_vectors[idx].reshape(1, -1)
        faiss.normalize_L2(item_vec)

        content_score = np.dot(user_profile, item_vec.T)[0][0]
        collab_score = 1 if post in collab_posts else 0

        final_score = alpha * collab_score + (1 - alpha) * content_score

        scores[post] = final_score

    # 4️⃣ Sort
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [post for post, _ in ranked[:top_k]]