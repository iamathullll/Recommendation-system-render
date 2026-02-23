# Follow Recommendation System

Hybrid graph-based follow recommendation system.

## Features
- Social graph scoring
- Interaction similarity
- Popularity fallback
- Cold-start safe
- Offline precomputation

## Architecture
Offline batch job + Online lightweight serving

## Run Training
python app/train_follow_model.py

## Serving
Use follow_service.get_follow_recommendations(user_id)
