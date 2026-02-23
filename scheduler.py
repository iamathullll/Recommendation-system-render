import time
import datetime
import logging
import subprocess
import sys

# ============================================================
# CONFIG
# ============================================================

DAILY_RUN_HOUR = 2
FOLLOW_INTERVAL_HOURS = 8

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# TRAINING FUNCTIONS (ALL VIA SUBPROCESS)
# ============================================================

def run_module(module_name, model_name):
    logging.info(f"Starting {model_name} training...")

    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True
    )

    logging.info(f"{model_name} training completed.")


def train_content():
    run_module(
        "services.content_recommendation.ml.training",
        "Content"
    )


def train_collaborative():
    run_module(
        "services.collaborative_recommendation.train",
        "Collaborative"
    )


def train_follow():
    run_module(
        "services.follow_recommendation.train_follow_model",
        "Follow"
    )


# ============================================================
# SCHEDULER LOOP
# ============================================================

def run_scheduler():

    logging.info("Unified scheduler started.")

    last_daily_run = None
    last_follow_run = None

    while True:
        now = datetime.datetime.now()

        # -----------------------------------------
        # DAILY: Content + Collaborative at 2 AM
        # -----------------------------------------
        if now.hour == DAILY_RUN_HOUR and now.minute == 0:
            if last_daily_run != now.date():
                try:
                    train_content()
                    train_collaborative()
                    last_daily_run = now.date()
                except Exception:
                    logging.exception("Daily training failed.")

        # -----------------------------------------
        # FOLLOW: Every 8 hours
        # -----------------------------------------
        if (
            last_follow_run is None or
            (now - last_follow_run).total_seconds() >= FOLLOW_INTERVAL_HOURS * 3600
        ):
            try:
                train_follow()
                last_follow_run = now
            except Exception:
                logging.exception("Follow training failed.")

        time.sleep(60)


if __name__ == "__main__":
    run_scheduler()