import time
import datetime
import logging
from train_pipeline import train_and_build

# Logging setup
logging.basicConfig(
    filename="ml/artifacts/daily_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RUN_HOUR = 2   # 2 AM daily

def should_run_now():
    now = datetime.datetime.now()
    return now.hour == RUN_HOUR and now.minute == 0

def run_daily():
    logging.info("Daily scheduler started.")

    last_run_date = None

    while True:
        now = datetime.datetime.now()

        # Run once per day
        if should_run_now():
            if last_run_date != now.date():
                try:
                    logging.info("Starting daily training...")
                    train_and_build()
                    logging.info("Training completed successfully.")
                    last_run_date = now.date()
                except Exception as e:
                    logging.exception("Training failed.")

        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    run_daily()
