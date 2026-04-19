import time
import subprocess
from datetime import datetime

LAST_RUN_FILE = "pipeline/last_run.txt"


def has_run_today():
    try:
        with open(LAST_RUN_FILE, "r") as f:
            last_run = f.read().strip()
            return last_run == datetime.now().strftime("%Y-%m-%d")
    except FileNotFoundError:
        return False


def mark_as_run():
    with open(LAST_RUN_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))


def run_job():
    print("Running scheduled training pipeline...")

    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "pipeline.train"],
            check=True,
            capture_output=True,
            text=True
        )

        print("Training completed successfully")
        print(result.stdout)

        mark_as_run()

    except subprocess.CalledProcessError as e:
        print("Training failed")
        print(e.stderr)


print("Scheduler started... watching time...")

while True:
    now = datetime.now()
    print(f"Checking time: {now}")

    # Saturday = 5, Noon = 12:00
    if (
        now.weekday() == 5 and
        now.hour == 12 and
        now.minute == 0 and
        not has_run_today()
    ):
        run_job()
        time.sleep(60)  # prevent double execution

    time.sleep(10)