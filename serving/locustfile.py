import os
import random
import sys
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config  # noqa: E402

config = load_config(ROOT / "dataset.yaml")
features = pd.read_csv(ROOT / "data/processed/test.csv")[config.features]
rows = features.astype(object).where(features.notna(), None).to_dict("records")


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.3)

    def on_start(self):
        self.client.headers["Host"] = os.environ["ISVC_HOST"]

    @task
    def predict(self):
        self.client.post("/predict", json={"instances": [random.choice(rows)]})
