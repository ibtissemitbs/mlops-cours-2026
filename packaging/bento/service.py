"""API BentoML : prédiction avec le modèle scikit-learn."""

from typing import Annotated

import bentoml
import bentoml.sklearn
import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, create_model

MODEL_TAG = "scoring_model:latest"

META = bentoml.models.get(MODEL_TAG).info.metadata
FEATURES = [*META["numeric"], *META["categorical"]]
THRESHOLD = float(META["threshold"])

Numeric = Annotated[
    float | None,
    Field(strict=True, allow_inf_nan=False),
]
Categorical = Annotated[
    str | None,
    Field(strict=True, min_length=1),
]

Instance = create_model(
    "Instance",
    __config__=ConfigDict(extra="forbid"),
    **{
        column: (Numeric, ...)
        for column in META["numeric"]
    },
    **{
        column: (Categorical, ...)
        for column in META["categorical"]
    },
)


@bentoml.service(
    name="scoring_bento",
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class ScoringBento:
    bento_model = bentoml.models.BentoModel(MODEL_TAG)

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(
            self.bento_model
        )

    @bentoml.api
    def predict(
        self,
        instances: Annotated[
            list[Instance],
            Field(min_length=1, max_length=1000),
        ],
    ) -> dict:
        df = pd.DataFrame(
            [instance.model_dump() for instance in instances],
            columns=FEATURES,
        )

        for column in META["numeric"]:
            df[column] = df[column].astype("float64")

        for column in META["categorical"]:
            df[column] = df[column].where(
                df[column].notna(), np.nan
            )

        probabilities = self.model.predict_proba(df)[:, 1]

        return {
            "model_version": META["model_version"],
            "positive_class": META["positive_class"],
            "threshold": THRESHOLD,
            "predictions": [
                {
                    "probability": float(probability),
                    "prediction": int(
                        probability >= THRESHOLD
                    ),
                }
                for probability in probabilities
            ],
        }
