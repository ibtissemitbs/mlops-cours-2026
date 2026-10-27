"""Lance les 22 configurations classiques du TP."""

import os
import subprocess
import sys
from itertools import product

CONFIGS = [
    # 1 référence naïve
    ("dummy", []),
    # 4 valeurs de C x 2 pondérations = 8 essais
    *[
        ("logistic_regression", ["--C", str(c), "--class_weight", w])
        for c, w in product(
            [0.01, 0.1, 1.0, 10.0],
            ["none", "balanced"],
        )
    ],
    # 3 nombres d'arbres x 3 profondeurs = 9 essais
    *[
        ("random_forest", ["--n_estimators", str(n), "--max_depth", str(d)])
        for n, d in product([100, 200, 400], [4, 8, "none"])
    ],
    # 2 nombres d'arbres x 2 profondeurs = 4 essais
    *[
        ("gradient_boosting", ["--n_estimators", str(n), "--max_depth", str(d)])
        for n, d in product([100, 200], [2, 3])
    ],
]


def main() -> None:
    print(f"Lancement de {len(CONFIGS)} runs...", flush=True)
    env = {**os.environ, "LAUNCHED_BY": "grid_search"}

    for i, (model, args) in enumerate(CONFIGS, 1):
        cmd = [sys.executable, "-m", "src.train", "--model", model, *args]
        print(f"[{i}/{len(CONFIGS)}] {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, check=True, env=env)

    print("Tous les runs sont termines.", flush=True)


if __name__ == "__main__":
    main()
