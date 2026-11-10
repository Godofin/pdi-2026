#!/usr/bin/env bash
# Ver comentário em start_mlflow.ps1 sobre --artifacts-destination vs --default-artifact-root.
cd "$(dirname "$0")/.."
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts \
    --host 127.0.0.1 \
    --port 5000 \
    --workers 1  # SQLite: 1 worker evita disputa de lock (ver start_mlflow.ps1)
