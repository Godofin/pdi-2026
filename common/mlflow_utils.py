"""Utilitários compartilhados pelos 4 projetos: setup do MLflow, logging de figuras/tabelas e seeds."""
from __future__ import annotations

import inspect
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: scripts rodam em terminal/CI
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


def setup_mlflow(experiment_name: str) -> str:
    """Aponta para o tracking server (ou MLFLOW_TRACKING_URI) e seleciona o experimento."""
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if uri.startswith("http"):
        import requests

        try:
            requests.get(f"{uri.rstrip('/')}/health", timeout=3).raise_for_status()
        except Exception as exc:  # pragma: no cover - mensagem amigável
            raise SystemExit(
                f"Tracking server MLflow inacessível em {uri} ({exc}).\n"
                "Inicie com:  scripts/start_mlflow.ps1  (ou start_mlflow.sh)\n"
                "ou rode sem servidor:  $env:MLFLOW_TRACKING_URI='sqlite:///mlflow.db'"
            ) from None
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def data_source_tag(data_dir) -> str:
    """'synthetic' para dados de data/make_synthetic.py, 'real' para o resto (filtro na UI: tags.data_source)."""
    return "synthetic" if "synthetic" in Path(data_dir).parts else "real"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def model_name_kwarg(name: str, log_model_fn) -> dict:
    """MLflow 3 usa `name=`; MLflow 2 usa `artifact_path=`. Mantém compatibilidade com ambos."""
    params = inspect.signature(log_model_fn).parameters
    return {"name": name} if "name" in params else {"artifact_path": name}


def log_sklearn_model(model, name: str = "model", trusted_types: list[str] | None = None, **kwargs):
    """Loga modelo sklearn preferindo skops (formato seguro, padrão no MLflow 3).

    Estimadores de terceiros (LightGBM, UMAP/numba...) não são serializáveis/confiáveis no skops;
    nesse caso cai para cloudpickle — só carregue esses modelos de um tracking server confiável.
    """
    import mlflow.sklearn

    base = {**model_name_kwarg(name, mlflow.sklearn.log_model), **kwargs}
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    if "skops_trusted_types" in params:
        try:
            return mlflow.sklearn.log_model(sk_model=model, serialization_format="skops",
                                            skops_trusted_types=trusted_types or [], **base)
        except Exception as exc:
            print(f"[info] skops não serializou o modelo ({type(exc).__name__}); usando cloudpickle")
    return mlflow.sklearn.log_model(sk_model=model, serialization_format="cloudpickle", **base)


def log_fig(fig, artifact_file: str) -> None:
    """Loga uma figura matplotlib como artefato e fecha a figura."""
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)


def log_df(df: pd.DataFrame, artifact_file: str, index: bool = True) -> None:
    """Loga um DataFrame como CSV em um caminho de artefato (ex.: 'tables/perfil.csv')."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(artifact_file).name
        df.to_csv(path, index=index)
        parent = str(Path(artifact_file).parent)
        mlflow.log_artifact(str(path), artifact_path=None if parent == "." else parent)


def log_json(obj: dict, artifact_file: str) -> None:
    mlflow.log_dict(json.loads(json.dumps(obj, default=_json_default)), artifact_file)


def log_dataset(df: pd.DataFrame, name: str, context: str = "training") -> None:
    """Registra o dataset de entrada (schema + digest) na aba 'Datasets' do MLflow."""
    try:
        mlflow.log_input(mlflow.data.from_pandas(df, name=name), context=context)
    except Exception as exc:  # não bloqueia o pipeline por causa de metadado
        print(f"[aviso] log_input falhou: {exc}")


def find_file(directory: Path, patterns: list[str]) -> Path:
    """Procura recursivamente o primeiro arquivo que casa com algum padrão glob."""
    for pat in patterns:
        hits = sorted(directory.rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Nenhum arquivo {patterns} em {directory}. Rode `python data/download_data.py` "
        "ou `python data/make_synthetic.py` para testar com dados sintéticos."
    )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
