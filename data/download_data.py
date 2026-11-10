"""Baixa os 4 datasets para data/raw/.

Pré-requisito (Kaggle): crie um token em https://www.kaggle.com/settings -> "Create New Token"
e salve em %USERPROFILE%\\.kaggle\\kaggle.json (Windows) ou ~/.kaggle/kaggle.json.

Uso:
    python data/download_data.py              # todos
    python data/download_data.py ccdata telco # apenas alguns

Observação sobre o projeto 3: o link `crawford/gene-expression` do Kaggle é o dataset de
leucemia de Golub (72 amostras x 7.129 genes). O dataset descrito no projeto (801 pacientes x
20.531 genes, 5 tipos de tumor: BRCA/KIRC/COAD/LUAD/PRAD) é o TCGA PANCAN, publicado na UCI.
Por padrão baixamos o da UCI (`gene_pancan`); `gene_golub` baixa o do Kaggle. O pipeline aceita ambos.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

KAGGLE = {
    "ccdata": ("arjunbhasin2013/ccdata", "ccdata"),
    "telco": ("blastchar/telco-customer-churn", "telco"),
    "gene_golub": ("crawford/gene-expression", "gene_golub"),
    "cmapss": ("behrad3d/nasa-cmaps", "cmapss"),
}
UCI_PANCAN = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"


def kaggle_download(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def uci_pancan(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "pancan.zip"
    print(f"> baixando {UCI_PANCAN}")
    urllib.request.urlretrieve(UCI_PANCAN, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    for tgz in dest.rglob("*.tar.gz"):
        with tarfile.open(tgz) as tf:
            tf.extractall(dest, filter="data")
        tgz.unlink()
    zip_path.unlink()


def main(names: list[str]) -> None:
    targets = names or ["ccdata", "telco", "gene_pancan", "cmapss"]
    for name in targets:
        if name == "gene_pancan":
            uci_pancan(RAW / "gene_expression")
        elif name in KAGGLE:
            slug, folder = KAGGLE[name]
            kaggle_download(slug, RAW / folder)
        else:
            raise SystemExit(f"Dataset desconhecido: {name}. Opções: {list(KAGGLE) + ['gene_pancan']}")
    print("Concluído. Arquivos em", RAW)


if __name__ == "__main__":
    main(sys.argv[1:])
