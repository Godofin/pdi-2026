"""Projeto 3 — Redução de dimensionalidade em RNA-Seq (p >> n): PCA + UMAP vs. Autoencoder esparso.

Pipeline (tudo ajustado só no split de treino; validação mede generalização):
  1. Normalização: log-CPM se forem contagens brutas; log2 com clipping se microarray (Golub);
     nada se já estiver em escala log (TCGA PANCAN da UCI).
  2. Remove genes quase constantes e mantém os `top_genes` mais variáveis; StandardScaler.
  3. PCA: scree plot, variância acumulada, nº de componentes para 85/90/95% e MSE de reconstrução.
  4. UMAP sobre as 50 PCs (varredura de n_neighbors) — trustworthiness, kNN-acc, silhueta por tumor.
  5. Autoencoder não-linear (PyTorch) com penalidade L1 no espaço latente, comparado ao PCA
     na MESMA dimensão latente (MSE de reconstrução na validação + trustworthiness).

Execução:
  python -m p03_dimred.train
  python -m p03_dimred.train --data-dir data/synthetic/gene_expression --quick
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.mlflow_utils import (data_source_tag, DATA_RAW, log_dataset, log_df, log_fig, log_json, log_sklearn_model, model_name_kwarg,
                                 seed_everything, setup_mlflow)

EXPERIMENT = "03_dimred_gene_expression"


# --------------------------------------------------------------------------- dados
def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Aceita o TCGA PANCAN (data.csv + labels.csv) ou o Golub do Kaggle (crawford/gene-expression)."""
    data = sorted(data_dir.rglob("data.csv"))
    if data:
        X = pd.read_csv(data[0], index_col=0)
        y = pd.read_csv(data[0].with_name("labels.csv"), index_col=0)["Class"].reindex(X.index)
        return X, y.astype(str), "tcga_pancan"
    golub = sorted(data_dir.rglob("data_set_ALL_AML_train.csv"))
    if golub:
        parts = [golub[0], *data_dir.rglob("data_set_ALL_AML_independent.csv")]
        frames = []
        for p in parts:
            df = pd.read_csv(p)
            df = df[[c for c in df.columns if c.isdigit()]].T  # remove colunas 'call' e descrições
            df.columns = pd.read_csv(p, usecols=["Gene Accession Number"])["Gene Accession Number"]
            frames.append(df)
        X = pd.concat(frames)
        X.index = X.index.astype(int)
        actual = pd.read_csv(next(data_dir.rglob("actual.csv"))).set_index("patient")["cancer"]
        X = X.sort_index()
        return X, actual.reindex(X.index).astype(str), "golub_leukemia"
    raise FileNotFoundError(f"Nenhum data.csv (PANCAN) ou data_set_ALL_AML_train.csv (Golub) em {data_dir}")


def normalize(X: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    vals = X.values
    if mode == "auto":
        if vals.min() < 0:
            mode = "microarray_log2"
        elif np.allclose(vals, np.round(vals)) and vals.max() > 1000:
            mode = "logcpm"
        else:
            mode = "none"
    if mode == "logcpm":
        lib = X.sum(axis=1).replace(0, 1)
        X = np.log2(X.div(lib, axis=0) * 1e6 + 1)
    elif mode == "microarray_log2":
        X = np.log2(X.clip(lower=20, upper=16000))
    return X, mode


def select_genes(X_tr: pd.DataFrame, var_min: float, top_n: int) -> list[str]:
    var = X_tr.var(axis=0)
    var = var[var > var_min]
    return var.sort_values(ascending=False).index[:top_n].tolist()


# --------------------------------------------------------------------------- avaliação de embeddings
def embedding_scores(X_high: np.ndarray, Z: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict:
    cv = StratifiedKFold(n_splits=min(5, np.bincount(pd.factorize(y)[0]).min()), shuffle=True, random_state=seed)
    knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, y, cv=cv).mean()
    return {"trustworthiness": trustworthiness(X_high, Z, n_neighbors=k),
            "knn_accuracy": knn,
            "silhouette_by_label": silhouette_score(Z, y)}


def scatter_png(Z, y, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for lab in np.unique(y):
        m = y == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=.75, label=lab)
    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    ax.legend(markerscale=2, fontsize=8)
    return fig


def log_interactive(Z, y, ids, split, title, path):
    try:
        import plotly.express as px
    except ImportError:
        return
    df = pd.DataFrame({"dim1": Z[:, 0], "dim2": Z[:, 1], "tumor": y, "sample": ids, "split": split})
    fig = px.scatter(df, x="dim1", y="dim2", color="tumor", symbol="split", hover_data=["sample"],
                     title=title, opacity=.8, width=900, height=650)
    mlflow.log_figure(fig, path)


# --------------------------------------------------------------------------- autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, latent: int, hidden=(512, 128), dropout: float = 0.1):
        super().__init__()
        enc, d = [], d_in
        for h in hidden:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*enc, nn.Linear(d, latent))
        dec, d = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]
            d = h
        self.decoder = nn.Sequential(*dec, nn.Linear(d, d_in))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        return self.encoder(torch.as_tensor(x, dtype=torch.float32)).numpy()

    @torch.no_grad()
    def reconstruction_mse(self, x: np.ndarray) -> float:
        self.eval()
        xt = torch.as_tensor(x, dtype=torch.float32)
        return float(nn.functional.mse_loss(self(xt)[0], xt))


def train_autoencoder(X_tr, X_va, latent, args):
    model = SparseAutoencoder(X_tr.shape[1], latent, hidden=tuple(args.ae_hidden), dropout=args.ae_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ae_lr, weight_decay=args.ae_weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32)),
                                     batch_size=args.ae_batch_size, shuffle=True, drop_last=len(X_tr) > args.ae_batch_size)
    xva = torch.as_tensor(X_va, dtype=torch.float32)
    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(args.ae_epochs):
        model.train()
        tot, tot_l1, n = 0.0, 0.0, 0
        for (xb,) in tr:
            xhat, z = model(xb)
            mse = nn.functional.mse_loss(xhat, xb)
            l1 = z.abs().mean()
            loss = mse + args.ae_l1 * l1
            opt.zero_grad(); loss.backward(); opt.step()
            tot += mse.item() * len(xb); tot_l1 += l1.item() * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(xva)[0], xva))
        sched.step(val)
        history.append((tot / n, val))
        mlflow.log_metrics({"train_reconstruction_mse": tot / n, "val_reconstruction_mse": val,
                            "latent_l1": tot_l1 / n, "lr": opt.param_groups[0]["lr"]}, step=epoch)
        if val < best - 1e-5:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.ae_patience:
                break
    model.load_state_dict(best_state)
    return model, np.array(history)


def log_encoder(ae: SparseAutoencoder, x_example: np.ndarray, registered_name: str) -> None:
    """Registra o ENCODER (genes -> latente), que é o que se serve para gerar embeddings.

    `ae.encoder` é um nn.Sequential só com camadas nativas do torch, então o modelo carrega sem
    depender deste script. Formato pickle: o padrão 'pt2' (torch.export) fixaria o tamanho do batch.
    O state_dict do AE completo (com decoder) vai como artefato para reconstruções.
    """
    import inspect
    import tempfile

    from mlflow.models import infer_signature

    enc = ae.encoder.eval()
    sig = infer_signature(x_example, ae.encode(x_example))
    kw = model_name_kwarg("encoder", mlflow.pytorch.log_model)
    if "serialization_format" in inspect.signature(mlflow.pytorch.log_model).parameters:
        kw["serialization_format"] = "pickle"
    mlflow.pytorch.log_model(enc, signature=sig, input_example=x_example, registered_model_name=registered_name, **kw)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "autoencoder_state_dict.pt"
        torch.save({"state_dict": ae.state_dict(), "d_in": x_example.shape[1],
                    "latent": int(ae.encode(x_example).shape[1])}, path)
        mlflow.log_artifact(str(path), artifact_path="full_autoencoder")


def pca_reconstruction_mse(pca_full: PCA, X: np.ndarray, d: int) -> float:
    Z = (X - pca_full.mean_) @ pca_full.components_[:d].T
    return float(np.mean((Z @ pca_full.components_[:d] + pca_full.mean_ - X) ** 2))


# --------------------------------------------------------------------------- main
def main(args):
    seed_everything(args.seed)
    setup_mlflow(EXPERIMENT)
    X_raw, y, source = load_data(Path(args.data_dir))
    X_norm, norm_mode = normalize(X_raw.astype(float), args.normalization)
    idx_tr, idx_va = train_test_split(X_norm.index, test_size=args.val_size, stratify=y, random_state=args.seed)
    genes = select_genes(X_norm.loc[idx_tr], args.var_min, args.top_genes)

    scaler = StandardScaler().fit(X_norm.loc[idx_tr, genes])
    S_tr = scaler.transform(X_norm.loc[idx_tr, genes])
    S_va = scaler.transform(X_norm.loc[idx_va, genes])
    S_all = scaler.transform(X_norm[genes])
    y_all = y.values
    split = np.where(X_norm.index.isin(idx_tr), "train", "val")
    ids = X_norm.index.astype(str).values

    with mlflow.start_run(run_name="pca_umap_vs_autoencoder") as parent:
        mlflow.set_tags({"data_source": data_source_tag(args.data_dir), "project": "03_dimred", "dataset": source, "stage": "experimentation"})
        mlflow.log_params({"n_samples": X_raw.shape[0], "n_genes_raw": X_raw.shape[1],
                           "n_genes_selected": len(genes), "normalization": norm_mode,
                           "var_min": args.var_min, "top_genes": args.top_genes, "val_size": args.val_size,
                           "n_classes": y.nunique(), "p_over_n": round(X_raw.shape[1] / X_raw.shape[0], 1)})
        mlflow.log_metric("genes_removed_low_variance", int((X_norm.loc[idx_tr].var() <= args.var_min).sum()))
        log_df(pd.Series(genes, name="gene").to_frame(), "tables/selected_genes.csv", index=False)
        log_df(y.value_counts().to_frame("n"), "tables/class_counts.csv")
        log_dataset(X_norm[genes[:200]].assign(label=y.values), f"{source}_top200_genes")  # amostra do schema

        # ------------------------------------------------ PCA
        pca_full = PCA(svd_solver="full", random_state=args.seed).fit(S_tr)
        evr = pca_full.explained_variance_ratio_
        cum = np.cumsum(evr)
        n_for = {f"pca_n_components_{int(t * 100)}pct": int(np.searchsorted(cum, t) + 1) for t in (0.85, 0.90, 0.95)}
        mlflow.log_metrics(n_for)
        for i, v in enumerate(evr[:50]):
            mlflow.log_metric("pca_explained_variance_ratio", float(v), step=i + 1)
            mlflow.log_metric("pca_cumulative_variance", float(cum[i]), step=i + 1)
        for d in sorted({2, 10, 50, *args.latent_dims}):
            if d <= len(evr):
                mlflow.log_metric(f"pca{d}_val_reconstruction_mse", pca_reconstruction_mse(pca_full, S_va, d))

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        m = min(60, len(evr))
        ax1.bar(range(1, m + 1), evr[:m], color="steelblue", label="variância explicada")
        ax2 = ax1.twinx()
        ax2.plot(range(1, m + 1), cum[:m], "r.-", label="acumulada")
        for t in (0.85, 0.90):
            ax2.axhline(t, ls=":", c="gray")
        k85 = n_for["pca_n_components_85pct"]
        if k85 <= m:
            ax2.axvline(k85, ls="--", c="red", alpha=.5)
        ax1.set(xlabel="componente", ylabel="razão de variância", title=f"Scree plot — 85% da variância em {k85} PCs")
        ax2.set_ylabel("acumulada"); ax2.set_ylim(0, 1.02)
        log_fig(fig, "plots/pca_scree.png")

        n_pcs = min(args.n_pcs, S_tr.shape[0] - 1, len(genes))
        pca50 = PCA(n_components=n_pcs, random_state=args.seed).fit(S_tr)
        P_all = pca50.transform(S_all)
        scores = embedding_scores(S_all, P_all[:, :2], y_all, args.trust_k, args.seed)
        mlflow.log_metrics({f"pca2_{k}": v for k, v in scores.items()})
        log_fig(scatter_png(P_all[:, :2], y_all, "PCA (PC1 x PC2)"), "plots/pca_2d.png")
        log_interactive(P_all[:, :2], y_all, ids, split, "PCA 2D", "interactive/pca_2d.html")
        summary = {"PCA-2": scores}

        # ------------------------------------------------ UMAP sobre as PCs
        import umap
        best_umap, best_tw = None, -1
        for nn_ in args.umap_neighbors:
            with mlflow.start_run(run_name=f"umap_nn{nn_}", nested=True):
                params = {"n_neighbors": nn_, "min_dist": args.umap_min_dist, "n_components": 2,
                          "metric": "euclidean", "input": f"PCA-{n_pcs}", "random_state": args.seed}
                mlflow.log_params(params)
                umap_kwargs = {k: v for k, v in params.items() if k != "input"}
                reducer = umap.UMAP(**umap_kwargs).fit(P_all[split == "train"])
                U_all = reducer.transform(P_all)
                sc = embedding_scores(S_all, U_all, y_all, args.trust_k, args.seed)
                mlflow.log_metrics(sc)
                log_fig(scatter_png(U_all, y_all, f"UMAP (n_neighbors={nn_}) sobre PCA-{n_pcs}"), "plots/umap_2d.png")
                print(f"UMAP nn={nn_}: {sc}")
                if sc["trustworthiness"] > best_tw:
                    best_tw, best_umap, best_U, best_sc = sc["trustworthiness"], reducer, U_all, sc
        mlflow.log_param("umap_best_n_neighbors", best_umap.n_neighbors)
        mlflow.log_metrics({f"umap_{k}": v for k, v in best_sc.items()})
        log_fig(scatter_png(best_U, y_all, f"UMAP (melhor n_neighbors={best_umap.n_neighbors})"), "plots/umap_2d_best.png")
        log_interactive(best_U, y_all, ids, split, "UMAP 2D sobre PCA (cor = tipo de tumor)", "interactive/umap_2d.html")
        summary["UMAP-2"] = best_sc

        pipe = Pipeline([("scaler", scaler), ("pca", pca50), ("umap", best_umap)])
        log_sklearn_model(pipe, "pca_umap_model", registered_model_name="gene_expression_pca_umap")

        # ------------------------------------------------ Autoencoder esparso
        for d in args.latent_dims:
            with mlflow.start_run(run_name=f"sparse_autoencoder_latent{d}", nested=True):
                mlflow.log_params({"latent_dim": d, "hidden": str(args.ae_hidden), "l1_latent": args.ae_l1,
                                   "lr": args.ae_lr, "weight_decay": args.ae_weight_decay, "dropout": args.ae_dropout,
                                   "batch_size": args.ae_batch_size, "max_epochs": args.ae_epochs,
                                   "patience": args.ae_patience})
                torch.manual_seed(args.seed)
                ae, hist = train_autoencoder(S_tr, S_va, d, args)
                Z_all = ae.encode(S_all)
                ae_mse, pca_mse = ae.reconstruction_mse(S_va), pca_reconstruction_mse(pca_full, S_va, d)
                sc = embedding_scores(S_all, Z_all, y_all, args.trust_k, args.seed)
                pca_sc = embedding_scores(S_all, P_all[:, :d], y_all, args.trust_k, args.seed) if d <= n_pcs else {}
                mlflow.log_metrics({"val_reconstruction_mse_best": ae_mse, f"pca{d}_val_reconstruction_mse": pca_mse,
                                    "mse_gain_vs_pca": pca_mse - ae_mse, "epochs_trained": len(hist),
                                    "latent_active_units": int((np.abs(Z_all).mean(0) > 1e-2).sum()),
                                    **sc, **{f"pca{d}_{k}": v for k, v in pca_sc.items()}})
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(hist[:, 0], label="treino"); ax.plot(hist[:, 1], label="validação")
                ax.axhline(pca_mse, ls="--", c="gray", label=f"PCA-{d} (val)")
                ax.set(xlabel="época", ylabel="MSE de reconstrução", yscale="log", title=f"Autoencoder latent={d}")
                ax.legend()
                log_fig(fig, "plots/ae_loss_curves.png")
                if d == 2:
                    log_fig(scatter_png(Z_all, y_all, "Autoencoder esparso — espaço latente 2D"), "plots/ae_latent_2d.png")
                    log_interactive(Z_all, y_all, ids, split, "Autoencoder latente 2D", "interactive/ae_latent_2d.html")
                log_encoder(ae, S_va[:5].astype(np.float32), f"gene_expression_sparse_ae_latent{d}")
                print(f"AE latent={d}: val MSE={ae_mse:.4f} (PCA-{d}={pca_mse:.4f}) {sc}")
                summary[f"AE-{d}"] = {"val_mse": ae_mse, "pca_val_mse": pca_mse, **sc}
                if d <= n_pcs:
                    summary[f"PCA-{d}"] = {"val_mse": pca_mse, **pca_sc}

        comp = pd.DataFrame(summary).T
        log_df(comp.round(4), "tables/method_comparison.csv")
        log_json(summary, "summary.json")
        print(comp.round(4))
        print(f"Run: {parent.info.run_id}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_RAW / "gene_expression"))
    p.add_argument("--normalization", choices=["auto", "logcpm", "microarray_log2", "none"], default="auto")
    p.add_argument("--var-min", type=float, default=1e-2, help="variância mínima (escala log) para manter o gene")
    p.add_argument("--top-genes", type=int, default=5000)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--umap-neighbors", type=int, nargs="+", default=[5, 15, 50])
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--trust-k", type=int, default=10)
    p.add_argument("--latent-dims", type=int, nargs="+", default=[2, 10])
    p.add_argument("--ae-hidden", type=int, nargs="+", default=[512, 128])
    p.add_argument("--ae-l1", type=float, default=1e-3)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--ae-weight-decay", type=float, default=1e-4)
    p.add_argument("--ae-dropout", type=float, default=0.1)
    p.add_argument("--ae-batch-size", type=int, default=64)
    p.add_argument("--ae-epochs", type=int, default=400)
    p.add_argument("--ae-patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.umap_neighbors, a.ae_epochs, a.ae_patience, a.top_genes = [15], 40, 10, min(a.top_genes, 1500)
    return a


if __name__ == "__main__":
    main(parse_args())
