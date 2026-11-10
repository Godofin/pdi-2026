"""Projeto 1 — Segmentação de clientes de cartão de crédito (K-Means vs. GMM).

Pipeline:
  1. Imputação por mediana (MINIMUM_PAYMENTS, CREDIT_LIMIT) + Yeo-Johnson para caudas longas.
  2. RobustScaler.
  3. Varredura k ∈ [k_min, k_max]: K-Means (k-means++) e GMM (vários tipos de covariância, BIC/AIC).
  4. Seleção do melhor modelo, estabilidade por bootstrap de seeds (ARI), perfil dos clusters.

Execução:
  python -m p01_clustering.train                     # dataset real em data/raw/ccdata
  python -m p01_clustering.train --data-dir data/synthetic/ccdata --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib import cm
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler

from common.mlflow_utils import (data_source_tag, DATA_RAW, find_file, log_dataset, log_df, log_fig, log_json,
                                 log_sklearn_model, seed_everything, setup_mlflow)

EXPERIMENT = "01_clustering_credit_card"


# --------------------------------------------------------------------------- dados
def load_data(data_dir: Path) -> pd.DataFrame:
    path = find_file(data_dir, ["CC GENERAL.csv", "*.csv"])
    df = pd.read_csv(path)
    return df.drop(columns=["CUST_ID"], errors="ignore")


def skewed_columns(cols) -> list[str]:
    """Variáveis monetárias e de contagem (caudas longas). Frequências já estão em [0,1] e TENURE é
    quase constante (~85% = 12): Yeo-Johnson nelas gera lambdas extremos e valores explosivos."""
    return [c for c in cols if not c.endswith("FREQUENCY") and c not in {"TENURE", "PRC_FULL_PAYMENT"}]


def make_preprocessor(columns) -> Pipeline:
    skewed = skewed_columns(columns)
    other = [c for c in columns if c not in skewed]
    ct = ColumnTransformer([
        ("skewed", Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False))]), skewed),
        ("bounded", SimpleImputer(strategy="median"), other),
    ], verbose_feature_names_out=False)
    ct.set_output(transform="default")
    # quantile_range=(5,95): com o IQR padrão (25,75) colunas concentradas (TENURE, PRC_FULL_PAYMENT)
    # teriam IQR=0 e ficariam sem escala, dominando as distâncias.
    return Pipeline([("columns", ct), ("scale", RobustScaler(quantile_range=(5.0, 95.0)))])


# --------------------------------------------------------------------------- métricas
def cluster_metrics(X: np.ndarray, labels: np.ndarray, sample_size: int | None, seed: int) -> dict:
    if len(np.unique(labels)) < 2:
        return {"silhouette_score": -1.0, "davies_bouldin_index": np.inf, "calinski_harabasz": 0.0}
    return {
        "silhouette_score": silhouette_score(X, labels, sample_size=sample_size, random_state=seed),
        "davies_bouldin_index": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def fit_model(algo: str, k: int, X: np.ndarray, seed: int, cov: str = "full"):
    if algo == "kmeans":
        m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(X)
        return m, m.labels_
    m = GaussianMixture(n_components=k, covariance_type=cov, n_init=3, reg_covar=1e-4,
                        random_state=seed).fit(X)
    return m, m.predict(X)


# --------------------------------------------------------------------------- gráficos
def plot_sweep(results: pd.DataFrame):
    km = results[results.algorithm == "kmeans"].sort_values("k")
    figs = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(km.k, km.inertia, "o-")
    ax.set(xlabel="k", ylabel="Inércia (WCSS)", title="Método do cotovelo — K-Means")
    figs["plots/elbow_kmeans.png"] = fig

    fig, ax = plt.subplots(figsize=(6, 4))
    for (algo, cov), g in results.groupby(["algorithm", "covariance_type"]):
        if algo == "gmm":
            g = g.sort_values("k")
            ax.plot(g.k, g.bic, "o-", label=f"BIC ({cov})")
            ax.plot(g.k, g.aic, "x--", alpha=.6, label=f"AIC ({cov})")
    ax.set(xlabel="k", ylabel="Critério de informação", title="GMM — BIC/AIC (menor é melhor)")
    ax.legend(fontsize=7)
    figs["plots/gmm_bic_aic.png"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, better in zip(axes, ["silhouette_score", "davies_bouldin_index", "calinski_harabasz"],
                                  ["↑", "↓", "↑"]):
        for key, g in results.groupby("model_id_family"):
            g = g.sort_values("k")
            ax.plot(g.k, g[metric], "o-", label=key, ms=4)
        ax.set(xlabel="k", title=f"{metric} {better}")
    axes[0].legend(fontsize=7)
    figs["plots/metrics_vs_k.png"] = fig
    return figs


def plot_silhouette(X, labels, seed, max_points=5000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_points, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    s = silhouette_samples(Xs, ls)
    fig, ax = plt.subplots(figsize=(6, 5))
    y = 10
    ks = np.unique(ls)
    for i, c in enumerate(ks):
        vals = np.sort(s[ls == c])
        ax.fill_betweenx(np.arange(y, y + len(vals)), 0, vals, color=cm.tab10(i % 10), alpha=.8)
        ax.text(-0.05, y + len(vals) / 2, str(c))
        y += len(vals) + 10
    ax.axvline(s.mean(), color="red", ls="--", label=f"média={s.mean():.3f}")
    ax.set(xlabel="Coeficiente de silhueta", ylabel="Cluster", yticks=[], title="Silhouette plot por cluster")
    ax.legend()
    per_cluster = pd.DataFrame({"cluster": ls, "silhouette": s}).groupby("cluster").silhouette.agg(["mean", "min", "count"])
    return fig, per_cluster


def plot_projection(X, labels, centers, seed):
    pca = PCA(n_components=2, random_state=seed).fit(X)
    Z, C = pca.transform(X), pca.transform(centers)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=4, alpha=.4)
    ax.scatter(C[:, 0], C[:, 1], c="black", marker="X", s=200, edgecolor="white", label="centróides")
    for i, (x, y) in enumerate(C):
        ax.annotate(str(i), (x, y), xytext=(6, 6), textcoords="offset points", weight="bold")
    ev = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ev[0]:.1%})", ylabel=f"PC2 ({ev[1]:.1%})", title="Projeção 2D (PCA) com centróides")
    ax.legend(*sc.legend_elements(), title="cluster", fontsize=7, loc="best")
    return fig


def plot_profile(profile_z: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(profile_z) + 2))
    im = ax.imshow(profile_z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(profile_z.shape[1]), profile_z.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(profile_z)), [f"cluster {i}" for i in profile_z.index])
    fig.colorbar(im, ax=ax, label="z-score da mediana (escala transformada)")
    ax.set_title("Perfil dos segmentos")
    return fig


# --------------------------------------------------------------------------- main
def main(args):
    seed_everything(args.seed)
    setup_mlflow(EXPERIMENT)
    raw = load_data(Path(args.data_dir))
    pre = make_preprocessor(raw.columns)
    X = pre.fit_transform(raw)
    feat_names = list(pre.named_steps["columns"].get_feature_names_out())
    sample = None if len(X) <= args.silhouette_sample else args.silhouette_sample

    ks = range(args.k_min, args.k_max + 1)
    covs = ["full"] if args.quick else ["full", "diag", "tied"]

    with mlflow.start_run(run_name="sweep_kmeans_vs_gmm") as parent:
        mlflow.set_tags({"data_source": data_source_tag(args.data_dir), "project": "01_clustering", "dataset": "ccdata_kaggle", "stage": "experimentation"})
        mlflow.log_params({"k_min": args.k_min, "k_max": args.k_max, "imputer": "median",
                           "transform": "yeo-johnson (monetárias/contagens)", "scaler": "RobustScaler(5-95)",
                           "yeojohnson_columns": ",".join(skewed_columns(raw.columns)), "seed": args.seed,
                           "gmm_covariance_types": ",".join(covs), "n_rows": len(raw), "n_features": raw.shape[1]})
        log_dataset(raw, "ccdata")
        mlflow.log_metric("missing_minimum_payments", int(raw.get("MINIMUM_PAYMENTS", pd.Series()).isna().sum()))

        rows = []
        for algo, cov in [("kmeans", "-")] + [("gmm", c) for c in covs]:
            for k in ks:
                with mlflow.start_run(run_name=f"{algo}{'' if cov == '-' else '_' + cov}_k{k}", nested=True):
                    params = {"algorithm": algo, "n_clusters": k, "covariance_type": cov, "random_state": args.seed}
                    params.update({"init": "k-means++", "n_init": 10} if algo == "kmeans" else {"n_init": 3})
                    mlflow.log_params(params)
                    model, labels = fit_model(algo, k, X, args.seed, cov)
                    metrics = cluster_metrics(X, labels, sample, args.seed)
                    if algo == "kmeans":
                        metrics["inertia"] = model.inertia_
                    else:
                        metrics.update(bic=model.bic(X), aic=model.aic(X), converged=float(model.converged_))
                    metrics["min_cluster_frac"] = np.bincount(labels, minlength=k).min() / len(labels)
                    mlflow.log_metrics(metrics)
                    rows.append({"algorithm": algo, "covariance_type": cov, "k": k,
                                 "model_id_family": algo if cov == "-" else f"gmm_{cov}", **metrics})
                    print(f"{algo:6s} {cov:5s} k={k:2d} sil={metrics['silhouette_score']:.3f} "
                          f"DB={metrics['davies_bouldin_index']:.3f}")

        results = pd.DataFrame(rows)
        for c in ["inertia", "bic", "aic"]:
            if c not in results:
                results[c] = np.nan
        log_df(results, "tables/sweep_results.csv", index=False)
        for path, fig in plot_sweep(results).items():
            log_fig(fig, path)

        # Seleção: maior silhueta entre soluções sem clusters minúsculos (<min_cluster_frac)
        valid = results[results.min_cluster_frac >= args.min_cluster_frac]
        best = (valid if len(valid) else results).sort_values("silhouette_score", ascending=False).iloc[0]
        best_algo, best_k, best_cov = best.algorithm, int(best.k), best.covariance_type
        print(f"\n>> melhor: {best_algo} {best_cov} k={best_k} (silhouette={best.silhouette_score:.3f})")

        model, labels = fit_model(best_algo, best_k, X, args.seed, best_cov if best_cov != "-" else "full")
        centers = model.cluster_centers_ if best_algo == "kmeans" else model.means_

        # Estabilidade: ARI entre a solução de referência e re-treinos com outras seeds
        aris = [adjusted_rand_score(labels, fit_model(best_algo, best_k, X, s, best_cov if best_cov != "-" else "full")[1])
                for s in range(args.seed + 1, args.seed + 1 + args.stability_runs)]

        mlflow.log_params({"best_algorithm": best_algo, "best_k": best_k, "best_covariance_type": best_cov})
        mlflow.log_metrics({f"best_{m}": float(best[m]) for m in
                            ["silhouette_score", "davies_bouldin_index", "calinski_harabasz"]})
        mlflow.log_metrics({"stability_ari_mean": float(np.mean(aris)), "stability_ari_min": float(np.min(aris))})

        fig, per_cluster = plot_silhouette(X, labels, args.seed)
        log_fig(fig, "plots/silhouette_best.png")
        log_df(per_cluster, "tables/silhouette_per_cluster.csv")
        log_fig(plot_projection(X, labels, centers, args.seed), "plots/projection_2d_centroids.png")

        # Perfil dos clusters: medianas na escala original + heatmap em z-score
        prof = raw.assign(cluster=labels).groupby("cluster").median()
        prof.insert(0, "size", np.bincount(labels))
        prof.insert(1, "share", prof["size"] / len(labels))
        log_df(prof.round(3), "tables/cluster_profile_median_original_scale.csv")
        Xz = pd.DataFrame(X, columns=feat_names).assign(cluster=labels).groupby("cluster").median()
        log_fig(plot_profile((Xz - Xz.mean()) / Xz.std(ddof=0).replace(0, 1)), "plots/cluster_profile_heatmap.png")

        # Modelo final = pré-processamento + clusterizador, servível com .predict(df_bruto)
        final = Pipeline([("preprocess", pre), ("cluster", model)])
        sig = infer_signature(raw.head(50), final.predict(raw.head(50)))
        log_sklearn_model(final, "model", trusted_types=["numpy.dtype"], signature=sig,
                          input_example=raw.head(5), registered_model_name="credit_card_segmentation")
        log_json({"best": best.to_dict(), "stability_ari": aris}, "summary.json")
        print(f"Run: {parent.info.run_id}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_RAW / "ccdata"))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--min-cluster-frac", type=float, default=0.03, help="descarta soluções com cluster < 3%%")
    p.add_argument("--silhouette-sample", type=int, default=10000)
    p.add_argument("--stability-runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="smoke test: k<=5, só GMM full")
    a = p.parse_args()
    if a.quick:
        a.k_max, a.stability_runs = min(a.k_max, 5), 2
    return a


if __name__ == "__main__":
    main(parse_args())
