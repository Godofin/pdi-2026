"""Projeto 2 — Previsão de churn (Telco) com LightGBM + Optuna, calibração e threshold de negócio.

Pipeline:
  1. Limpeza (TotalCharges em branco) + features derivadas.
  2. TargetEncoder do sklearn DENTRO do Pipeline: ele faz cross-fitting interno no fit_transform,
     e como o Pipeline é re-ajustado em cada fold da CV, não há vazamento de alvo.
  3. Optuna (TPE) maximizando PR-AUC em StratifiedKFold — cada trial vira um run aninhado no MLflow.
  4. Calibração com CalibratedClassifierCV (sigmoide vs. isotônica), escolhida pelo Brier em OOF.
  5. Threshold que minimiza custo esperado (custo FN ≠ custo FP), escolhido em predições OOF do
     treino e só então aplicado no holdout de teste.
  6. SHAP TreeExplainer (global + local), modelo calibrado registrado + booster LightGBM nativo.

Execução:
  python -m p02_churn.train --n-trials 60
  python -m p02_churn.train --data-dir data/synthetic/telco --quick
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
import sklearn
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from common.mlflow_utils import (data_source_tag, DATA_RAW, find_file, log_dataset, log_df, log_fig, log_json,
                                 log_sklearn_model, seed_everything, setup_mlflow)

EXPERIMENT = "02_ensemble_telco_churn"
warnings.filterwarnings("ignore", message="X does not have valid feature names")
sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]


# --------------------------------------------------------------------------- dados
def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(find_file(data_dir, ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "*.csv"]))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True))
    # clientes com tenure=0 ainda não foram cobrados -> TotalCharges = 0 (não é dado faltante aleatório)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    y = (df.pop("Churn") == "Yes").astype(int)
    df = df.drop(columns=["customerID"], errors="ignore")
    return add_features(df), y


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"]).astype(str)
    df["n_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICES if c in df)
    df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["charge_delta"] = df["MonthlyCharges"] - df["avg_charge_per_month"]  # aumento recente de preço
    df["tenure_bucket"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, np.inf],
                                 labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(str)
    # interação de alta cardinalidade: exatamente onde o target encoding faz diferença
    df["contract_x_payment_x_internet"] = df["Contract"] + "|" + df["PaymentMethod"] + "|" + df["InternetService"]
    return df


def split_columns(X: pd.DataFrame):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return cat, num


def make_target_encoder(seed: int) -> TargetEncoder:
    """Cross-fitting interno de 5 folds. sklearn>=1.9 recebe o splitter em `cv` (random_state depreciado)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if "random_state" in TargetEncoder().get_params() and sklearn_version < (1, 9):
        return TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True, random_state=seed)
    return TargetEncoder(target_type="binary", smooth="auto", cv=cv)


def make_lgbm_pipeline(cat, num, params: dict, seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("te", make_target_encoder(seed), cat),
        ("num", "passthrough", num),
    ], verbose_feature_names_out=False)
    clf = lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=-1, verbose=-1, **params)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_logreg_pipeline(cat, num) -> Pipeline:
    pre = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
                             ("num", StandardScaler(), num)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


# --------------------------------------------------------------------------- métricas
def expected_calibration_error(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def prob_metrics(y, p, prefix="") -> dict:
    y = np.asarray(y)
    return {f"{prefix}pr_auc": average_precision_score(y, p), f"{prefix}roc_auc": roc_auc_score(y, p),
            f"{prefix}brier_score": brier_score_loss(y, p), f"{prefix}log_loss": log_loss(y, p),
            f"{prefix}ece": expected_calibration_error(y, p)}


def business_cost(y, p, thr, c_fn, c_fp) -> float:
    pred = p >= thr
    return float(((~pred) & (y == 1)).sum() * c_fn + (pred & (y == 0)).sum() * c_fp)


def optimal_threshold(y, p, c_fn, c_fp):
    grid = np.linspace(0.01, 0.99, 197)
    costs = np.array([business_cost(y, p, t, c_fn, c_fp) for t in grid])
    return float(grid[costs.argmin()]), grid, costs


# --------------------------------------------------------------------------- optuna
def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # pesar a classe minoritária distorce probabilidades; deixamos o Optuna decidir e a
        # calibração corrige depois
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
    }


def run_optuna(X, y, cat, num, args):
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    def objective(trial):
        params = suggest_params(trial)
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number:03d}", nested=True):
            mlflow.log_params(params)
            oof = cross_val_predict(make_lgbm_pipeline(cat, num, params, args.seed), X, y, cv=cv,
                                    method="predict_proba")[:, 1]
            m = prob_metrics(y, oof, prefix="cv_")
            mlflow.log_metrics(m)
        return m["cv_pr_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="lgbm_churn",
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    return study


# --------------------------------------------------------------------------- gráficos
def plot_pr_roc(y, curves: dict):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        a1.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        a2.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    a1.axhline(np.mean(y), ls=":", c="gray", label=f"prevalência={np.mean(y):.2f}")
    a1.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall (teste)")
    a2.plot([0, 1], [0, 1], ":", c="gray")
    a2.set(xlabel="FPR", ylabel="TPR", title="Curva ROC (teste)")
    a1.legend(fontsize=8); a2.legend(fontsize=8)
    return fig


def plot_reliability(y, curves: dict):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    a1.plot([0, 1], [0, 1], ":", c="gray", label="perfeitamente calibrado")
    for name, p in curves.items():
        fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        a1.plot(mp, fr, "o-", label=f"{name} (Brier={brier_score_loss(y, p):.4f})")
        a2.hist(p, bins=30, histtype="step", label=name)
    a1.set(ylabel="Fração de positivos", title="Diagrama de confiabilidade (teste)")
    a2.set(xlabel="Probabilidade prevista", ylabel="contagem")
    a1.legend(fontsize=8)
    return fig


def plot_cost_curve(grid, costs, thr):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, costs)
    ax.axvline(thr, c="red", ls="--", label=f"threshold ótimo={thr:.2f}")
    ax.axvline(0.5, c="gray", ls=":", label="0.5")
    ax.set(xlabel="threshold", ylabel="custo esperado (OOF treino)", title="Custo de negócio vs. threshold")
    ax.legend()
    return fig


# --------------------------------------------------------------------------- main
def main(args):
    seed_everything(args.seed)
    setup_mlflow(EXPERIMENT)
    X, y = load_data(Path(args.data_dir))
    cat, num = split_columns(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    with mlflow.start_run(run_name="lgbm_optuna_calibrated") as parent:
        mlflow.set_tags({"data_source": data_source_tag(args.data_dir), "project": "02_churn", "dataset": "telco_customer_churn", "algorithm": "LightGBM",
                         "stage": "experimentation"})
        mlflow.log_params({"n_trials": args.n_trials, "cv_folds": args.cv_folds, "test_size": args.test_size,
                           "cost_fn": args.cost_fn, "cost_fp": args.cost_fp, "encoder": "TargetEncoder(cv=5)",
                           "n_train": len(X_tr), "n_test": len(X_te), "churn_rate": round(float(y.mean()), 4),
                           "categorical_features": len(cat), "numeric_features": len(num)})
        log_dataset(X_tr.assign(Churn=y_tr.values), "telco_train")
        log_dataset(X_te.assign(Churn=y_te.values), "telco_test", context="test")

        # ---- baseline interpretável
        with mlflow.start_run(run_name="baseline_logreg", nested=True):
            base = make_logreg_pipeline(cat, num).fit(X_tr, y_tr)
            p_base = base.predict_proba(X_te)[:, 1]
            mlflow.log_metrics(prob_metrics(y_te, p_base, prefix="test_"))

        # ---- otimização bayesiana
        study = run_optuna(X_tr, y_tr, cat, num, args)
        best_params = {**study.best_params, "subsample_freq": 1}
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_pr_auc", study.best_value)
        log_df(study.trials_dataframe(), "optuna/trials.csv", index=False)
        try:
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
            log_fig(plot_optimization_history(study).figure, "optuna/optimization_history.png")
            if len(study.trials) >= 5:
                log_fig(plot_param_importances(study).figure, "optuna/param_importances.png")
        except Exception as exc:
            print(f"[aviso] gráficos do optuna: {exc}")

        # ---- modelo não calibrado
        raw_model = make_lgbm_pipeline(cat, num, best_params, args.seed).fit(X_tr, y_tr)
        p_raw = raw_model.predict_proba(X_te)[:, 1]

        # ---- calibração: escolhe método pelo Brier em OOF do treino (nada de olhar o teste)
        calibrated, oof_by_method = {}, {}
        for method in ["sigmoid", "isotonic"]:
            cal = CalibratedClassifierCV(make_lgbm_pipeline(cat, num, best_params, args.seed),
                                         method=method, cv=args.cv_folds, ensemble=True)
            oof_by_method[method] = cross_val_predict(cal, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
            calibrated[method] = cal.fit(X_tr, y_tr)
            mlflow.log_metric(f"oof_brier_{method}", brier_score_loss(y_tr, oof_by_method[method]))
        best_method = min(oof_by_method, key=lambda m: brier_score_loss(y_tr, oof_by_method[m]))
        model, oof = calibrated[best_method], oof_by_method[best_method]
        p_cal = model.predict_proba(X_te)[:, 1]
        mlflow.log_param("calibration_method", best_method)

        # ---- threshold de negócio (OOF treino) -> aplicado no teste
        thr, grid, costs = optimal_threshold(y_tr.values, oof, args.cost_fn, args.cost_fp)
        y_hat = (p_cal >= thr).astype(int)
        cost_model = business_cost(y_te.values, p_cal, thr, args.cost_fn, args.cost_fp)
        cost_none = business_cost(y_te.values, p_cal, 1.01, args.cost_fn, args.cost_fp)   # não age
        cost_all = business_cost(y_te.values, p_cal, -0.01, args.cost_fn, args.cost_fp)  # oferta p/ todos
        mlflow.log_metrics({
            **prob_metrics(y_te, p_raw, prefix="test_uncalibrated_"),
            **prob_metrics(y_te, p_cal, prefix="test_"),
            "test_f1_macro": f1_score(y_te, y_hat, average="macro"),
            "test_f1_macro_at_0.5": f1_score(y_te, (p_cal >= .5).astype(int), average="macro"),
            "business_threshold": thr,
            "test_business_cost": cost_model,
            "test_cost_no_action": cost_none,
            "test_cost_target_everyone": cost_all,
            "test_savings_vs_best_naive": min(cost_none, cost_all) - cost_model,
        })
        print(f"PR-AUC={average_precision_score(y_te, p_cal):.4f} ROC-AUC={roc_auc_score(y_te, p_cal):.4f} "
              f"Brier={brier_score_loss(y_te, p_cal):.4f} thr={thr:.2f} calib={best_method}")

        # ---- artefatos gráficos
        curves = {"LogReg baseline": p_base, "LightGBM": p_raw, f"LightGBM + {best_method}": p_cal}
        log_fig(plot_pr_roc(y_te, curves), "plots/pr_roc_curves.png")
        log_fig(plot_reliability(y_te, curves), "plots/reliability_diagram.png")
        log_fig(plot_cost_curve(grid, costs, thr), "plots/business_cost_curve.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_te, y_hat), display_labels=["Fica", "Churn"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"Matriz de confusão @ threshold={thr:.2f}")
        log_fig(fig, "plots/confusion_matrix_business_threshold.png")

        # ---- SHAP (no modelo não calibrado: a calibração é monotônica, o ranking de features é o mesmo)
        pre, booster = raw_model.named_steps["pre"], raw_model.named_steps["clf"]
        Xt_te = pd.DataFrame(pre.transform(X_te), columns=pre.get_feature_names_out(), index=X_te.index)
        explainer = shap.TreeExplainer(booster)
        sv = explainer(Xt_te)
        if sv.values.ndim == 3:  # algumas versões retornam (n, features, classes)
            sv = sv[:, :, 1]
        plt.figure(); shap.plots.beeswarm(sv, max_display=20, show=False)
        log_fig(plt.gcf(), "shap/summary_beeswarm.png")
        plt.figure(); shap.plots.bar(sv, max_display=20, show=False)
        log_fig(plt.gcf(), "shap/global_importance_bar.png")
        top = int(np.argmax(p_cal))
        plt.figure(); shap.plots.waterfall(sv[top], max_display=15, show=False)
        log_fig(plt.gcf(), "shap/local_waterfall_highest_risk.png")
        imp = pd.Series(np.abs(sv.values).mean(0), index=Xt_te.columns, name="mean_abs_shap").sort_values(ascending=False)
        log_df(imp.to_frame(), "shap/mean_abs_shap.csv")

        # ---- modelo: calibrado (sklearn, registrado) + booster nativo do LightGBM
        sig = infer_signature(X_te.head(50), model.predict_proba(X_te.head(50)))
        log_sklearn_model(model, "model", signature=sig, input_example=X_te.head(5),
                          pyfunc_predict_fn="predict_proba", registered_model_name="telco_churn_lgbm_calibrated")
        booster.booster_.save_model("lgbm_booster.txt")
        mlflow.log_artifact("lgbm_booster.txt", artifact_path="native_model")
        Path("lgbm_booster.txt").unlink()
        log_json({"best_params": best_params, "threshold": thr, "calibration": best_method,
                  "feature_order_after_preprocess": list(Xt_te.columns)}, "native_model/metadata.json")
        print(f"Run: {parent.info.run_id}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_RAW / "telco"))
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cost-fn", type=float, default=500.0, help="custo de perder um cliente que churnou (FN)")
    p.add_argument("--cost-fp", type=float, default=100.0, help="custo de oferta de retenção desnecessária (FP)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.n_trials, a.cv_folds = 5, 3
    return a


if __name__ == "__main__":
    main(parse_args())
