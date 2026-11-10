"""Gera dados sintéticos com o MESMO formato/schema dos datasets reais do Kaggle.

Serve para smoke test dos pipelines sem precisar de credenciais do Kaggle:
    python data/make_synthetic.py
    python -m p01_clustering.train --data-dir data/synthetic/ccdata --quick
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "synthetic"
rng = np.random.default_rng(0)


def ccdata(n: int = 2000) -> None:
    seg = rng.integers(0, 4, n)
    scale = np.array([500, 3000, 1500, 6000])[seg]
    freq = np.array([0.2, 0.9, 0.5, 0.1])[seg]
    df = pd.DataFrame({"CUST_ID": [f"C{10000 + i}" for i in range(n)]})
    df["BALANCE"] = rng.lognormal(np.log(scale), 1.0)
    df["BALANCE_FREQUENCY"] = np.clip(rng.normal(0.85, 0.15, n), 0, 1)
    df["PURCHASES"] = rng.lognormal(np.log(scale * freq + 1), 1.2) * (rng.random(n) < 0.8)
    df["ONEOFF_PURCHASES"] = df["PURCHASES"] * rng.beta(2, 2, n)
    df["INSTALLMENTS_PURCHASES"] = df["PURCHASES"] - df["ONEOFF_PURCHASES"]
    df["CASH_ADVANCE"] = rng.lognormal(np.log(scale * (1 - freq) + 1), 1.3) * (rng.random(n) < 0.5)
    df["PURCHASES_FREQUENCY"] = np.clip(freq + rng.normal(0, 0.15, n), 0, 1)
    df["ONEOFF_PURCHASES_FREQUENCY"] = np.clip(df["PURCHASES_FREQUENCY"] * rng.beta(2, 3, n), 0, 1)
    df["PURCHASES_INSTALLMENTS_FREQUENCY"] = np.clip(df["PURCHASES_FREQUENCY"] * rng.beta(3, 2, n), 0, 1)
    df["CASH_ADVANCE_FREQUENCY"] = np.clip((1 - freq) * rng.beta(1, 4, n), 0, 1)
    df["CASH_ADVANCE_TRX"] = rng.poisson(10 * df["CASH_ADVANCE_FREQUENCY"])
    df["PURCHASES_TRX"] = rng.poisson(40 * df["PURCHASES_FREQUENCY"])
    df["CREDIT_LIMIT"] = np.round(rng.lognormal(np.log(scale * 2 + 1000), 0.5), -2)
    df["PAYMENTS"] = rng.lognormal(np.log(scale * 0.6 + 50), 1.0)
    df["MINIMUM_PAYMENTS"] = rng.lognormal(np.log(scale * 0.2 + 20), 1.0)
    df["PRC_FULL_PAYMENT"] = np.clip(rng.beta(0.5, 2, n) * freq * 1.5, 0, 1)
    df["TENURE"] = rng.choice([6, 7, 8, 9, 10, 11, 12], n, p=[.02, .02, .03, .03, .05, .05, .8])
    df.loc[rng.random(n) < 0.035, "MINIMUM_PAYMENTS"] = np.nan
    df.loc[rng.integers(0, n), "CREDIT_LIMIT"] = np.nan
    (OUT / "ccdata").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ccdata" / "CC GENERAL.csv", index=False)


def telco(n: int = 3000) -> None:
    yn = np.array(["Yes", "No"])
    df = pd.DataFrame({"customerID": [f"{i:04d}-SYNTH" for i in range(n)]})
    df["gender"] = rng.choice(["Male", "Female"], n)
    df["SeniorCitizen"] = (rng.random(n) < 0.16).astype(int)
    df["Partner"] = rng.choice(yn, n)
    df["Dependents"] = rng.choice(yn, n, p=[.3, .7])
    df["tenure"] = rng.integers(0, 73, n)
    df["PhoneService"] = rng.choice(yn, n, p=[.9, .1])
    df["MultipleLines"] = np.where(df.PhoneService == "No", "No phone service", rng.choice(yn, n))
    df["InternetService"] = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22])
    for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[c] = np.where(df.InternetService == "No", "No internet service", rng.choice(yn, n))
    df["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
    df["PaperlessBilling"] = rng.choice(yn, n, p=[.6, .4])
    df["PaymentMethod"] = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n)
    df["MonthlyCharges"] = np.round(20 + 50 * (df.InternetService == "Fiber optic") + rng.normal(25, 15, n).clip(0), 2)
    df["TotalCharges"] = np.round(df.MonthlyCharges * df.tenure, 2).astype(str)
    df.loc[df.tenure == 0, "TotalCharges"] = " "
    logit = (-1.2 + 1.3 * (df.Contract == "Month-to-month") - 0.04 * df.tenure
             + 0.8 * (df.InternetService == "Fiber optic") + 0.5 * (df.PaymentMethod == "Electronic check")
             - 0.4 * (df.TechSupport == "Yes") + 0.3 * df.SeniorCitizen)
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    (OUT / "telco").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv", index=False)


def gene_expression(n: int = 300, p: int = 2000, k: int = 5) -> None:
    labels = rng.choice(["BRCA", "KIRC", "COAD", "LUAD", "PRAD"][:k], n)
    codes = pd.Categorical(labels).codes
    latent = rng.normal(0, 1, (k, 10))[codes] + rng.normal(0, 0.6, (n, 10))
    W = rng.normal(0, 1, (10, p)) * (rng.random(p) < 0.3)
    X = np.clip(latent @ W + rng.normal(6, 1.5, p) + rng.normal(0, 0.8, (n, p)), 0, None)
    X[:, rng.random(p) < 0.05] = 0.0  # genes nunca expressos (variância zero)
    ids = [f"sample_{i}" for i in range(n)]
    out = OUT / "gene_expression"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X.round(4), index=ids, columns=[f"gene_{j}" for j in range(p)]).to_csv(out / "data.csv")
    pd.DataFrame({"Class": labels}, index=ids).to_csv(out / "labels.csv")


def cmapss(subsets=("FD001", "FD003"), n_train: int = 60, n_test: int = 30) -> None:
    out = OUT / "cmapss"
    out.mkdir(parents=True, exist_ok=True)
    base = rng.uniform(1, 1600, 21)
    sens = rng.normal(0, 1, 21) * 8
    sens[[0, 4, 5, 9, 15, 17, 18]] = 0  # sensores constantes, como no FD001 real
    for s_i, sub in enumerate(subsets):
        def engine(unit: int, truncate: bool):
            T = int(rng.weibull(3.0) * 220 * (1 + 0.15 * s_i)) + 120
            stop = int(rng.uniform(0.3, 0.95) * T) if truncate else T
            t = np.arange(1, stop + 1)
            wear = (t / T) ** 2.5 * rng.uniform(0.8, 1.2)
            S = base + np.outer(wear, sens) + rng.normal(0, 0.4, (len(t), 21)) * (sens != 0)
            ops = np.column_stack([rng.normal(0, 0.002, len(t)), rng.normal(0, 0.0003, len(t)), np.full(len(t), 100.0)])
            return np.column_stack([np.full(len(t), unit), t, ops, S]), T - stop
        train = np.vstack([engine(u, False)[0] for u in range(1, n_train + 1)])
        tests = [engine(u, True) for u in range(1, n_test + 1)]
        fmt = ["%d", "%d"] + ["%.4f"] * 24
        np.savetxt(out / f"train_{sub}.txt", train, fmt=fmt, delimiter=" ")
        np.savetxt(out / f"test_{sub}.txt", np.vstack([a for a, _ in tests]), fmt=fmt, delimiter=" ")
        np.savetxt(out / f"RUL_{sub}.txt", np.array([r for _, r in tests]), fmt="%d")


if __name__ == "__main__":
    ccdata(); telco(); gene_expression(); cmapss()
    print("Dados sintéticos gerados em", OUT)
