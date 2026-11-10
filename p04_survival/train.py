"""Projeto 4 — Confiabilidade de motores turbofan (NASA C-MAPSS): tempo até a falha / RUL.

Formulação (landmarking): em cada "snapshot" de um motor no ciclo t, as covariáveis são features
de janela móvel dos sensores até t (histórico de degradação) + o próprio t; o alvo é a vida
residual (tempo de t até a falha), com censura à direita.

  * Treino: motores de train_FDxxx rodam até a falha. Para exercitar censura de verdade,
    `--censor-frac` dos motores tem a trajetória truncada num ponto aleatório (evento = 0).
  * Validação: 20% dos motores de treino (por motor, sem vazamento) — seleção de hiperparâmetros.
  * Teste: test_FDxxx truncado + RUL_FDxxx → snapshot no último ciclo observado, evento conhecido.

Modelos:
  1. Kaplan-Meier (por frota/subset) — referência não paramétrica.
  2. Cox PH com penalização ElasticNet (lifelines), erro-padrão robusto por motor (cluster_col),
     grid de penalizer × l1_ratio + teste/plots de resíduos de Schoenfeld.
  3. Cox com covariáveis tempo-variantes (CoxTimeVaryingFitter, formato start/stop) — usado para
     inferência e ranking de risco; não prevê RUL (covariáveis internas exigiriam um joint model).
  4. Weibull AFT (lifelines).
  5. Weibull AFT Bayesiano hierárquico (PyMC): intercepto por frota com pooling parcial,
     censura via pm.Censored, HDI dos β, R-hat/ESS, WAIC e LOO-CV (PSIS).

Métricas: C-index, Brier dinâmico (IPCW) e Integrated Brier Score, RMSE/MAE do RUL e score NASA.

Execução:
  python -m p04_survival.train --subsets FD001 FD003
  python -m p04_survival.train --data-dir data/synthetic/cmapss --quick
"""
from __future__ import annotations

import argparse
import os
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

from common.mlflow_utils import (data_source_tag, DATA_RAW, find_file, log_dataset, log_df, log_fig, log_json,
                                 model_name_kwarg, seed_everything, setup_mlflow)

EXPERIMENT = "04_survival_turbofan_rul"
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
warnings.filterwarnings("ignore", category=FutureWarning)
if shutil.which("g++") is None:  # sem compilador C: PyTensor avisaria a cada import; o NUTS roda via nutpie/numba
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


# =========================================================================== dados
def read_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    return df


def load_subsets(data_dir: Path, subsets: list[str]):
    trains, tests = [], []
    for sub in subsets:
        tr = read_cmapss(find_file(data_dir, [f"train_{sub}.txt"]))
        te = read_cmapss(find_file(data_dir, [f"test_{sub}.txt"]))
        rul = pd.read_csv(find_file(data_dir, [f"RUL_{sub}.txt"]), sep=r"\s+", header=None).iloc[:, 0].values
        for df in (tr, te):
            df.insert(0, "fleet", sub)
            df.insert(0, "unit_id", sub + "_" + df["unit"].astype(str))
        te_last = te.groupby("unit_id", sort=False).cycle.max()
        units = te.drop_duplicates("unit_id")[["unit_id", "unit"]].set_index("unit_id")["unit"]
        te_rul = pd.Series(rul[units.values - 1], index=units.index, name="rul_true")
        trains.append(tr)
        tests.append((te, te_last, te_rul))
    train = pd.concat(trains, ignore_index=True)
    test = pd.concat([t[0] for t in tests], ignore_index=True)
    test_rul = pd.concat([t[2] for t in tests])
    return train, test, test_rul


def regime_key(df: pd.DataFrame) -> pd.Series:
    """Condição operacional ≈ combinação arredondada dos 3 settings (6 regimes em FD002/FD004).

    O `+ 0.0` normaliza -0.0 para 0.0: sem isso, ruído como -0.0003 vira o texto "-0.0" e o FD001
    (uma única condição) seria dividido em regimes espúrios.
    """
    r = lambda s, d: (s.round(d) + 0.0).astype(str)  # noqa: E731
    return r(df.op1, 0) + "_" + r(df.op2, 2) + "_" + r(df.op3, 0)


class RegimeNormalizer:
    """z-score dos sensores dentro de cada regime operacional (estatísticas do treino)."""

    def fit(self, df: pd.DataFrame, sensors: list[str]):
        r = regime_key(df)
        self.sensors = sensors
        self.stats = df.groupby(r)[sensors].agg(["mean", "std"])
        self.global_ = df[sensors].agg(["mean", "std"])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        r = regime_key(df)
        for s in self.sensors:
            mu = r.map(self.stats[(s, "mean")]).fillna(self.global_.loc["mean", s])
            sd = r.map(self.stats[(s, "std")]).fillna(self.global_.loc["std", s]).replace(0, 1).fillna(1)
            df[s] = (df[s] - mu) / sd
        return df


def informative_sensors(df: pd.DataFrame, tol: float = 1e-4) -> list[str]:
    sens = [f"s{i}" for i in range(1, 22)]
    std_by_regime = df.groupby(regime_key(df))[sens].std().max()
    return [s for s in sens if std_by_regime[s] > tol]


def rolling_features(df: pd.DataFrame, sensors: list[str], w: int) -> pd.DataFrame:
    """Média, desvio e inclinação (tendência linear) em janela móvel por motor."""
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit_id", sort=False)
    roll = lambda col: g[col].transform(lambda x: x.rolling(w, min_periods=1).mean())  # noqa: E731
    out = df[["unit_id", "fleet", "cycle"]].copy()
    t = df["cycle"].astype(float)
    df["_t2"] = t ** 2
    mt, mtt = roll("cycle"), roll("_t2")
    var_t = (mtt - mt ** 2).replace(0, np.nan)
    for s in sensors:
        ms = roll(s)
        df["_ts"] = t * df[s]
        out[f"{s}_mean"] = ms
        out[f"{s}_std"] = g[s].transform(lambda x: x.rolling(w, min_periods=2).std()).fillna(0)
        out[f"{s}_slope"] = ((roll("_ts") - mt * ms) / var_t).fillna(0)
    return out


def apply_censoring(train: pd.DataFrame, units: np.ndarray, frac: float, rng) -> pd.DataFrame:
    """Trunca `frac` dos motores num ciclo aleatório entre 40% e 95% da vida → censura à direita."""
    life = train.groupby("unit_id").cycle.max()
    cens = rng.choice(units, size=int(round(frac * len(units))), replace=False)
    end = life.copy()
    end[cens] = np.floor(life[cens] * rng.uniform(0.4, 0.95, len(cens))).astype(int)
    event = pd.Series(1, index=life.index)
    event[cens] = 0
    return end, event


def snapshots(feats: pd.DataFrame, end: pd.Series, event: pd.Series, stride: int, min_cycle: int) -> pd.DataFrame:
    df = feats[feats.unit_id.isin(end.index)].copy()
    df["end"] = df.unit_id.map(end)
    df = df[(df.cycle < df.end) & (df.cycle >= min_cycle) & (df.cycle % stride == 0)]
    df["duration"] = (df.end - df.cycle).astype(float)
    df["event"] = df.unit_id.map(event).astype(int)
    return df.drop(columns="end").reset_index(drop=True)


# =========================================================================== métricas
def brier_ipcw(T, E, surv: np.ndarray, times: np.ndarray):
    """Brier dinâmico com IPCW (Graf et al., 1999); distribuição de censura via KM no próprio conjunto."""
    T, E = np.asarray(T, float), np.asarray(E, int)
    if (E == 0).any():
        kmc = KaplanMeierFitter().fit(T, 1 - E)
        G = lambda x: np.clip(kmc.survival_function_at_times(x).values, 0.05, 1)  # noqa: E731
    else:
        G = lambda x: np.ones_like(np.atleast_1d(x), dtype=float)  # noqa: E731
    bs = []
    for j, t in enumerate(times):
        died = (T <= t) & (E == 1)
        alive = T > t
        term = np.zeros(len(T))
        term[died] = surv[died, j] ** 2 / G(T[died])
        term[alive] = (1 - surv[alive, j]) ** 2 / G(np.array([t]))[0]
        bs.append(term.mean())
    bs = np.array(bs)
    ibs = np.trapezoid(bs, times) / (times[-1] - times[0]) if len(times) > 1 else bs[0]
    return bs, float(ibs)


def nasa_score(pred, true) -> float:
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(name, T, E, risk, surv, times, rul_pred=None, prefix=""):
    m = {f"{prefix}c_index": concordance_index(T, -np.asarray(risk), E)}
    if surv is not None:
        bs, ibs = brier_ipcw(T, E, surv, times)
        m[f"{prefix}integrated_brier_score"] = ibs
        for h in (25, 50, 100):
            if times[0] <= h <= times[-1]:
                m[f"{prefix}brier_t{h}"] = float(np.interp(h, times, bs))
    if rul_pred is not None:
        ev = np.asarray(E) == 1
        err = np.asarray(rul_pred)[ev] - np.asarray(T)[ev]
        m[f"{prefix}rul_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        m[f"{prefix}rul_mae"] = float(np.mean(np.abs(err)))
        m[f"{prefix}nasa_score"] = nasa_score(np.asarray(rul_pred)[ev], np.asarray(T)[ev])
    print(f"  [{name}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return m


def finite_median(med, cap):
    med = np.asarray(med, float)
    return np.where(np.isfinite(med), med, cap)


# =========================================================================== modelo servível
class SurvivalRULModel(mlflow.pyfunc.PythonModel):
    """Recebe as features de snapshot (não padronizadas) e retorna RUL mediano + risco relativo."""

    def __init__(self, fitter, features, mean, std, cap):
        self.fitter, self.features, self.mean, self.std, self.cap = fitter, features, mean, std, cap

    def predict(self, context, model_input, params=None):
        X = (model_input[self.features] - self.mean) / self.std
        med = finite_median(self.fitter.predict_median(X), self.cap)
        out = pd.DataFrame({"rul_median": med}, index=model_input.index)
        if hasattr(self.fitter, "predict_partial_hazard"):
            out["partial_hazard"] = np.asarray(self.fitter.predict_partial_hazard(X))
        return out


# =========================================================================== gráficos
def plot_km(units: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    KaplanMeierFitter().fit(units.life, units.observed, label="todas").plot_survival_function(ax=ax, ci_show=True, lw=2)
    if units.fleet.nunique() > 1:
        for f, g in units.groupby("fleet"):
            KaplanMeierFitter().fit(g.life, g.observed, label=f).plot_survival_function(ax=ax, ci_show=False)
    ax.set(xlabel="ciclos", ylabel="S(t)", title="Kaplan-Meier — vida total dos motores (treino, com censura)")
    return fig


def plot_schoenfeld(cph, df, cols, max_cols=6):
    res = cph.compute_residuals(df, kind="scaled_schoenfeld")
    t = df.loc[res.index, "duration"].values
    order = np.argsort(t)
    rank = np.empty_like(order); rank[order] = np.arange(len(t))
    cols = cols[:max_cols]
    fig, axes = plt.subplots(int(np.ceil(len(cols) / 3)), 3, figsize=(13, 3.6 * np.ceil(len(cols) / 3)), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        r = res[c].values
        ax.scatter(rank, r, s=3, alpha=.25)
        smooth = pd.Series(r[order]).rolling(max(25, len(r) // 20), center=True, min_periods=5).mean()
        ax.plot(np.arange(len(r)), smooth, c="red", lw=2)
        ax.axhline(0, c="k", ls=":")
        ax.set(title=c, xlabel="rank(tempo)", ylabel="resíduo de Schoenfeld escalado")
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Resíduos de Schoenfeld — tendência ≠ 0 indica violação de riscos proporcionais")
    return fig


def plot_group_survival(test_snap, curves: dict, times):
    fleets = test_snap.fleet.values
    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4.2), squeeze=False)
    for ax, (name, S) in zip(axes.ravel(), curves.items()):
        for f in np.unique(fleets):
            m = fleets == f
            ax.plot(times, S[m].mean(0), lw=2, label=f"{f} — previsto (média)")
            KaplanMeierFitter().fit(test_snap.duration[m], test_snap.event[m]).plot_survival_function(
                ax=ax, ci_show=False, ls="--", label=f"{f} — KM observado")
        ax.set(title=name, xlabel="ciclos à frente (vida residual)", ylabel="S(t | histórico)")
        ax.legend(fontsize=7)
    fig.suptitle("Sobrevivência residual média por grupo operacional (teste)")
    return fig


def plot_risk_groups(test_snap, risk, label):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    q = pd.qcut(pd.Series(risk).rank(method="first"), 3, labels=["baixo risco", "médio", "alto risco"])
    for lab in q.cat.categories:
        m = (q == lab).values
        KaplanMeierFitter().fit(test_snap.duration[m], test_snap.event[m], label=lab).plot_survival_function(ax=ax)
    ax.set(title=f"KM do RUL real por tercil de risco previsto ({label})", xlabel="ciclos", ylabel="S(t)")
    return fig


def plot_rul(true, preds: dict):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.8 * len(preds), 4.3), squeeze=False)
    lim = max(np.max(true), *(np.max(p) for p in preds.values())) * 1.05
    for ax, (n, p) in zip(axes.ravel(), preds.items()):
        ax.scatter(true, p, s=12, alpha=.6)
        ax.plot([0, lim], [0, lim], "k:")
        ax.set(title=n, xlabel="RUL real", ylabel="RUL previsto (mediana)", xlim=(0, lim), ylim=(0, lim))
    return fig


# =========================================================================== bayesiano
def fit_bayesian_aft(train_snap, test_snap, feats, times, args):
    import arviz as az
    import pymc as pm

    rng = np.random.default_rng(args.seed)
    sub = (train_snap.groupby("unit_id", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), args.bayes_snapshots_per_unit), random_state=int(rng.integers(1_000_000_000)))))
    fleets = sorted(train_snap.fleet.unique())
    fidx = pd.Categorical(sub.fleet, categories=fleets).codes
    X, t, e = sub[feats].values, sub.duration.values, sub.event.values
    upper = np.where(e == 1, np.inf, t)  # censurado: só sabemos que T > t
    hier = len(fleets) > 1

    coords = {"feature": feats, "fleet": fleets}
    with pm.Model(coords=coords):
        mu_a = pm.Normal("mu_a", np.log(np.median(t)), 1.5)
        if hier:  # parametrização não centrada (evita "funil" no NUTS)
            sigma_a = pm.HalfNormal("sigma_a", 0.5)
            a = pm.Deterministic("a_fleet", mu_a + pm.Normal("a_offset", 0, 1, dims="fleet") * sigma_a, dims="fleet")
            intercept = a[fidx]
        else:
            intercept = mu_a
        beta = pm.Normal("beta", 0, 0.5, dims="feature")
        k = pm.Gamma("k_shape", alpha=2.0, beta=1.0)
        lam = pm.math.exp(intercept + pm.math.dot(X, beta))
        pm.Censored("T", pm.Weibull.dist(alpha=k, beta=lam), lower=None, upper=upper, observed=t)

        kw = dict(draws=args.bayes_draws, tune=args.bayes_tune, chains=args.bayes_chains,
                  random_seed=args.seed, target_accept=args.bayes_target_accept, progressbar=False)
        try:
            idata = pm.sample(nuts_sampler="nutpie", **kw)
            sampler = "nutpie"
        except Exception as exc:
            print(f"[aviso] nutpie indisponível ({exc}); usando sampler NUTS do PyMC")
            idata = pm.sample(cores=1, **kw)
            sampler = "pymc"
        pm.compute_log_likelihood(idata)

    post = idata.posterior
    beta_d = post["beta"].stack(s=("chain", "draw")).values            # (F, S)
    k_d = post["k_shape"].stack(s=("chain", "draw")).values            # (S,)
    if hier:
        a_d = post["a_fleet"].stack(s=("chain", "draw")).values        # (G, S)
        tf = pd.Categorical(test_snap.fleet, categories=fleets).codes
        log_lam = a_d[tf] + test_snap[feats].values @ beta_d           # (n, S)
    else:
        log_lam = post["mu_a"].stack(s=("chain", "draw")).values[None, :] + test_snap[feats].values @ beta_d
    lam = np.exp(log_lam)
    S = np.exp(-(times[None, :, None] / lam[:, None, :]) ** k_d[None, None, :]).mean(-1)  # (n, times)
    rul_med = np.median(lam * np.log(2) ** (1 / k_d[None, :]), axis=1)

    summ = az.summary(idata, var_names=["beta", "k_shape", "mu_a"] + (["sigma_a", "a_fleet"] if hier else []),
                      hdi_prob=0.94)
    loo, waic = az.loo(idata), az.waic(idata)
    diag = {"bayes_rhat_max": float(summ.r_hat.max()), "bayes_ess_bulk_min": float(summ.ess_bulk.min()),
            "bayes_divergences": int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1,
            "bayes_elpd_loo": float(loo.elpd_loo), "bayes_p_loo": float(loo.p_loo),
            "bayes_looic": float(-2 * loo.elpd_loo), "bayes_pareto_k_max": float(np.max(loo.pareto_k.values)),
            "bayes_elpd_waic": float(waic.elpd_waic), "bayes_waic": float(-2 * waic.elpd_waic),
            "bayes_n_obs": len(sub)}
    return dict(idata=idata, summary=summ, S=S, risk=-np.log(rul_med), rul=rul_med, diag=diag,
                sampler=sampler, hier=hier)


# =========================================================================== main
def main(args):
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    setup_mlflow(EXPERIMENT)

    train_raw, test_raw, test_rul = load_subsets(Path(args.data_dir), args.subsets)
    sensors = informative_sensors(train_raw)
    norm = RegimeNormalizer().fit(train_raw, sensors)
    train_f = rolling_features(norm.transform(train_raw), sensors, args.window)
    test_f = rolling_features(norm.transform(test_raw), sensors, args.window)
    feats = ["cycle"] + [c for c in train_f.columns if c.endswith(("_mean", "_std", "_slope"))]

    units = train_raw.unit_id.unique()
    rng.shuffle(units)
    n_val = int(len(units) * args.val_frac)
    val_units, tr_units = units[:n_val], units[n_val:]
    life = train_raw.groupby("unit_id").cycle.max()
    end_tr, ev_tr = apply_censoring(train_raw, tr_units, args.censor_frac, rng)
    end_tr, ev_tr = end_tr[tr_units], ev_tr[tr_units]

    tr_snap = snapshots(train_f, end_tr, ev_tr, args.stride, args.window)
    va_snap = snapshots(train_f, life[val_units], pd.Series(1, index=val_units), args.stride, args.window)
    te_snap = test_f.groupby("unit_id", sort=False).tail(1).reset_index(drop=True)
    te_snap["duration"] = te_snap.unit_id.map(test_rul).astype(float)
    te_snap["event"] = 1

    mu, sd = tr_snap[feats].mean(), tr_snap[feats].std().replace(0, 1)
    z = lambda d: d.assign(**((d[feats] - mu) / sd))  # noqa: E731
    tr_z, va_z, te_z = z(tr_snap), z(va_snap), z(te_snap)
    cap = float(tr_snap.duration.max() * 1.5)
    times = np.linspace(max(1.0, np.percentile(te_snap.duration, 2)), np.percentile(te_snap.duration, 95), 60)

    with mlflow.start_run(run_name="survival_rul_" + "_".join(args.subsets)) as parent:
        mlflow.set_tags({"data_source": data_source_tag(args.data_dir), "project": "04_survival", "dataset": "nasa_cmapss", "stage": "experimentation",
                         "subsets": ",".join(args.subsets)})
        mlflow.log_params({"window": args.window, "stride": args.stride, "censor_frac": args.censor_frac,
                           "val_frac": args.val_frac, "n_sensors_used": len(sensors), "n_features": len(feats),
                           "sensors_used": ",".join(sensors), "n_train_units": len(tr_units),
                           "n_val_units": len(val_units), "n_test_units": len(te_snap),
                           "n_train_snapshots": len(tr_snap), "n_operating_regimes": regime_key(train_raw).nunique()})
        mlflow.log_metric("train_censoring_rate_units", float(1 - ev_tr.mean()))
        log_dataset(tr_snap, "cmapss_train_snapshots")

        # -------------------------------------------- Kaplan-Meier
        km_units = pd.DataFrame({"life": end_tr, "observed": ev_tr, "fleet": train_raw.drop_duplicates("unit_id")
                                 .set_index("unit_id").loc[tr_units, "fleet"]})
        log_fig(plot_km(km_units), "plots/kaplan_meier_by_fleet.png")
        mlflow.log_metric("km_median_life", float(KaplanMeierFitter().fit(km_units.life, km_units.observed).median_survival_time_))

        cols_fit = feats + ["duration", "event", "unit_id"]
        results, preds, curves = {}, {}, {}

        # -------------------------------------------- Cox ElasticNet: grid
        best = (None, -np.inf, None)
        for pen in args.penalizers:
            for l1 in args.l1_ratios:
                with mlflow.start_run(run_name=f"coxph_pen{pen}_l1{l1}", nested=True):
                    mlflow.log_params({"model": "CoxPH", "penalizer": pen, "l1_ratio": l1, "robust_cluster": "unit_id"})
                    cph = CoxPHFitter(penalizer=pen, l1_ratio=l1)
                    cph.fit(tr_z[cols_fit], "duration", "event", cluster_col="unit_id")
                    ci = concordance_index(va_z.duration, -cph.predict_partial_hazard(va_z[feats]), va_z.event)
                    nz = int((cph.params_.abs() > 1e-3).sum())
                    mlflow.log_metrics({"val_c_index": ci, "n_nonzero_coefs": nz,
                                        "train_partial_aic": float(cph.AIC_partial_)})
                    print(f"Cox pen={pen} l1={l1}: val C={ci:.4f} nz={nz}")
                    if ci > best[1]:
                        best = (cph, ci, {"penalizer": pen, "l1_ratio": l1})
        cph, _, cph_params = best

        with mlflow.start_run(run_name="coxph_best", nested=True):
            mlflow.log_params({"model": "CoxPH_ElasticNet", **cph_params})
            S_cox = cph.predict_survival_function(te_z[feats], times=times).T.values
            preds["CoxPH"] = finite_median(cph.predict_median(te_z[feats]), cap)
            m = evaluate("CoxPH", te_z.duration, te_z.event, cph.predict_partial_hazard(te_z[feats]),
                         S_cox, times, preds["CoxPH"], "test_")
            m.update(evaluate("CoxPH-val", va_z.duration, va_z.event, cph.predict_partial_hazard(va_z[feats]),
                              None, times, None, "val_"))
            mlflow.log_metrics(m); results["CoxPH"] = m; curves["CoxPH"] = S_cox
            log_df(cph.summary, "tables/coxph_summary.csv")

            # Schoenfeld: teste formal + gráficos para as covariáveis mais relevantes
            ph = proportional_hazard_test(cph, tr_z[cols_fit], time_transform="rank").summary
            log_df(ph, "tables/schoenfeld_test.csv")
            mlflow.log_metrics({"schoenfeld_min_p": float(ph.p.min()),
                                "schoenfeld_n_violations_p05": int((ph.p < 0.05).sum()),
                                "schoenfeld_frac_violations_p05": float((ph.p < 0.05).mean())})
            top = cph.params_.abs().sort_values(ascending=False).index.tolist()
            log_fig(plot_schoenfeld(cph, tr_z[cols_fit], top), "plots/schoenfeld_residuals.png")
            fig, ax = plt.subplots(figsize=(7, 0.32 * min(25, len(top)) + 1.5))
            cph.plot(columns=top[:25], ax=ax)
            ax.set_title("Cox ElasticNet — log(HR) com IC 95% (robusto por motor)")
            log_fig(fig, "plots/coxph_coefficients.png")

        # -------------------------------------------- Cox tempo-variante (start/stop)
        if not args.no_ctv:
            with mlflow.start_run(run_name="cox_time_varying", nested=True):
                tv_feats = [f for f in feats if f != "cycle"]
                long = train_f[train_f.unit_id.isin(tr_units)].copy()
                long["end"] = long.unit_id.map(end_tr)
                long = long[(long.cycle <= long.end) & ((long.cycle % args.stride == 0) | (long.cycle == long.end))]
                long = long.sort_values(["unit_id", "cycle"])
                g = long.groupby("unit_id")
                long["start"] = g.cycle.shift(1)
                long["stop"] = long.cycle
                long["event"] = ((long.cycle == long.end) & long.unit_id.map(ev_tr).eq(1)).astype(int)
                # covariáveis do intervalo (start, stop] = valores conhecidos em `start` (sem look-ahead)
                long[tv_feats] = g[tv_feats].shift(1)
                long = long.dropna(subset=["start"]).copy()
                long[tv_feats] = (long[tv_feats] - mu[tv_feats]) / sd[tv_feats]
                # O lifelines multiplica a penalidade pelo nº de LINHAS. Aqui há ~vida/stride intervalos por
                # motor e no máximo 1 evento por motor, então reescalamos por eventos/linhas para que a
                # força da regularização seja comparável à do Cox de snapshots (senão β → 0).
                pen_ctv = cph_params["penalizer"] * max(long.event.sum(), 1) / len(long)
                ctv = CoxTimeVaryingFitter(penalizer=pen_ctv, l1_ratio=cph_params["l1_ratio"])
                ctv.fit(long[["unit_id", "start", "stop", "event"] + tv_feats], id_col="unit_id",
                        event_col="event", start_col="start", stop_col="stop")
                # Uso: inferência (HR do estado ATUAL de degradação) + ranking de risco no último ciclo.
                # Não geramos RUL/S(t) com ele: as covariáveis são internas e crescem com o desgaste;
                # congelá-las no último valor subestima o risco futuro (exigiria um joint model).
                m = evaluate("CoxTimeVarying", te_z.duration, te_z.event,
                             ctv.predict_partial_hazard(te_z[tv_feats]), None, times, None, "test_")
                mlflow.log_params({"model": "CoxTimeVarying", "n_intervals": len(long), "penalizer_rescaled": pen_ctv,
                                   "l1_ratio": cph_params["l1_ratio"]})
                mlflow.log_metrics(m); results["CoxTimeVarying"] = m
                log_df(ctv.summary, "tables/cox_time_varying_summary.csv")

        # -------------------------------------------- Weibull AFT (lifelines)
        with mlflow.start_run(run_name="weibull_aft", nested=True):
            aft = WeibullAFTFitter(penalizer=args.aft_penalizer)
            aft.fit(tr_z[feats + ["duration", "event"]], "duration", "event")
            S_aft = aft.predict_survival_function(te_z[feats], times=times).T.values
            preds["WeibullAFT"] = finite_median(aft.predict_median(te_z[feats]), cap)
            m = evaluate("WeibullAFT", te_z.duration, te_z.event, -preds["WeibullAFT"], S_aft, times,
                         preds["WeibullAFT"], "test_")
            va_med = finite_median(aft.predict_median(va_z[feats]), cap)
            m.update(evaluate("WeibullAFT-val", va_z.duration, va_z.event, -va_med, None, times, None, "val_"))
            mlflow.log_params({"model": "WeibullAFT", "penalizer": args.aft_penalizer})
            mlflow.log_metrics({**m, "weibull_shape_rho": float(np.exp(aft.params_.loc["rho_"].iloc[0])),
                                "train_aic": float(aft.AIC_)})
            results["WeibullAFT"] = m; curves["WeibullAFT"] = S_aft
            log_df(aft.summary, "tables/weibull_aft_summary.csv")

        # -------------------------------------------- Bayesiano hierárquico (PyMC)
        if not args.no_bayes:
            import arviz as az
            with mlflow.start_run(run_name="bayesian_weibull_aft_pymc", nested=True):
                bfeats = cph.params_.abs().sort_values(ascending=False).index[:args.bayes_top_features].tolist()
                mlflow.log_params({"model": "BayesianWeibullAFT", "features": ",".join(bfeats),
                                   "draws": args.bayes_draws, "tune": args.bayes_tune, "chains": args.bayes_chains,
                                   "snapshots_per_unit": args.bayes_snapshots_per_unit,
                                   "priors": "beta~N(0,0.5); k~Gamma(2,1); a_fleet~N(mu_a,sigma_a); sigma_a~HN(0.5)"})
                b = fit_bayesian_aft(tr_z, te_z, bfeats, times, args)
                mlflow.log_params({"sampler": b["sampler"], "hierarchical": b["hier"]})
                preds["BayesWeibullAFT"] = b["rul"]
                m = evaluate("BayesWeibullAFT", te_z.duration, te_z.event, b["risk"], b["S"], times, b["rul"], "test_")
                mlflow.log_metrics({**m, **b["diag"]}); results["BayesWeibullAFT"] = {**m, **b["diag"]}
                curves["BayesWeibullAFT"] = b["S"]
                log_df(b["summary"], "tables/bayes_posterior_summary_hdi94.csv")
                axes = az.plot_posterior(b["idata"], var_names=["beta"], hdi_prob=0.94, ref_val=0)
                log_fig(np.ravel(axes)[0].figure, "plots/bayes_posterior_beta_hdi.png")
                axes = az.plot_forest(b["idata"], var_names=["beta"] + (["a_fleet"] if b["hier"] else []),
                                      hdi_prob=0.94, combined=True)
                log_fig(np.ravel(axes)[0].figure, "plots/bayes_forest_hdi.png")
                axes = az.plot_trace(b["idata"], var_names=["beta", "k_shape"])
                log_fig(np.ravel(axes)[0].figure, "plots/bayes_trace.png")
                b["idata"].to_netcdf("idata.nc")
                mlflow.log_artifact("idata.nc", artifact_path="posterior")
                Path("idata.nc").unlink()

        # -------------------------------------------- comparação, gráficos e registro
        log_fig(plot_group_survival(te_snap, curves, times), "plots/survival_curves_by_group.png")
        log_fig(plot_risk_groups(te_snap, -preds["CoxPH"], "CoxPH"), "plots/km_by_predicted_risk_tercile.png")
        log_fig(plot_rul(te_snap.duration.values, preds), "plots/rul_pred_vs_true.png")
        comp = pd.DataFrame(results).T
        log_df(comp.round(4), "tables/model_comparison.csv")
        print(comp.round(4).to_string())

        # escolhe o modelo para registro pelo C-index de VALIDAÇÃO (o teste fica intocado)
        val_ci = {"CoxPH": results["CoxPH"]["val_c_index"], "WeibullAFT": results["WeibullAFT"]["val_c_index"]}
        chosen = max(val_ci, key=val_ci.get)
        fitter = cph if chosen == "CoxPH" else aft
        mlflow.log_param("registered_model_choice", chosen)
        for k in ["c_index", "integrated_brier_score", "rul_rmse"]:
            mlflow.log_metric(f"best_test_{k}", results[chosen][f"test_{k}"])
        wrapper = SurvivalRULModel(fitter, feats, mu, sd, cap)
        example = te_snap[feats].head(5)
        mlflow.pyfunc.log_model(python_model=wrapper, input_example=example,
                                registered_model_name="turbofan_rul_survival",
                                **model_name_kwarg("model", mlflow.pyfunc.log_model))
        log_json({"chosen": chosen, "features": feats, "sensors": sensors, "results": results}, "summary.json")
        print(f"Run: {parent.info.run_id}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_RAW / "cmapss"))
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"],
                   help="mais de um subset ativa o intercepto hierárquico por frota no modelo bayesiano")
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--stride", type=int, default=10, help="espaçamento (ciclos) entre snapshots")
    p.add_argument("--censor-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--penalizers", type=float, nargs="+", default=[0.01, 0.05, 0.2])
    p.add_argument("--l1-ratios", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    p.add_argument("--aft-penalizer", type=float, default=0.01)
    p.add_argument("--no-ctv", action="store_true", help="pula o Cox tempo-variante")
    p.add_argument("--no-bayes", action="store_true", help="pula o modelo PyMC")
    p.add_argument("--bayes-top-features", type=int, default=6)
    p.add_argument("--bayes-snapshots-per-unit", type=int, default=3)
    p.add_argument("--bayes-draws", type=int, default=1000)
    p.add_argument("--bayes-tune", type=int, default=1000)
    p.add_argument("--bayes-chains", type=int, default=4)
    p.add_argument("--bayes-target-accept", type=float, default=0.95,
                   help="passo menor no NUTS; com poucas frotas sigma_a gera geometria difícil")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.penalizers, a.l1_ratios = [0.05], [0.5]
        a.bayes_draws, a.bayes_tune, a.bayes_chains = 500, 500, 2
    return a


if __name__ == "__main__":
    main(parse_args())
