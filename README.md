# Portfólio de ML com MLflow — 4 projetos

| # | Projeto | Módulo | Experimento MLflow | Modelo registrado |
|---|---|---|---|---|
| 1 | Segmentação de clientes (K-Means vs GMM) | `p01_clustering` | `01_clustering_credit_card` | `credit_card_segmentation` |
| 2 | Churn com LightGBM + Optuna + calibração | `p02_churn` | `02_ensemble_telco_churn` | `telco_churn_lgbm_calibrated` |
| 3 | Genômica p≫n: PCA + UMAP vs Autoencoder | `p03_dimred` | `03_dimred_gene_expression` | `gene_expression_pca_umap`, `gene_expression_sparse_ae_latent{2,10}` |
| 4 | Sobrevivência / RUL de turbofans | `p04_survival` | `04_survival_turbofan_rul` | `turbofan_rul_survival` |

> **Documentação completa e resultados:** [`docs/`](docs/README.md) — um arquivo por projeto, guia do
> MLflow, registro de decisões/correções e limitações. Resumo dos resultados com os dados reais:
>
> | Projeto | Resultado |
> |---|---|
> | 1 · Clusterização | K-Means k=6, silhueta 0,250, 6 segmentos interpretáveis |
> | 2 · Churn | PR-AUC 0,659 · ROC-AUC 0,843 · −40% de custo vs. estratégia sem modelo |
> | 3 · Genômica | UMAP trustworthiness 0,960 · 99,6% kNN · AE reconstrói melhor que PCA |
> | 4 · Sobrevivência | C-index 0,840 · RUL com viés otimista documentado |

```
common/mlflow_utils.py   setup do tracking, log de figuras/tabelas/datasets, compat. MLflow 2/3
data/download_data.py    baixa os datasets (Kaggle API + UCI)
data/make_synthetic.py   dados sintéticos com o mesmo schema (smoke test sem Kaggle)
scripts/start_mlflow.*   tracking server SQLite + proxy de artefatos
pNN_*/train.py           pipeline completo de cada projeto (python -m pNN_xxx.train --help)
```

## 1. Ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # ou requirements-lock.txt para as versões exatas usadas
```

## 2. Tracking server

```powershell
.\scripts\start_mlflow.ps1      # http://127.0.0.1:5000
```

Duas diferenças em relação ao comando original:
- `--artifacts-destination ./mlartifacts` no lugar de `--default-artifact-root ./mlruns`. Com caminho relativo
  no `default-artifact-root`, cada cliente grava os artefatos relativo ao *seu* diretório atual; com o proxy,
  o servidor centraliza tudo.
- `--workers 1`. O padrão são 4 processos uvicorn; com SQLite eles disputam o lock de escrita e, nos testes no
  Windows, alguns requests ficaram pendurados até o timeout de 120 s do cliente. Com 1 worker não ocorreu mais.

Se matar o servidor à força no Windows, confira se não sobraram workers órfãos segurando a porta 5000.

Sem servidor (ex.: CI): `$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"`.

## 3. Dados

1. Token do Kaggle (Kaggle → Settings → API). No formato novo, salve a string em
   `%USERPROFILE%\.kaggleccess_token` (ver [docs/00_visao_geral.md](docs/00_visao_geral.md));
   no formato antigo, o arquivo `kaggle.json` vai na mesma pasta. Nunca versione o token.
2. `python data/download_data.py` → `data/raw/{ccdata,telco,gene_expression,cmapss}`.

**Atenção, projeto 3:** o link `crawford/gene-expression` é o dataset de leucemia de Golub (72 × 7.129).
O dataset descrito (801 pacientes × 20.531 genes, 5 tumores) é o **TCGA PANCAN da UCI**, que o script baixa
por padrão. O loader aceita os dois (`python data/download_data.py gene_golub` para o Golub).

Para testar sem Kaggle: `python data/make_synthetic.py` e passe `--data-dir data/synthetic/<nome> --quick`.

## 4. Execução

```powershell
python -m p01_clustering.train
python -m p02_churn.train --n-trials 60 --cost-fn 500 --cost-fp 100
python -m p03_dimred.train
python -m p04_survival.train --subsets FD001 FD003
```

## Serialização de modelos (MLflow 3)

O MLflow 3 salva sklearn em `skops` por padrão e recusa tipos não confiáveis. `common.log_sklearn_model`
tenta skops primeiro e cai para cloudpickle quando há estimadores de terceiros (LightGBM, UMAP) — carregue
esses modelos só de um tracking server confiável. No projeto 3 o encoder do Autoencoder é registrado em
formato pickle (o padrão `pt2`/torch.export fixaria o tamanho do batch); o `state_dict` completo vai junto.

## O que cada projeto faz (e decisões importantes)

### 1 · Clusterização
- Mediana → Yeo-Johnson → RobustScaler, num `Pipeline` único (o modelo registrado recebe o CSV bruto).
- Varredura k=2..10: K-Means (k-means++) e GMM `full/diag/tied` (BIC/AIC). Cada combinação = run aninhado.
- Seleção pela silhueta, descartando soluções com cluster < 3% dos clientes (`--min-cluster-frac`).
- Estabilidade: ARI entre re-treinos com seeds diferentes.
- Artefatos: cotovelo, BIC/AIC, métricas×k, silhouette plot por cluster, PCA 2D com centróides,
  perfil dos segmentos (medianas na escala original + heatmap).

### 2 · Churn
- `TargetEncoder` do sklearn dentro do `Pipeline` → cross-fitting interno e re-ajuste em cada fold:
  sem vazamento de alvo. Inclui uma interação Contract×Payment×Internet (alta cardinalidade).
- Optuna/TPE maximiza PR-AUC em CV estratificada; cada trial vira um run aninhado.
- Calibração sigmoide vs isotônica escolhida pelo **Brier OOF do treino**; threshold ótimo de negócio
  (custo FN vs FP) também escolhido em OOF e só depois aplicado no teste.
- Métricas: PR-AUC, ROC-AUC, Brier, ECE, log-loss, F1-macro, custo esperado vs estratégias ingênuas.
- Artefatos: curvas PR/ROC, diagrama de confiabilidade, curva custo×threshold, matriz de confusão,
  SHAP (beeswarm, barras, waterfall do cliente de maior risco), booster LightGBM nativo (`.txt`).
  ONNX não foi incluído: TargetEncoder + CalibratedClassifierCV não têm conversor no skl2onnx.

### 3 · Redução de dimensionalidade
- Normalização automática (log-CPM para contagens; log2 com clipping para microarray; nada se já log).
- Filtro de variância + top-N genes, tudo ajustado só no treino; validação mede reconstrução.
- PCA: scree plot, nº de PCs para 85/90/95%, MSE de reconstrução.
- UMAP sobre PCA-50 com varredura de `n_neighbors`; Autoencoder PyTorch com L1 no latente, comparado
  ao PCA **na mesma dimensão** (MSE de validação, trustworthiness, kNN-acc, silhueta por tumor).
- Artefatos: scatter interativo (Plotly HTML) do UMAP/PCA/AE, curvas de loss por época.

### 4 · Sobrevivência
- Landmarking: snapshots a cada `--stride` ciclos, features de janela móvel (média, desvio, inclinação)
  dos sensores normalizados por regime operacional; alvo = vida residual com censura.
- Censura à direita real: `--censor-frac` dos motores de treino são truncados (evento=0).
- Modelos: Kaplan-Meier, Cox ElasticNet (grid, SE robusto por motor), Cox tempo-variante (start/stop,
  covariáveis do início do intervalo), Weibull AFT e **Weibull AFT Bayesiano hierárquico** (PyMC,
  intercepto por frota com pooling parcial, `pm.Censored`).
- Cox tempo-variante: o lifelines multiplica a penalidade pelo nº de *linhas*; no formato start/stop há
  dezenas de intervalos por motor e 1 evento, então a penalidade é reescalada por eventos/linhas (sem isso
  os β iam a ~0). Ele é usado para inferência e ranking de risco, **não** para prever RUL: as covariáveis
  são internas e crescem com o desgaste, e congelá-las no último valor subestima o risco futuro (isso
  pediria um *joint model*). RUL e Brier vêm do Cox de snapshots, do Weibull AFT e do Bayesiano.
- O modelo registrado (`turbofan_rul_survival`) é escolhido pelo C-index de **validação**, não de teste.
- Diagnósticos: teste e gráficos de resíduos de Schoenfeld, R-hat/ESS/divergências, WAIC e LOO (PSIS).
- Métricas: C-index, Brier dinâmico IPCW + IBS, RMSE/MAE do RUL, score NASA.
- Limitação: snapshots do mesmo motor são correlacionados — no Cox isso é tratado com SE robusto por
  cluster; no modelo Bayesiano usamos poucos snapshots por motor, e WAIC/LOO são aproximados.
- Sampler: `nutpie` (compila via numba). Sem g++ no Windows o NUTS puro do PyMC fica muito lento.
