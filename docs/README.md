# Documentação — Portfólio de ML com MLflow

Quatro projetos de Machine Learning (clusterização, ensemble supervisionado, redução de dimensionalidade
e análise de sobrevivência), todos rastreados num servidor MLflow local e executados nos datasets reais.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| [00_visao_geral.md](00_visao_geral.md) | Objetivo, estrutura do repositório, ambiente, como reproduzir tudo |
| [01_clusterizacao.md](01_clusterizacao.md) | Segmentação de clientes de cartão — K-Means vs GMM |
| [02_churn.md](02_churn.md) | Churn Telco — LightGBM + Optuna, calibração, threshold de negócio, SHAP |
| [03_genomica.md](03_genomica.md) | RNA-Seq TCGA (p ≫ n) — PCA + UMAP vs Autoencoder esparso |
| [04_sobrevivencia.md](04_sobrevivencia.md) | Turbofans NASA C-MAPSS — Cox, Cox tempo-variante, Weibull AFT, AFT Bayesiano |
| [05_guia_mlflow.md](05_guia_mlflow.md) | Como subir o servidor e navegar na interface para ver tudo o que foi registrado |
| [06_decisoes_e_correcoes.md](06_decisoes_e_correcoes.md) | Registro de decisões técnicas e dos bugs encontrados e corrigidos |
| [07_limitacoes_e_proximos_passos.md](07_limitacoes_e_proximos_passos.md) | O que os resultados não sustentam e o que fazer em seguida |

Gráficos, tabelas e métricas de cada run estão exportados em [`assets/`](assets/) (um `runs.json` por
projeto, com todos os parâmetros e métricas, inclusive dos runs aninhados).

## Resumo dos resultados (dados reais)

| Projeto | Modelo final | Métrica principal | Leitura |
|---|---|---|---|
| 1 · Clusterização | K-Means, k=6 | Silhueta 0,250 · estabilidade (ARI) 0,996 | 6 segmentos interpretáveis; k=2 tinha silhueta maior (0,278) mas só separava "compra" de "saque" |
| 2 · Churn | LightGBM + calibração isotônica | PR-AUC 0,659 · ROC-AUC 0,843 · Brier 0,136 | Economia de 40,9 mil (−40%) vs. melhor estratégia sem modelo; regressão logística chega quase lá (PR-AUC 0,651) |
| 3 · Genômica | UMAP sobre PCA-50; AE esparso | Trustworthiness 0,960 (UMAP) · kNN 99,6% | 5 tumores separados; AE reconstrói melhor que PCA na mesma dimensão |
| 4 · Sobrevivência | Cox ElasticNet | C-index 0,840 (teste) · IBS 0,098 | Bom ranking de risco; RUL superestimado em todas as faixas (viés +13 a +28 ciclos) |

## Runs de referência no MLflow

| Experimento | Run (ID) | Modelo registrado (`@champion`) |
|---|---|---|
| `01_clustering_credit_card` | `73bfd09056dc491bb02b15dff8db0dcb` (k≥4) · `4efc92ecfb7649ff9b3bef51ab16f890` (k=2) | `credit_card_segmentation` v4 (`@k2_baseline` = v3) |
| `02_ensemble_telco_churn` | `09a557af18f14cd19cc4a76ac90f821e` | `telco_churn_lgbm_calibrated` v4 |
| `03_dimred_gene_expression` | `4a9b3de1e5934e889b7ae8e197926c66` | `gene_expression_pca_umap` v4, `gene_expression_sparse_ae_latent2` v3, `..._latent10` v3 |
| `04_survival_turbofan_rul` | `5522955c23af4b8ead96f19cda8d6cc0` | `turbofan_rul_survival` v4 |

Os IDs valem para o banco `mlflow.db` local onde os experimentos foram rodados. Quem clonar o repositório
e reexecutar os scripts terá IDs novos.
