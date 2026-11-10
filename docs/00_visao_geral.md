# 00 · Visão geral

## Objetivo

Construir quatro projetos de ML de categorias diferentes, com pipelines reprodutíveis e todo o ciclo
(parâmetros, métricas, gráficos, datasets e modelos) registrado num servidor MLflow local:

| # | Categoria | Dataset | Técnicas |
|---|---|---|---|
| 1 | Não supervisionado clássico | Credit Card Dataset for Clustering (Kaggle `arjunbhasin2013/ccdata`) | K-Means, Gaussian Mixture Models |
| 2 | Supervisionado tabular | Telco Customer Churn (Kaggle `blastchar/telco-customer-churn`) | LightGBM, Optuna, calibração, SHAP |
| 3 | Redução de dimensionalidade | Gene Expression Cancer RNA-Seq — TCGA PANCAN (UCI 401) | PCA, UMAP, Autoencoder (PyTorch) |
| 4 | Estatística / sobrevivência | NASA Turbofan C-MAPSS (Kaggle `behrad3d/nasa-cmaps`) | Kaplan-Meier, Cox, Weibull AFT, PyMC |

## Estrutura do repositório

```
common/mlflow_utils.py      setup do tracking, helpers de log (figuras, tabelas, datasets, modelos)
data/download_data.py       baixa os 4 datasets (Kaggle API + UCI)
data/make_synthetic.py      dados sintéticos com o mesmo schema (smoke test sem Kaggle)
scripts/start_mlflow.ps1    sobe o tracking server (Windows)
scripts/start_mlflow.sh     sobe o tracking server (Linux/macOS)
p01_clustering/train.py     projeto 1
p02_churn/train.py          projeto 2
p03_dimred/train.py         projeto 3
p04_survival/train.py       projeto 4
docs/                       esta documentação + assets exportados do MLflow
requirements.txt            dependências
```

Cada `train.py` é um script único e autocontido: lê os dados, executa o pipeline inteiro, registra tudo no
MLflow e registra o modelo final no Model Registry. Todos aceitam `--help`, `--data-dir`, `--seed` e
`--quick` (versão reduzida para teste rápido).

## Ambiente

- Python 3.11, Windows 11 (testado também o caminho de dados sintéticos).
- Principais versões instaladas: MLflow 3.18, scikit-learn 1.9, LightGBM 4.7, Optuna 4, SHAP 0.51,
  lifelines 0.30, PyMC 5.28 + nutpie 0.16, PyTorch 2.15 (CPU), UMAP 0.5, NumPy 2.4.
- Ambiente virtual **isolado** (`python -m venv .venv`, sem `--system-site-packages`): o Python global da
  máquina tinha pacotes compilados para NumPy 1.x, incompatíveis com o NumPy 2 exigido pelo PyMC.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como reproduzir tudo

1. **Servidor MLflow** (deixe rodando num terminal à parte):
   ```powershell
   .\scripts\start_mlflow.ps1
   ```
2. **Credencial do Kaggle**: gere um token em kaggle.com → Settings → API. O formato novo é uma string
   que vai no arquivo `%USERPROFILE%\.kaggle\access_token` (sem quebra de linha, em ASCII) ou na
   variável `KAGGLE_API_TOKEN`. Nunca versione esse arquivo.
   ```powershell
   Set-Content "$env:USERPROFILE\.kaggle\access_token" -Value "SEU_TOKEN" -NoNewline -Encoding ascii
   ```
3. **Dados** (~240 MB em `data/raw/`):
   ```powershell
   python data/download_data.py
   ```
4. **Pipelines**:
   ```powershell
   python -m p01_clustering.train --k-min 4
   python -m p02_churn.train --n-trials 60
   python -m p03_dimred.train
   python -m p04_survival.train --subsets FD001 FD003
   ```

Tempos medidos nesta máquina (CPU): projeto 1 ≈ 1–2 min, projeto 2 ≈ 4,5 min, projeto 3 ≈ 4 min,
projeto 4 ≈ 3,5 min. Os projetos 1+2 e 3+4 rodaram em paralelo sem problema com o servidor de 1 worker.

### Teste rápido sem Kaggle

```powershell
python data/make_synthetic.py
python -m p01_clustering.train --data-dir data/synthetic/ccdata --quick
```

Runs feitos com dados sintéticos recebem a tag `data_source = synthetic` automaticamente (e os reais,
`real`), então dá para separá-los na interface. Ver [05_guia_mlflow.md](05_guia_mlflow.md).

## Padrão de registro no MLflow (comum aos 4 projetos)

- **Um experimento por projeto** (`01_clustering_credit_card`, `02_ensemble_telco_churn`,
  `03_dimred_gene_expression`, `04_survival_turbofan_rul`).
- **Um run pai por execução**, com **runs aninhados** para cada configuração testada (cada k, cada trial
  do Optuna, cada `n_neighbors` do UMAP, cada combinação de penalização do Cox, cada modelo).
- Tags: `project`, `dataset`, `stage`, `data_source`. Descrição (nota) nos runs reais.
- Parâmetros e métricas do pipeline; gráficos em `plots/`, tabelas em `tables/`, resumo em `summary.json`.
- Dataset de entrada registrado com `mlflow.log_input` (aba *Datasets*: schema e digest).
- Modelo final registrado no **Model Registry**, com alias `@champion` na versão treinada com dados reais.

## Proteções contra vazamento de informação (resumo)

| Projeto | Proteção |
|---|---|
| 1 | Não há alvo; o pré-processamento faz parte do modelo registrado (mesma transformação em produção) |
| 2 | Target encoding com *cross-fitting* dentro do `Pipeline`; calibração e threshold escolhidos em predições **out-of-fold do treino**; teste usado só no fim |
| 3 | Filtro de genes, padronização, PCA, UMAP e AE ajustados **só no treino**; reconstrução medida na validação |
| 4 | Divisão **por motor** (não por linha); hiperparâmetros e modelo registrado escolhidos pela **validação**; teste oficial intocado até a avaliação final |
