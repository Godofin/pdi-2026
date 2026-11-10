# 06 · Registro de decisões e correções

Tudo o que mudou em relação ao plano original, e todo bug encontrado durante o desenvolvimento, na
ordem em que apareceu. Os itens 1–9 foram pegos nos testes com dados sintéticos; os itens 10–12 só
apareceram com os dados reais.

## Mudanças em relação ao plano original

| Plano original | O que foi feito | Motivo |
|---|---|---|
| `mlflow server --default-artifact-root ./mlruns` | `--artifacts-destination ./mlartifacts` | Com caminho relativo, cada **cliente** grava artefatos relativo à sua pasta atual; com o proxy, o servidor centraliza |
| `mlflow server` (4 workers padrão) | `--workers 1` | Ver item 6 |
| Kaggle `arjunbhasin2008/ccdata` | `arjunbhasin2013/ccdata` | O slug original não existe (403) |
| Kaggle `crawford/gene-expression` | TCGA PANCAN da UCI | O link era o dataset de Golub (72 × 7.129); o descrito no projeto é o PANCAN (801 × 20.531). O loader aceita os dois |
| "Log-CPM" no projeto 3 | Normalização automática | O PANCAN já vem normalizado; log-CPM seria aplicado duas vezes |
| "85% da variância em 50 componentes" | 169 componentes | Valor medido, não suposição |
| Modelo serializado "nativo/ONNX" | Nativo (booster LightGBM) + sklearn | `skl2onnx` não converte `TargetEncoder` + `CalibratedClassifierCV` |
| "Target Encoding / Target Splitting dentro da CV" | `TargetEncoder` do sklearn no `Pipeline` | Faz cross-fitting interno; dentro do `Pipeline`, é re-ajustado em cada fold |
| Cox **ou** Bayesiano | Os dois, mais Cox tempo-variante, Weibull AFT e Kaplan-Meier | Comparação direta das abordagens |
| `pip install` no Python global | Ambiente virtual isolado | Ver item 1 |

## Bugs e problemas encontrados

### 1. Conflito de NumPy no ambiente

**Sintoma:** `AttributeError: _ARRAY_API not found` ao importar SHAP.
**Causa:** o primeiro venv usava `--system-site-packages`. O PyMC instalou NumPy 2.4, mas o torch 2.1 e
o OpenCV do Python global foram compilados para NumPy 1.x.
**Correção:** venv totalmente isolado + `requirements.txt` com torch CPU próprio.

### 2. Yeo-Johnson explodindo TENURE

**Sintoma:** silhueta de 0,99 nos dados sintéticos, alta demais para ser verdade.
**Causa:** Yeo-Johnson em `TENURE` (85% dos valores = 12) estimou um λ extremo e gerou valores da ordem
de −1,2 × 10¹⁷. Essa coluna dominou as distâncias e os clusters eram só "TENURE = 12 vs. ≠ 12".
**Correção:** Yeo-Johnson só nas variáveis monetárias e de contagem (o que o enunciado pedia);
`RobustScaler(quantile_range=(5, 95))` para colunas com IQR ≈ 0. Silhueta realista: 0,18 (sintético),
0,25–0,28 (real).

### 3. MLflow 3 recusando salvar modelos sklearn

**Sintoma:** `UntrustedTypesFoundException: ['numpy.dtype']`.
**Causa:** o MLflow 3.18 salva sklearn em `skops` por padrão e exige declarar tipos "confiáveis".
**Correção:** helper `common.log_sklearn_model`, que tenta skops (com `trusted_types`) e cai para
cloudpickle quando há estimadores de terceiros (LightGBM, UMAP).

### 4. MLflow 3 recusando salvar o Autoencoder

**Sintoma:** `Unsupported signature type for the selected serialization format ('pt2')`.
**Causa:** o formato padrão `pt2` (torch.export) exige uma saída tensorial única; o AE devolve
`(reconstrução, z)`. Além disso, o `pt2` fixa o tamanho do batch do exemplo.
**Correção:** registrar o **encoder** (`nn.Sequential` com camadas nativas, que carrega sem o código do
projeto) em formato pickle, com assinatura; o `state_dict` do AE completo vai como artefato.

### 5. `DataFrame.T` no Kaplan-Meier

**Sintoma:** `TypeError: Wrong dtype 'object'` no `KaplanMeierFitter`.
**Causa:** colunas chamadas `T` e `E`; `units.T` é a **transposta** do DataFrame, não a coluna.
**Correção:** renomeadas para `life` / `observed`.

### 6. Requests do MLflow pendurados por 120 s

**Sintoma:** `ReadTimeout` intermitente em `runs/create`; o request nunca aparecia no log do servidor.
**Causa:** 4 workers uvicorn disputando o lock de escrita do SQLite.
**Correção:** `--workers 1`. Os tempos de execução caíram (projeto 4: de 171 s para 49 s nos testes) e o
problema não voltou, nem com dois scripts em paralelo.
**Armadilha extra:** ao matar o servidor à força no Windows, os workers ficaram **órfãos** segurando a
porta 5000, e um teste rodou no servidor antigo sem que isso fosse percebido. Ver o guia do MLflow,
seção "Problemas comuns".

### 7. Cox tempo-variante com C-index 0,36

**Sintoma:** C-index pior que aleatório.
**Causa 1:** o lifelines multiplica a penalidade ElasticNet pelo número de **linhas**. No formato
start/stop há dezenas de intervalos por motor e no máximo 1 evento, então a penalidade ficava ~20×
mais forte que no Cox de snapshots, e os β iam a ~0.
**Correção 1:** `penalizer_ctv = penalizer × eventos / linhas`. C-index foi para 0,85 (sintético) e 0,81
(real).
**Causa 2 (conceitual):** a primeira tentativa de prever RUL com esse modelo usou
S(h | idade, x) = exp(−[H₀(a+h) − H₀(a)]·e^{βx}) com x congelado no último valor. Como as covariáveis
de degradação **crescem** com o tempo, isso subestima o risco futuro (RMSE ~430 no sintético).
**Correção 2:** o modelo passou a ser usado só para inferência (HR) e ordenação de risco. Previsão de RUL
com covariáveis internas exigiria um *joint model*.

### 8. Avisos e depreciações

- `TargetEncoder(random_state=...)` está depreciado no sklearn 1.9. Agora se passa um `StratifiedKFold`
  em `cv` (com fallback para versões antigas).
- PyTensor sem g++ no Windows emitia aviso a cada import. Silenciado com `PYTENSOR_FLAGS=cxx=` (o
  sampling roda pelo nutpie/numba, que não usa o compilador C).
- `SettingWithCopyWarning` no formato start/stop: `.copy()` explícito.

### 9. Linha confusa na chamada do UMAP

Código morto num `if False` ao montar os kwargs do UMAP; reescrito.

### 10. Slug do ccdata (dados reais)

`arjunbhasin2008/ccdata` → 403. O dataset correto é `arjunbhasin2013/ccdata` (encontrado com
`kaggle datasets list -s "credit card dataset for clustering"`).

### 11. Regimes de operação espúrios no C-MAPSS (dados reais)

**Sintoma:** o FD001 (uma única condição de operação) aparecia com 4 regimes.
**Causa:** ruído como −0,0003 arredondado vira `-0.0`, e `str(-0.0) = "-0.0"` ≠ `"0.0"`. A chave do
regime separava zeros positivos e negativos, e a normalização por regime usava grupos arbitrários.
**Correção:** `(valor.round(d) + 0.0)` normaliza −0,0 para 0,0. Agora: FD001+FD003 = 1 regime,
FD002 = 6 regimes (correto).

### 12. Clusterização escolhendo k = 2 (dados reais)

Não é bug: a silhueta favorece k=2, mas a solução só separa "compra" de "saque". Reexecutado com
`--k-min 4` → k=6, com segmentos acionáveis. As duas versões estão no registry (`@champion` e
`@k2_baseline`).

## Organização do MLflow

- **Tag `data_source`** (`real` / `synthetic`) em todos os runs e versões de modelo. Aplicada
  retroativamente aos runs existentes e automática nos próximos (`common.data_source_tag`).
- **Alias `@champion`** na versão de cada modelo treinada com dados reais.
- **Descrição** (nota) nos runs pai reais, apontando para o arquivo de documentação.
- Runs sintéticos e runs que falharam durante os testes **foram mantidos** (nada foi apagado); use o
  filtro `tags.data_source = "real"`.

## Verificação

- Os 4 pipelines rodaram de ponta a ponta com dados sintéticos e reais, sem erro.
- Modelos registrados carregados via `mlflow.pyfunc.load_model("models:/...@champion")`, com predições
  testadas (churn, segmentação, RUL).
- `pyflakes` sem avisos em todo o código.
