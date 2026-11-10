# 07 · Limitações e próximos passos

## O que os resultados **não** sustentam

| Projeto | Cuidado |
|---|---|
| 1 | Silhueta de 0,25: os segmentos são regiões de um contínuo, não grupos isolados. Os segmentos 4 (inativo) e 5 (rotativo misto) têm fronteiras fracas. Não há validação externa (inadimplência, receita) de que os segmentos diferem em resultado de negócio |
| 2 | O ganho do LightGBM sobre a regressão logística é pequeno (+0,009 de PR-AUC). A economia de 40,9 mil depende de custos **assumidos** (500/100) e supõe que toda oferta retém o cliente |
| 3 | A divisão do BRCA em dois grupos no UMAP é hipótese (possíveis subtipos), sem anotação para confirmar. UMAP não preserva distâncias globais: a posição relativa entre grupos não tem significado |
| 4 | O RUL é **superestimado** em todas as faixas (viés +13 a +28 ciclos), o lado perigoso para manutenção. A censura do treino é induzida. O teste de Schoenfeld rejeita proporcionalidade nas tendências dos sensores. O Bayesiano tem 7 divergências e σ_a mal identificado (só 2 frotas) |

## Próximos passos sugeridos

### 1 · Clusterização
- Validar os segmentos contra um resultado externo (ex.: inadimplência futura), se houver.
- Testar HDBSCAN (densidade, permite "ruído") e K-Prototypes/GMM em subespaços.
- Atribuição suave: usar as probabilidades do GMM para os clientes de fronteira (segmentos 4 e 5).

### 2 · Churn
- Substituir os custos fixos por valor do cliente (CLV) e taxa real de aceitação da oferta.
- Avaliar a regressão logística com interações como modelo de produção (explicável, quase o mesmo
  desempenho).
- Uplift modeling: prever quem **muda de ideia** com a oferta, não só quem vai sair.
- Validação temporal, se houver data de referência.

### 3 · Genômica
- Repetir com a lista completa de 20 mil genes no Autoencoder (hoje: top 5.000).
- Variational Autoencoder (latente probabilístico) e comparação com t-SNE.
- Cruzar com anotação de subtipo PAM50 do TCGA-BRCA para testar a hipótese dos dois grupos.

### 4 · Sobrevivência
- **Previsão conservadora de RUL:** reportar o quantil 10–20% da distribuição de vida residual em vez da
  mediana, e medir a taxa de "falha antes da manutenção".
- Corrigir o viés otimista: recalibrar S(t) (ex.: regressão isotônica sobre o tempo previsto) ou
  modelar a forma do risco com splines (Royston-Parmar).
- Rodar com FD001–FD004 (4 frotas, 6 regimes) para dar ao modelo hierárquico mais grupos.
- Cox estratificado por faixa de idade ou com interações covariável × tempo (violação de PH).
- *Joint model* longitudinal-sobrevivência para usar o Cox tempo-variante na previsão.
- Comparar com regressão direta do RUL limitado a 125 ciclos (padrão da literatura) para ter uma
  referência comparável.

### Engenharia / MLOps
- Trocar SQLite por PostgreSQL no backend do MLflow (permite vários workers e acesso concorrente).
- "Models from code" do MLflow 3 em vez de pickle, para modelos auditáveis.
- Testes automatizados (pytest) rodando o modo `--quick` com dados sintéticos em CI.
- Fixar as versões em `requirements.txt` (hoje com `>=`), para reprodutibilidade exata.
