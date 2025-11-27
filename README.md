# Relatório Técnico – Pipeline de Classificação de Procedimentos Médicos

## 1. Visão Geral do Projeto

Este projeto implementa um **pipeline completo de Machine Learning** para classificação binária de procedimentos médicos hospitalares, com objetivo de prever se um procedimento **requer atenção especial** (`target = 1`) ou não (`target = 0`).

O pipeline foi construído para demonstrar:

- **Feature engineering estruturado**
- **Comparação e otimização de modelos (Random Forest, XGBoost, LightGBM)**
- **Versionamento de dados e pipeline com DVC**
- **Tracking de experimentos com MLflow**
- **Boas práticas de ML Engineering**: reprodutibilidade, organização modular do código, métricas e gráficos versionados.

### 1.1. Arquitetura do pipeline

Principais componentes de código:

- `src/data/preprocess.py`  
  - EDA (`analise_exploratoria`)  
  - Feature engineering (`build_features`)  
  - Pré-processamento e split (`preprocess`)  

- `src/models/train.py`  
  - Carrega parâmetros e dados processados  
  - Define modelos e espaços de busca  
  - Faz RandomizedSearchCV + validação cruzada estratificada  
  - Escolhe o melhor modelo  
  - Calcula métricas de treino, gera plots e salva `models/best_model.pkl`  

- `src/models/evaluate.py`  
  - Carrega `best_model.pkl` e `data/processed/test.csv`  
  - Calcula métricas em teste  
  - Gera plots e arquivos JSON específicos de teste  

- `src/utils/mlflow_utils.py`  
  - Configuração centralizada do MLflow  
  - Funções utilitárias para log de métricas, parâmetros e artifacts  

- Arquivos de configuração e pipeline:  
  - `params.yaml`  
  - `dvc.yaml` e `dvc.lock`  

Fluxo resumido:

1. `preprocess.py` lê o CSV bruto, faz EDA, trata dados, cria features e faz split em treino e teste.
2. `train.py` treina e ajusta diferentes modelos (RandomForest, XGBoost, LightGBM) usando validação cruzada, escolhe o melhor e salva o modelo final.
3. `evaluate.py` avalia o modelo final no conjunto de teste e gera métricas e gráficos.
4. DVC orquestra as stages (`preprocess`, `train`, `evaluate`) e versiona dados, modelos e métricas.
5. MLflow registra experimentos, parâmetros, métricas e artifacts.

---

## 2. Dataset e Preparação

### 2.1. Estrutura dos dados

O dataset original é `data/raw/procedimentos_medicos.csv`, contendo:

- **Variáveis numéricas**: custos unitários e volumes de produção por trimestre  
  - `custo_unitario_trim1..4`  
  - `volume_producao_trim1..4`  
- **Variáveis categóricas**:  
  - `tipo_unidade` (Cirúrgica, Emergencial, Ambulatorial, etc.)  
  - `regiao` (Norte, Nordeste, Sudeste, Sul)  
  - `especialidade` (Oncologia, Ortopedia, Pediatria, etc.)  
- **Variável alvo**:  
  - `target` (0 ou 1) – indica se o procedimento requer atenção especial  
- **ID**:  
  - `centro_custo_id` (removido na modelagem)

Após o pré-processamento, o dataset é dividido em:

- `data/processed/train.csv` – 3.425 linhas × 32 colunas  
- `data/processed/test.csv` – 857 linhas × 32 colunas  

As colunas incluem:

- Numéricas originais (`custo_*`, `volume_*`)
- Numéricas derivadas (features de engenharia)
- Dummies de variáveis categóricas
- `target` (apenas para treino)

### 2.2. Split de treino e teste

O split é feito em `preprocess.py`, com parâmetros definidos em `params.yaml`:

- `test_size = 0.2` (aprox. 80/20)
- `random_state = 42`
- Split estratificado pela coluna `target`

Isso garante reprodutibilidade e preserva a proporção de classes em treino e teste.

---

## 3. Análise Exploratória dos Dados (EDA)

A EDA é feita na função `analise_exploratoria()` em `preprocess.py`, que:

- Gera estatísticas descritivas (`df.describe()`)
- Mostra tipos de dados e nulos (`df.info()`, `df.isna().sum()`)
- Analisa a distribuição da variável alvo
- Calcula correlação entre variáveis numéricas
- Detecta outliers pelo método IQR
- Salva gráficos e resumos em:

  - `reports/figures/*.png` (distribuições, correlação, missing)  
  - `reports/eda/*.json` (missing_values, outliers_iqr, target_proportions, correlation_matrix)

### 3.1. Distribuição do target

A partir de `target_proportions.json`:

- `target = 1`: ≈ 51,96%  
- `target = 0`: ≈ 48,04%  

Ou seja, o dataset está **praticamente balanceado**. Um gráfico de barras é salvo em `reports/figures/target_distribution.png`.

### 3.2. Valores faltantes

No dataset bruto, o EDA mostra valores faltantes em:

- `custo_unitario_trim2`  
- `custo_unitario_trim3`  
- `volume_producao_trim3`  

O resumo é salvo em `missing_values.json` e um gráfico de barras em `reports/figures/missing_values.png`.

Decisão:  
No pré-processamento, optou-se por **dropar linhas com valores faltantes** (`df = df.dropna()`), simplificando o tratamento e garantindo que o restante do pipeline trabalhe apenas com registros completos.

### 3.3. Outliers

Outliers são detectados por IQR em todas as features numéricas e salvos em `outliers_iqr.json`.

Resumo qualitativo:

- Há um número considerável de outliers em custos unitários por trimestre (dezenas a centenas de registros), o que é coerente com a existência de procedimentos mais complexos e caros.
- Há menos outliers em volumes.

Decisão:

- Manter os outliers, pois:

  - Eles podem representar casos críticos (procedimentos de alto custo).  
  - Modelos de árvores e boosting são relativamente robustos a outliers.

### 3.4. Correlações

A matriz de correlação é salva em `correlation_matrix.json` e visualizada em `reports/figures/correlation_matrix.png`.

Principais insights:

- **Custo vs target**:  
  - Correlação positiva moderada (~0,28–0,30) entre custos unitários trimestrais e `target`.  
  - Procedimentos mais caros tendem a ser classificados como “atenção especial”.

- **Volume vs target**:  
  - Correlação negativa (~−0,24 a −0,25) entre `volume_producao_trim*` e `target`.  
  - Procedimentos de alto volume tendem a ter menos risco.

- **Multicolinearidade entre custos trimestrais**:  
  - Correlações > 0,97 entre `custo_unitario_trim1..4`.  
  - Isso é importante para interpretação e para modelos lineares, mas menos crítico em modelos de árvores/boosting.

---

## 4. Feature Engineering

A lógica de criação de novas features está na função `build_features(df)` em `preprocess.py`.

### 4.1. Features derivadas criadas

Foram criadas pelo menos **6 features derivadas**:

1. **`custo_medio_ano`**  
   Média dos custos unitários trimestrais:

   \[
   custo\_medio\_ano = \frac{custo\_unitario\_trim1 + custo\_unitario\_trim2 + custo\_unitario\_trim3 + custo\_unitario\_trim4}{4}
   \]

2. **`volume_total_ano`**  
   Soma dos volumes dos quatro trimestres:

   \[
   volume\_total\_ano = \sum_{i=1}^4 volume\_producao\_trimi
   \]

3. **`delta_custo_t4_t1`**  
   Diferença absoluta entre custo unitário do 4º e 1º trimestre:

   \[
   delta\_custo\_t4\_t1 = custo\_unitario\_trim4 - custo\_unitario\_trim1
   \]

4. **`delta_custo_pct_t4_t1`**  
   Diferença percentual de custo entre T4 e T1, com proteção para divisão por zero:

   \[
   delta\_custo\_pct\_t4\_t1 = \frac{custo\_unitario\_trim4 - custo\_unitario\_trim1}{custo\_unitario\_trim1}
   \]

5. **`volume_medio_ano`**  
   Média dos volumes trimestrais:

   \[
   volume\_medio\_ano = \frac{\sum_{i=1}^4 volume\_producao\_trimi}{4}
   \]

6. **`ticket_medio_ano`**  
   Aproximação de ticket médio anual:

   \[
   ticket\_medio\_ano = custo\_medio\_ano \times volume\_medio\_ano
   \]

Essas features agregam informações de custo e volume ao longo do ano e capturam:

- Nível médio de custo
- Relevância operacional do procedimento
- Tendência de variação de custo
- Intensidade financeira (custo × volume)

### 4.2. Tratamento de missing, encoding e ID

Na função `preprocess()`:

- São removidas as linhas com `NaN` (`df.dropna()`).
- A coluna `centro_custo_id` é descartada por ser apenas um identificador.
- Variáveis categóricas são transformadas via **One-Hot Encoding** com `pd.get_dummies(drop_first=True)`, gerando:

  - `tipo_unidade_*`
  - `regiao_*`
  - `especialidade_*`

Isso mantém interpretabilidade e funciona bem com modelos de árvores, dado o número reduzido de categorias.

### 4.3. Normalização / padronização

Não foi utilizada normalização ou padronização explícita, pois os modelos principais são baseados em árvores (RandomForest, XGBoost, LightGBM), que não exigem escalonamento de features.

---

## 5. Seleção de Features

A seleção de features foi feita com duas abordagens principais:

1. **Análise de correlação**  
   - Identificação de alta multicolinearidade entre custos trimestrais.  
   - Conclusão: manter tanto as variáveis trimestrais quanto as features derivadas (médias, deltas), já que modelos de árvores lidam bem com colinearidade, e a informação temporal é relevante.

2. **Importância de features nos modelos de árvores**  
   - RandomForest, XGBoost e LightGBM fornecem `feature_importances_`.  
   - Em geral, as features mais importantes tendem a ser:

     - `custo_medio_ano`, `ticket_medio_ano`  
     - `volume_total_ano`  
     - `delta_custo_pct_t4_t1`  
     - Dummies de `tipo_unidade` e `especialidade`

Neste projeto, a seleção de features foi usada mais para **interpretação** do que para poda agressiva de variáveis, priorizando manter o máximo de informação relevante para os modelos.

---

## 6. Modelagem, Validação e Otimização

### 6.1. Algoritmos testados

Em `train.py`, são testados três modelos:

1. **RandomForestClassifier**
2. **XGBClassifier** (XGBoost)
3. **LGBMClassifier** (LightGBM)

Cada modelo tem seu espaço de hiperparâmetros definido em `params.yaml`, incluindo:

- Número de árvores (`n_estimators`)
- Profundidade máxima (`max_depth`)
- Taxa de aprendizado (`learning_rate` – para XGBoost/LightGBM)
- Outros parâmetros como `subsample`, `num_leaves`, etc.

### 6.2. Estratégia de validação

- **Validação cruzada estratificada (StratifiedKFold)**:
  - `n_splits = 5`
  - `shuffle = True`
  - `random_state = 42`

- **RandomizedSearchCV**:
  - `n_iter = 20`
  - `scoring = "roc_auc"`
  - `n_jobs = -1` (paralelismo)
  - `refit = True` (refit no melhor conjunto de hiperparâmetros)

Para cada modelo:

- O melhor conjunto de hiperparâmetros (baseado em ROC-AUC médio na CV) é registrado no MLflow.
- A métrica `cv_roc_auc` é usada para comparação final.

### 6.3. Escolha do modelo final

De acordo com `metrics.json`, o melhor modelo foi o **XGBoost**, com:

- `cv_roc_auc ≈ 0,9489`
- `roc_auc` em treino ≈ 0,9514

O modelo final é salvo em `models/best_model.pkl` e utilizado tanto no script de treinamento quanto no de avaliação.

---

## 7. Avaliação de Resultados

### 7.1. Métricas em Treino

O script `train.py` calcula métricas de treino e salva em `metrics.json`:

- **Accuracy:** 0,93197  
- **Precision:** 0,99870  
- **Recall:** 0,86949  
- **F1:** 0,92963  
- **ROC-AUC:** 0,95144  
- **Best model:** `xgboost`

Além disso, são gerados:

- `plots/confusion_matrix.json` e `plots/roc_curve.json`
- Imagens PNG: `reports/figures/confusion_matrix.png` e `reports/figures/roc_curve.png`

Esses resultados indicam:

- Excelente capacidade discriminativa (AUC ≈ 0,95)
- Precision altíssima para a classe positiva (~0,999)
- Recall também elevado (~0,87), com pouquíssimos falsos positivos em treino.

### 7.2. Métricas em Teste

O script `evaluate.py` calcula métricas em teste e salva em `metrics_test.json`:

- **Accuracy:** 0,91599  
- **Precision:** 0,99204  
- **Recall:** 0,84424  
- **F1:** 0,91220  
- **ROC-AUC:** 0,94654  

Também são gerados:

- `plots/test_confusion_matrix.json` e `plots/test_roc_curve.json`
- Imagens PNG: `reports/figures/confusion_matrix_test.png` e `reports/figures/roc_curve_test.png`

Comparando treino e teste:

- As métricas de teste são levemente inferiores, o que é esperado.
- Não há sinal de overfitting severo.
- O modelo mantém **ROC-AUC próximo de 0,95** no conjunto de teste.

### 7.3. Análise por classe

A partir das matrizes de confusão (treino e teste):

- O número de **falsos positivos** (casos não críticos classificados como críticos) é muito baixo, graças à alta precision.  
- O principal trade-off está em **falsos negativos** (casos críticos classificados como não críticos).  
- Em teste, recall ≈ 0,844 para a classe positiva significa que cerca de 16% dos casos críticos não são identificados.

Dependendo do custo de erro em contexto real:

- Pode ser interessante ajustar o **limiar de decisão** (threshold) para aumentar o recall da classe positiva, mesmo à custa de mais falsos positivos.

---

## 8. Versionamento com DVC

O `dvc.yaml` define as stages principais da pipeline:

### 8.1. Stage `preprocess`

```yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw/procedimentos_medicos.csv
      - src/data/preprocess.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
    params:
      - preprocess.test_size
      - preprocess.random_state
```

- Garante que `train.csv` e `test.csv` sejam gerados de forma reprodutível a partir da base bruta.

### 8.2. Stage `train`

Stage intermediária (resumida):

```yaml
  train:
    cmd: python src/models/train.py
    deps:
      - data/processed/train.csv
      - src/models/train.py
      - src/utils/mlflow_utils.py
    outs:
      - models/best_model.pkl
    metrics:
      - metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.json
      - plots/roc_curve.json
```

- Treina e seleciona o melhor modelo
- Salva `best_model.pkl`  
- Produz `metrics.json` e arquivos de plot em JSON

### 8.3. Stage `evaluate`

```yaml
  evaluate:
    cmd: python src/models/evaluate.py
    deps:
      - data/processed/test.csv
      - models/best_model.pkl
      - src/models/evaluate.py
    metrics:
      - metrics_test.json:
          cache: false
    plots:
      - plots/test_confusion_matrix.json
      - plots/test_roc_curve.json
```

- Avalia o modelo no conjunto de teste
- Salva métricas (`metrics_test.json`) e curvas  

Com isso:

- `dvc repro` executa todo o pipeline (`preprocess` → `train` → `evaluate`).
- `dvc metrics show` e `dvc plots` permitem comparar métricas e curvas entre commits / branches / versões de dados.

---

## 9. Tracking de Experimentos com MLflow

A integração com MLflow está em `src/utils/mlflow_utils.py`.

### 9.1. Configuração

- O tracking URI é definido via variável de ambiente `MLFLOW_TRACKING_URI` ou, por padrão, `file:./experiments/mlruns`.
- O experimento é criado usando `mlflow.set_experiment(experiment_name)`.

Isso permite:

- Usar MLflow localmente
- Ou apontar para o MLflow integrado ao DagsHub (caso configurado)

### 9.2. Logs do `train.py`

Para cada modelo e para o resumo final do melhor modelo, são registrados:

- **Parâmetros** (`mlflow.log_params`):
  - Hiperparâmetros do melhor modelo
  - Tipo de modelo (`random_forest`, `xgboost`, `lightgbm`)

- **Métricas** (`mlflow.log_metrics`):
  - `cv_roc_auc` de cada modelo
  - `accuracy`, `precision`, `recall`, `f1`, `roc_auc` em treino para o melhor modelo

- **Artifacts** (`mlflow.log_artifact` / `mlflow.log_artifacts`):
  - `metrics.json`
  - PNGs de curva ROC e matriz de confusão de treino
  - Arquivo `models/best_model.pkl` (via função utilitária)

### 9.3. Logs do `evaluate.py`

No script de avaliação são registrados:

- Métricas de teste (`metrics_test.json`)
- Imagens de matriz de confusão e curva ROC de teste
- JSONs correspondentes (`test_confusion_matrix.json`, `test_roc_curve.json`)
- Tags indicando que o run é de avaliação de teste

---

## 10. Organização do Código e Boas Práticas

### 10.1. Organização de pastas

Estrutura sugerida do repositório:

```text
data/
  raw/
    procedimentos_medicos.csv
  processed/
    train.csv
    test.csv

models/
  best_model.pkl

reports/
  figures/
    *.png
  eda/
    *.json
  relatorio_tecnico.md

src/
  data/
    preprocess.py
  models/
    train.py
    evaluate.py
  utils/
    mlflow_utils.py

plots/
  *.json

params.yaml
dvc.yaml
dvc.lock
metrics.json
metrics_test.json
README.md
```

### 10.2. Boas práticas adotadas

- **Reprodutibilidade**:
  - `params.yaml` concentra hiperparâmetros e configs de split/CV.
  - DVC controla dados e pipeline.
  - MLflow rastreia experimentos, métricas e artifacts.

- **Modularização**:
  - Separação clara entre EDA, preprocessamento, treinamento, avaliação e logging.

- **Legibilidade**:
  - Uso consistente de nomes de variáveis e funções.
  - Docstrings nas funções principais.
  - Caminhos construídos com `os.path` e `BASE_DIR` para evitar problemas de path relativos.

---

## 11. Trade-offs, Limitações e Melhorias Futuras

### 11.1. Trade-offs

- O modelo final (XGBoost) apresenta **precision muito alta** (~0,99) e **recall elevado** (~0,84) na classe positiva.
- Isso significa:
  - Quase todos os casos marcados como “atenção especial” são realmente críticos.
  - Ainda assim, alguns casos críticos deixam de ser identificados (falsos negativos).

Dependendo do custo de erro no contexto real:

- Pode ser vantajoso ajustar o **threshold** de decisão para aumentar o recall da classe positiva, aceitando mais falsos positivos.

### 11.2. Limitações

- As linhas com missing foram **removidas** em vez de imputadas, o que é simples mas pode descartar informação útil.
- Os dados são **simulados**, sem variáveis clínicas detalhadas (ex.: diagnósticos, idade, comorbidades).
- O log de modelo no MLflow é feito via artifact do `best_model.pkl`, não utilizando ainda o recurso completo de *model registry* do MLflow.

### 11.3. Melhorias futuras

- Explorar **imputação** em vez de dropar linhas (ex.: mediana, KNN Imputer).
- Ajustar o threshold com base em **curvas Precision–Recall** e custos de erro definidos pelo time de negócio.
- Calibrar probabilidades (Platt / isotônica) para decisões mais confiáveis.
- Adicionar explicabilidade com **SHAP** para entender drivers de risco por procedimento.
- Integrar o MLflow com DagsHub (se ainda não estiver) para versionar código, dados e modelos em um único lugar.

---

## 12. Conclusão

O projeto implementa um pipeline de ML completo e reprodutível para classificação de procedimentos médicos, atendendo os requisitos do desafio:

- **Análise exploratória** com estatísticas, distribuições, correlações, missing e outliers.
- **Feature engineering** com diversas features derivadas bem justificadas.
- **Comparação de modelos** (RandomForest, XGBoost, LightGBM) com validação cruzada estratificada e RandomizedSearchCV.
- **Avaliação robusta** com Accuracy, Precision, Recall, F1-Score, ROC-AUC, matrizes de confusão e curvas ROC.
- **Versionamento** de dados e pipeline com DVC.
- **Tracking de experimentos** com MLflow, incluindo parâmetros, métricas e artifacts.
- **Boas práticas de ML Engineering**: organização modular, configuração externa, reprodutibilidade e documentação.

O modelo final (XGBoost) atinge:

- **ROC-AUC ≈ 0,95** em treino e teste  
- **F1 ≈ 0,91** em teste para a classe “atenção especial”  

Isso demonstra um bom equilíbrio entre desempenho preditivo e robustez, além de um pipeline pronto para ser estendido e colocado em produção em um contexto real do setor de saúde.
