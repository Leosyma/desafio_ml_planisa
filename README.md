
# Desafio Técnico – Engenharia de Machine Learning

## 📌 Visão Geral

Este repositório contém a solução completa para o **Desafio Técnico – Engenheiro de Machine Learning**, envolvendo:

- Processamento e limpeza de dados com **Pandas**
- Treinamento de modelos com avaliação e otimização via **GridSearchCV**
- Registro de métricas, parâmetros e artefatos no **MLflow**
- Versionamento de dados e modelos via **DVC**
- Armazenamento remoto no **DagsHub**
- Pipeline reprodutível com as etapas:
  **preprocess → train → evaluate**

O objetivo é prever **procedimentos médicos autorizados**, utilizando técnicas clássicas de Machine Learning em um pipeline organizado, versionado e automatizado.

---

## 🏗 Estrutura do Projeto

```
desafio_ml_planisa/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   └── best_model.pkl
│
├── plots/
│   ├── confusion_matrix.json
│   ├── roc_curve.json
│   ├── test_confusion_matrix.json
│   └── test_roc_curve.json
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data/preprocess.py
│   ├── models/train.py
│   ├── models/evaluate.py
│   └── utils/mlflow_utils.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── metrics.json
├── metrics_test.json
├── requirements.txt
└── README.md
```

---

## 🔧 Pipeline do Projeto (DVC)

O pipeline definido em **dvc.yaml** possui três etapas principais:

### **1️⃣ preprocess**
- Realiza tratamento dos dados brutos
- Gera `train.csv`, `val.csv` e `test.csv`

### **2️⃣ train**
- Treina Logistic Regression, Random Forest e XGBoost
- Executa GridSearchCV
- Seleciona o melhor modelo
- Salva `best_model.pkl`
- Gera métricas e gráficos em JSON + PNG
- Registra tudo no **MLflow**

### **3️⃣ evaluate**
- Carrega o modelo final
- Avalia no conjunto de teste
- Salva `metrics_test.json` e plots

---

## 📊 Métricas Obtidas

### **Validação (`metrics.json`)**
- Accuracy: 0.9245
- Precision: 0.9889
- Recall: 0.8643
- F1-score: 0.9224
- ROC-AUC: 0.9524

### **Teste (`metrics_test.json`)**
- Accuracy: 0.9290
- Precision: 0.9912
- Recall: 0.8712
- F1-score: 0.9273
- ROC-AUC: 0.9462

---

## 🚀 Como Executar

### 1. Instalar dependências
```
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
Criar `.env`:
```
MLFLOW_TRACKING_URI=https://dagshub.com/<usuario>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<usuario>
MLFLOW_TRACKING_PASSWORD=<token>
DAGSHUB_USER=<usuario>
DAGSHUB_TOKEN=<token>
```

### 3. Rodar pipeline completo
```
dvc repro
```

### 4. Enviar dados para DagsHub
```
dvc push
git add .
git commit -m "update pipeline"
git push
```

---

## 🛠 Melhorias Futuras

- Criar API (FastAPI)
- Criar Dockerfile + docker-compose
- Adicionar testes unitários
- Interpretabilidade com SHAP
- Monitoramento e retraining automático

---

## 📨 Contato

Fique à vontade para solicitar melhorias ou adaptações do projeto.
