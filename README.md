# 🚀 Desafio Técnico - Engenheiro de Machine Learning

Este repositório contém o desafio técnico e template para candidatos à vaga de **Engenheiro de Machine Learning**.

## 📋 Sobre o Desafio

O desafio consiste em desenvolver um pipeline completo de Machine Learning para classificação binária, demonstrando conhecimento em:

- Feature Engineering
- Seleção e otimização de modelos
- Versionamento de dados e experimentos (DVC)
- Tracking de experimentos (MLflow)
- Boas práticas de ML Engineering

## 🎯 Objetivos

- Testar pelo menos **3 algoritmos diferentes**
- Usar **DagsHub** para versionamento de dados (DVC)
- Registrar experimentos no **MLflow**
- Criar um relatório técnico completo

## 📦 Estrutura do Repositório

```
.
├── index.html                 # Página principal (GitHub Pages)
├── assets/                    # CSS e JavaScript
├── DESAFIO_ML_ENGINEER.md     # Documento completo do desafio
├── scripts/
│   └── generate_synthetic_data.py  # Gerador de dataset sintético
├── src/
│   ├── data/                   # Carregamento e pré-processamento
│   ├── features/               # Feature engineering
│   ├── models/                 # Treinamento e avaliação
│   └── utils/                  # Utilitários (MLflow, etc.)
├── dvc.yaml                    # Pipeline DVC
├── params.yaml                 # Parâmetros do pipeline
├── requirements.txt            # Dependências Python
└── setup_dagshub.py           # Script de configuração
```

## 🚀 Como Usar Este Template

1. **Clone ou fork este repositório**
   ```bash
   git clone https://github.com/planisa/desafio_ml.git
   cd desafio_ml
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

3. **Gere o dataset sintético**
   ```bash
   python scripts/generate_synthetic_data.py
   ```

4. **Configure o DagsHub**
   ```bash
   python setup_dagshub.py
   ```

5. **Execute o pipeline**
   ```bash
   dvc repro
   ```

## 📖 Documentação

- **[Desafio Completo](DESAFIO_ML_ENGINEER.md)** - Todos os requisitos e detalhes

## ❓ Dúvidas?

Em caso de dúvidas sobre o desafio, entre em contato com o time de recrutamento.

---

**Boa sorte! 🎯**
