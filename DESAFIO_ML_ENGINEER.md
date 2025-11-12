# Desafio Técnico - Engenheiro de Machine Learning

## 📋 Contexto

Você está se candidatando para uma vaga de Engenheiro de Machine Learning em uma empresa que desenvolve soluções de inteligência artificial para o setor de saúde. O time trabalha com classificação de procedimentos médicos, detecção de anomalias e sistemas de recomendação.

## 🎯 Objetivo do Desafio

Desenvolver um pipeline completo de Machine Learning para classificação binária, demonstrando conhecimento em:
- Feature Engineering
- Seleção e otimização de modelos
- Versionamento de dados e experimentos
- Tracking de experimentos com MLflow
- Boas práticas de ML Engineering

## 📦 Entregáveis

1. **Repositório no DagsHub** com:
   - Código do pipeline de ML
   - Dados versionados usando DVC
   - Experimentos registrados no MLflow
   - README documentando a solução

2. **Relatório técnico** (PDF ou Markdown) contendo:
   - Análise exploratória dos dados
   - Decisões de feature engineering
   - Comparação de modelos testados
   - Análise de resultados e métricas
   - Discussão sobre trade-offs e limitações

## 🗂️ Dataset

Você receberá um dataset simulado de procedimentos médicos hospitalares com as seguintes características:

- **Variáveis numéricas**: custos unitários por trimestre, volumes de produção, indicadores financeiros
- **Variáveis categóricas**: tipo de unidade, especialidade, região
- **Variável target**: classificação binária (0 ou 1) indicando se um procedimento requer atenção especial

O dataset estará disponível em: `data/raw/procedimentos_medicos.csv`

**Estrutura esperada do dataset:**
```python
- centro_custo_id: identificador único
- tipo_unidade: categoria (Cirúrgica, Emergencial, Ambulatorial, etc.)
- custo_unitario_trim1: float
- custo_unitario_trim2: float
- custo_unitario_trim3: float
- custo_unitario_trim4: float
- volume_producao_trim1: int
- volume_producao_trim2: int
- volume_producao_trim3: int
- volume_producao_trim4: int
- regiao: str
- especialidade: str
- target: int (0 ou 1) - variável a ser predita
```

## ✅ Requisitos Técnicos

### 1. Setup do Ambiente

- Criar um repositório no DagsHub
- Configurar DVC para versionamento de dados
- Configurar MLflow para tracking de experimentos
- Criar ambiente virtual (conda ou venv) com `requirements.txt`

### 2. Pipeline de ML

O pipeline deve incluir:

#### 2.1. Análise Exploratória de Dados (EDA)
- Estatísticas descritivas
- Análise de distribuições
- Análise de correlações
- Detecção de valores faltantes e outliers
- Análise de desbalanceamento de classes

#### 2.2. Feature Engineering
- Criar pelo menos 5 features derivadas relevantes (ex: comparações entre trimestres, médias móveis, diferenças percentuais)
- Tratamento de valores faltantes
- Encoding de variáveis categóricas
- Normalização/padronização quando apropriado
- Documentar a lógica de cada feature criada

#### 2.3. Seleção de Features
- Aplicar pelo menos uma técnica de seleção de features (ex: mutual information, feature importance, correlação)
- Justificar a escolha das features selecionadas

#### 2.4. Modelagem
Testar e comparar pelo menos **3 algoritmos diferentes**:
- Random Forest
- XGBoost ou LightGBM
- Um terceiro algoritmo à sua escolha (ex: Logistic Regression, SVM, Neural Network)

Para cada algoritmo:
- Usar validação cruzada (k-fold, preferencialmente stratified)
- Otimizar hiperparâmetros usando GridSearchCV ou RandomizedSearchCV
- Registrar todas as execuções no MLflow

#### 2.5. Avaliação
- Calcular métricas relevantes: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Gerar matriz de confusão
- Plotar curva ROC
- Analisar métricas por classe (considerando possível desbalanceamento)
- Usar conjunto de teste separado (não usado durante treinamento/validação)

### 3. Versionamento com DVC

- Versionar o dataset original
- Versionar datasets processados (train/test splits)
- Criar pipeline DVC (`dvc.yaml`) que execute:
  - Pré-processamento
  - Treinamento
  - Avaliação

### 4. Tracking com MLflow

Cada experimento deve registrar no MLflow:
- **Parâmetros**: hiperparâmetros do modelo, parâmetros de pré-processamento
- **Métricas**: todas as métricas de avaliação
- **Artifacts**: 
  - Modelo treinado (pickle ou MLflow format)
  - Gráficos (matriz de confusão, curva ROC, feature importance)
  - Relatório de métricas
- **Tags**: algoritmo usado, versão do dataset, descrição do experimento

### 5. Código

- Organizar código em módulos/funções reutilizáveis
- Seguir boas práticas Python (PEP 8)
- Adicionar docstrings nas funções principais
- Criar um script principal (`train.py` ou `main.py`) que execute o pipeline completo

## 📊 Critérios de Avaliação

### Conhecimento Técnico (40%)
- Qualidade do feature engineering
- Escolha adequada de algoritmos e hiperparâmetros
- Uso correto de métricas e validação
- Tratamento adequado de problemas comuns (desbalanceamento, overfitting, etc.)

### Ferramentas e Boas Práticas (30%)
- Uso correto do DagsHub/DVC para versionamento
- Implementação adequada do MLflow
- Organização e estrutura do código
- Documentação clara

### Resultados e Análise (20%)
- Performance dos modelos
- Qualidade da análise e interpretação dos resultados
- Discussão de trade-offs e limitações

### Criatividade e Inovação (10%)
- Features criativas e bem justificadas
- Abordagens interessantes para resolver problemas
- Melhorias além do básico

## 🚀 Como Entregar

1. **Fork ou clone** o repositório base (se fornecido) ou crie um novo repositório no DagsHub
2. **Desenvolva** a solução seguindo os requisitos
3. **Compartilhe** o link do repositório DagsHub
4. **Envie** o relatório técnico (PDF ou link para arquivo Markdown no repositório)

**Prazo sugerido**: 5-7 dias

## 📝 Observações Importantes

- **Não é necessário** criar APIs ou interfaces web - foque apenas no pipeline de ML
- Use dados sintéticos ou públicos se não tiver acesso ao dataset real
- Documente todas as decisões técnicas
- Seja criativo, mas mantenha o foco na qualidade técnica
- O código deve ser executável e reproduzível

## 🔧 Recursos Úteis

- [DagsHub Documentation](https://dagshub.com/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## ❓ Dúvidas?

Em caso de dúvidas sobre o desafio, entre em contato com o time de recrutamento.

---

**Boa sorte! 🎯**

