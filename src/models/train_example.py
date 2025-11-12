"""
Exemplo de script de treinamento usando MLflow.
Este é apenas um exemplo - o candidato deve implementar sua própria solução.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

from src.utils.mlflow_utils import setup_mlflow, log_model_artifacts, log_metrics_dict, log_params_dict


def load_data():
    """Carrega dados processados."""
    train_path = Path('data/processed/train.csv')
    return pd.read_csv(train_path)


def create_artifacts_dir():
    """Cria diretório para artifacts."""
    artifacts_dir = Path('reports/figures')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plota e salva matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path):
    """Plota e salva curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Treina Random Forest com otimização de hiperparâmetros.
    
    Este é apenas um exemplo básico. O candidato deve:
    - Implementar feature engineering adequado
    - Testar múltiplos algoritmos
    - Fazer validação cruzada adequada
    - Registrar tudo no MLflow
    """
    
    # Configurar MLflow
    setup_mlflow()
    
    with mlflow.start_run(run_name="random_forest_baseline"):
        
        # Parâmetros para busca
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced']
        }
        
        # Modelo base
        base_model = RandomForestClassifier(random_state=42)
        
        # Validação cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Busca de hiperparâmetros
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=20,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            random_state=42
        )
        
        print("🔍 Buscando melhores hiperparâmetros...")
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        
        # Predições
        y_pred = best_model.predict(X_val)
        y_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_val, y_proba)
        }
        
        # Registrar no MLflow
        log_params_dict(random_search.best_params_)
        log_metrics_dict(metrics)
        
        # Criar artifacts
        artifacts_dir = create_artifacts_dir()
        plot_confusion_matrix(y_val, y_pred, artifacts_dir / 'confusion_matrix.png')
        plot_roc_curve(y_val, y_proba, artifacts_dir / 'roc_curve.png')
        
        # Registrar artifacts
        log_model_artifacts(best_model, 'random_forest', str(artifacts_dir))
        
        # Salvar modelo
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        with open(models_dir / 'best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        print("\n✅ Treinamento concluído!")
        print(f"📊 Melhor F1-Score: {metrics['f1_score']:.4f}")
        print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return best_model, metrics


def main():
    """Função principal."""
    print("🚀 Iniciando treinamento...")
    
    # Carregar dados
    print("\n📂 Carregando dados...")
    df = load_data()
    
    # Separar features e target
    # NOTA: O candidato deve implementar feature engineering adequado antes disso
    X = df.drop(columns=['target'])  # Ajustar conforme necessário
    y = df['target']
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  - Treino: {len(X_train)} amostras")
    print(f"  - Validação: {len(X_val)} amostras")
    
    # Treinar modelo
    model, metrics = train_random_forest(X_train, y_train, X_val, y_val)
    
    print("\n✅ Pipeline concluído!")


if __name__ == '__main__':
    main()

