"""
Utilitários para configuração e uso do MLflow.
"""

import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()


def setup_mlflow(experiment_name="procedimentos_medicos_classification"):
    """
    Configura o MLflow para usar DagsHub como backend.
    
    Args:
        experiment_name: Nome do experimento no MLflow
    """
    # Tentar obter URI do MLflow do .env, senão usar local
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./experiments/mlruns')
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow configurado: {mlflow_uri}")
    print(f"📊 Experimento: {experiment_name}")


def log_model_artifacts(model, model_name, artifacts_dir=None):
    """
    Registra modelo e artifacts no MLflow.
    
    Args:
        model: Modelo treinado
        model_name: Nome do modelo
        artifacts_dir: Diretório com artifacts (gráficos, etc.)
    """
    # Registrar modelo
    mlflow.log_artifact("models/best_model.pkl", artifact_path="models")
    
    # Registrar artifacts se fornecidos
    if artifacts_dir and Path(artifacts_dir).exists():
        mlflow.log_artifacts(artifacts_dir, artifact_path="artifacts")
        print(f"📦 Artifacts registrados de: {artifacts_dir}")


def log_metrics_dict(metrics_dict):
    """
    Registra múltiplas métricas no MLflow.
    
    Args:
        metrics_dict: Dicionário com nome_metrica: valor
    """
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(metric_name, metric_value)
    
    print(f"📈 {len(metrics_dict)} métricas registradas")


def log_params_dict(params_dict):
    """
    Registra múltiplos parâmetros no MLflow.
    
    Args:
        params_dict: Dicionário com nome_parametro: valor
    """
    for param_name, param_value in params_dict.items():
        mlflow.log_param(param_name, str(param_value))
    
    print(f"⚙️  {len(params_dict)} parâmetros registrados")

