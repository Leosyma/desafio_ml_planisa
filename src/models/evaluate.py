# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:24:28 2025

@author: Leonardo
"""

# -*- coding: utf-8 -*-
"""
Avaliação do modelo em conjunto de teste.

- Carrega data/processed/test.csv (gerado em src/data/preprocess.py)
- Carrega models/best_model.pkl (gerado em src/models/train.py)
- Calcula métricas no conjunto de teste
- Gera arquivos esperados pelo DVC:
    - metrics_test.json
    - plots/test_confusion_matrix.json
    - plots/test_roc_curve.json
- Registra métricas e artefatos no MLflow
"""

from pathlib import Path
import os
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from src.utils.mlflow_utils import setup_mlflow, log_metrics_dict


# -------------------------------------------------------------------
# BASE_DIR compatível com script e notebook
# -------------------------------------------------------------------
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parents[2]
else:
    BASE_DIR = Path(os.getcwd()).resolve()


def load_test_data():
    """Carrega o conjunto de teste processado."""
    test_path = BASE_DIR / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_path}")

    df_test = pd.read_csv(test_path)

    if "target" not in df_test.columns:
        raise ValueError("A coluna 'target' não foi encontrada em data/processed/test.csv")

    y_test = df_test["target"]
    X_test = df_test.drop(columns=["target"])
    return X_test, y_test


def load_best_model():
    """Carrega o modelo salvo em models/best_model.pkl."""
    model_path = BASE_DIR / "models" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    model = joblib.load(model_path)
    return model


def plot_and_save_confusion_matrix(y_true, y_pred, png_path: Path):
    """Gera e salva matriz de confusão em PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Test")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def plot_and_save_roc_curve(y_true, y_proba, png_path: Path, label: str = "Test ROC"):
    """Gera e salva curva ROC em PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve - Test")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def save_dvc_test_outputs(y_true, y_pred, y_proba, metrics_test: dict):
    """
    Salva arquivos esperados pelo estágio 'evaluate' no dvc.yaml:
      - metrics_test.json
      - plots/test_confusion_matrix.json
      - plots/test_roc_curve.json
    E também PNGs em reports/figures para uso no relatório / MLflow.
    """
    # metrics_test.json (na raiz, conforme dvc.yaml)
    metrics_path = BASE_DIR / "metrics_test.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=4)

    # Diretório de plots (JSONs para DVC)
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix JSON
    cm_df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    cm_json_path = plots_dir / "test_confusion_matrix.json"
    cm_df.to_json(cm_json_path, orient="records")

    # ROC curve JSON
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_json_path = plots_dir / "test_roc_curve.json"
    roc_df.to_json(roc_json_path, orient="records")

    # PNGs em reports/figures (opcional para DVC, útil para MLflow e relatório)
    figures_dir = BASE_DIR / "reports" / "figures"
    cm_png = figures_dir / "confusion_matrix_test.png"
    roc_png = figures_dir / "roc_curve_test.png"

    plot_and_save_confusion_matrix(y_true, y_pred, cm_png)
    plot_and_save_roc_curve(y_true, y_proba, roc_png, label="Test ROC")

    return {
        "metrics_path": metrics_path,
        "cm_json": cm_json_path,
        "roc_json": roc_json_path,
        "cm_png": cm_png,
        "roc_png": roc_png,
    }


def main():
    # Configura MLflow (usa MLFLOW_TRACKING_URI do .env ou fallback local)
    setup_mlflow(experiment_name="procedimentos_medicos_classification")

    # Carrega dados e modelo
    X_test, y_test = load_test_data()
    model = load_best_model()

    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas no teste
    metrics_test = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # Salvar arquivos que o DVC espera (metrics + plots JSON/PNG)
    outputs = save_dvc_test_outputs(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics_test=metrics_test,
    )

    # Registrar no MLflow em um run separado
    with mlflow.start_run(run_name="test_evaluation"):
        log_metrics_dict(metrics_test)
        # Artefatos úteis
        mlflow.log_artifact(str(outputs["metrics_path"]))
        mlflow.log_artifact(str(outputs["cm_png"]))
        mlflow.log_artifact(str(outputs["roc_png"]))
        mlflow.log_artifact(str(outputs["cm_json"]))
        mlflow.log_artifact(str(outputs["roc_json"]))
        mlflow.set_tag("stage", "test_evaluation")

    print("✅ Avaliação no conjunto de teste concluída.")
    print("📊 Métricas de teste:", metrics_test)


if __name__ == "__main__":
    main()
