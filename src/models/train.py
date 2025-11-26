# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:20:36 2025

@author: Leonardo
"""

from pathlib import Path
import os
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# utilidades de MLflow (arquivo src/utils/mlflow_utils.py)
from src.utils.mlflow_utils import (
    setup_mlflow,
    log_model_artifacts,
    log_metrics_dict,
    log_params_dict,
)

# Define a pasta
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parents[2]
else:
    BASE_DIR = Path(os.getcwd()).resolve()

# Carrega parametros do modelo
def load_params():
    params_path = BASE_DIR / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Carrega o dataset de treino após ser processado
def load_train_data():
    train_path = BASE_DIR / "data" / "processed" / "train.csv"
    df = pd.read_csv(train_path)

    if "target" not in df.columns:
        raise ValueError("A coluna 'target' não foi encontrada em data/processed/train.csv")

    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def get_models_and_search_spaces(params: dict):
    """
    Retorna um dicionário com os modelos e seus espaços de busca
    de hiperparâmetros, de acordo com params.yaml.
    """
    models_cfg = params["models"]
    random_state = params["train"]["random_state"]

    models = {}

    # Random Forest
    rf = RandomForestClassifier(random_state=random_state)
    rf_space = {
        "n_estimators": models_cfg["random_forest"]["n_estimators"],
        "max_depth": models_cfg["random_forest"]["max_depth"],
        "min_samples_split": models_cfg["random_forest"]["min_samples_split"],
        "class_weight": models_cfg["random_forest"]["class_weight"],
    }
    models["random_forest"] = (rf, rf_space)

    # XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
    )
    xgb_space = {
        "n_estimators": models_cfg["xgboost"]["n_estimators"],
        "max_depth": models_cfg["xgboost"]["max_depth"],
        "learning_rate": models_cfg["xgboost"]["learning_rate"],
        "subsample": models_cfg["xgboost"]["subsample"],
    }
    models["xgboost"] = (xgb, xgb_space)

    # LightGBM
    lgbm = LGBMClassifier(
        objective="binary",
        random_state=random_state,
    )
    lgbm_space = {
        "n_estimators": models_cfg["lightgbm"]["n_estimators"],
        "max_depth": models_cfg["lightgbm"]["max_depth"],
        "learning_rate": models_cfg["lightgbm"]["learning_rate"],
        "num_leaves": models_cfg["lightgbm"]["num_leaves"],
    }
    models["lightgbm"] = (lgbm, lgbm_space)

    return models


def plot_and_save_confusion_matrix(y_true, y_pred, png_path: Path):
    """Gera e salva a matriz de confusão em PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Train")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def plot_and_save_roc_curve(y_true, y_proba, png_path: Path, label: str = "ROC Curve"):
    """Gera e salva a curva ROC em PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve - Train")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def save_dvc_metrics_and_plots(y_true, y_pred, y_proba, best_model_name: str, metrics: dict):
    """
    Salva os arquivos esperados pelo DVC:
    - metrics.json (na raiz)
    - plots/confusion_matrix.json
    - plots/roc_curve.json
    - reports/figures/confusion_matrix.png
    - reports/figures/roc_curve.png
    """
    # metrics.json
    metrics_out = metrics.copy()
    metrics_out["best_model"] = best_model_name
    metrics_path = BASE_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=4)

    # JSONs para DVC plots
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix JSON: lista de registros com actual/predicted
    cm_df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    cm_json_path = plots_dir / "confusion_matrix.json"
    cm_df.to_json(cm_json_path, orient="records")

    # ROC curve JSON: fpr/tpr
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_json_path = plots_dir / "roc_curve.json"
    roc_df.to_json(roc_json_path, orient="records")

    # PNGs
    figures_dir = BASE_DIR / "reports" / "figures"
    cm_png = figures_dir / "confusion_matrix.png"
    roc_png = figures_dir / "roc_curve.png"

    plot_and_save_confusion_matrix(y_true, y_pred, cm_png)
    plot_and_save_roc_curve(y_true, y_proba, roc_png, label=best_model_name)

    return {
        "metrics_path": metrics_path,
        "cm_json": cm_json_path,
        "roc_json": roc_json_path,
        "cm_png": cm_png,
        "roc_png": roc_png,
    }


def main():
    # Configura MLflow (usa .env ou fallback local)
    setup_mlflow(experiment_name="procedimentos_medicos_classification")

    params = load_params()
    X, y = load_train_data()
    models_dict = get_models_and_search_spaces(params)

    cv_folds = params["train"]["cv_folds"]
    random_state = params["train"]["random_state"]

    best_model = None
    best_model_name = None
    best_cv_score = -np.inf

    # Loop pelos modelos
    for name, (model, search_space) in models_dict.items():
        print(f"🔍 Treinando e otimizando modelo: {name}")

        with mlflow.start_run(run_name=name):
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            )

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=search_space,
                n_iter=20,
                scoring="roc_auc",
                n_jobs=-1,
                cv=cv,
                verbose=1,
                refit=True,
                random_state=random_state,
            )

            search.fit(X, y)

            best_estimator = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_

            # Log de hiperparâmetros e métrica de CV no MLflow
            log_params_dict(best_params)
            log_metrics_dict({"cv_roc_auc": best_score})

            # Atualiza melhor modelo global
            if best_score > best_cv_score:
                best_cv_score = best_score
                best_model = best_estimator
                best_model_name = name

            # Também registra o modelo desse algoritmo no MLflow
            # TENTATIVA de logar o modelo no MLflow
            try:
                log_model_artifacts(best_estimator, model_name=name, artifacts_dir=None)
            except Exception as e:
                print(f"⚠️ Não consegui registrar o modelo '{name}' no MLflow (ignorando erro): {e}")


    if best_model is None:
        raise RuntimeError("Nenhum modelo foi treinado corretamente.")

    print(f"\n✅ Melhor modelo: {best_model_name} | CV ROC-AUC = {best_cv_score:.4f}")

    # Avaliação no próprio conjunto de treino (para gerar arquivos exigidos pelo DVC)
    y_pred = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "cv_roc_auc": float(best_cv_score),
    }

    # Salva arquivos exigidos pelo DVC (metrics + plots)
    outputs = save_dvc_metrics_and_plots(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        best_model_name=best_model_name,
        metrics=metrics,
    )

    # Salva melhor modelo em models/best_model.pkl
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    # Cria um run consolidado no MLflow com os artefatos finais do melhor modelo
    with mlflow.start_run(run_name=f"{best_model_name}_train_summary"):
        log_metrics_dict(metrics)
        mlflow.log_artifact(str(outputs["metrics_path"]))
        mlflow.log_artifact(str(outputs["cm_png"]))
        mlflow.log_artifact(str(outputs["roc_png"]))
        mlflow.log_artifact(str(outputs["cm_json"]))
        mlflow.log_artifact(str(outputs["roc_json"]))
        try:
            log_model_artifacts(best_model, model_name=f"{best_model_name}_final", artifacts_dir=None)
        except Exception as e:
            print(f"⚠️ Não consegui registrar o modelo final no MLflow (ignorando erro): {e}")

        mlflow.set_tag("stage", "train_summary")
        mlflow.set_tag("best_model", best_model_name)

    print("\n🎯 Treinamento concluído com sucesso.")
    print(f"📁 Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()



