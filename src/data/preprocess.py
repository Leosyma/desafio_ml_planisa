# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 11:14:51 2025

@author: Leonardo
"""

# src/data/preprocess.py

import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import os

# Define a pasta
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parents[2]
else:
    BASE_DIR = Path(os.getcwd()).resolve()

# Carrega parametros do modelo
def load_params():
    with open(BASE_DIR / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Cria as features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) custo médio anual
    custo_cols = [f"custo_unitario_trim{i}" for i in range(1, 5)]
    df["custo_medio_ano"] = df[custo_cols].mean(axis=1)

    # 2) volume total anual
    vol_cols = [f"volume_producao_trim{i}" for i in range(1, 5)]
    df["volume_total_ano"] = df[vol_cols].sum(axis=1)

    # 3) variação absoluta de custo T4 - T1
    df["delta_custo_t4_t1"] = df["custo_unitario_trim4"] - df["custo_unitario_trim1"]

    # 4) variação percentual de custo T4 vs T1
    df["delta_custo_pct_t4_t1"] = (
        (df["custo_unitario_trim4"] - df["custo_unitario_trim1"])
        / df["custo_unitario_trim1"].replace(0, 1)
    )

    # 5) volume médio por trimestre
    df["volume_medio_ano"] = df[vol_cols].mean(axis=1)

    # 6) ticket médio aproximado no ano (custo médio * volume médio)
    df["ticket_medio_ano"] = df["custo_medio_ano"] * df["volume_medio_ano"]

    return df

# Pré-processamento do dado de entrada
def preprocess():
    params = load_params()
    cfg_pre = params["preprocess"]

    raw_path = BASE_DIR / "data" / "raw" / "procedimentos_medicos.csv"
    df = pd.read_csv(raw_path)
    
    # Informações do dateset
    print(df.info())
    print(df.isna().sum())
    
    # Tratamento simples de missing
    # Como temos poucos valores anulos iremos dropar ele
    df = df.dropna()

    # Feature engineering
    df = build_features(df)

    # Separa target
    target_col = "target"
    y = df[target_col]
    X = df.drop(columns=[target_col, "centro_custo_id"])

    # One-hot encoding simples (árvores lidam bem com isso)
    X = pd.get_dummies(X, drop_first=True)

    # Separa os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg_pre["test_size"],
        random_state=cfg_pre["random_state"],
        stratify=y,
    )

    # Monta dataframes com a target junto (fica mais fácil depois)
    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Salva os dados
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


if __name__ == "__main__":
    preprocess()
