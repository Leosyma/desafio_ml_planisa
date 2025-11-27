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
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

# Analise Exploratoria
def analise_exploratoria():
    raw_path = BASE_DIR / "data" / "raw" / "procedimentos_medicos.csv"
    df = pd.read_csv(raw_path)
    
    # Informações do dateset
    print(df.info())
    
    # Valores Ausentes
    print(df.isna().sum())
    
    # Contagem da variavel target
    print(df['target'].value_counts())
    
    # Gráfico de distribuição da target
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Distribuição da variável target")
    plt.xlabel("target")
    plt.ylabel("Contagem")
    fig_path = "reports/figures/target_distribution.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Distribuição da target salva em {fig_path}")

    # Tabela de proporções
    class_counts = df["target"].value_counts(normalize=True)
    prop_path = Path("reports/eda/target_proportions.json")
    prop_path.write_text(class_counts.to_json(), encoding="utf-8")
    print(f"📊 Proporção das classes salva em {prop_path}")
    
    # Histograma para colunas numéricas
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not num_cols:
        print("⚠️ Nenhuma coluna numérica encontrada para plotar.")
        return

    for col in num_cols:
        if col == "target":
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribuição de {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        fig_path = f"reports/figures/dist_{col}.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"📊 Distribuição de {col} salva em {fig_path}")
        
    # Correlação das variáveis numéricas
    num_df = df.select_dtypes(include=["int64", "float64"])
    if num_df.shape[1] < 2:
        print("⚠️ Menos de 2 colunas numéricas. Pulando correlação.")
        return

    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="viridis")
    plt.title("Matriz de correlação (variáveis numéricas)")
    fig_path = "reports/figures/correlation_matrix.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Matriz de correlação salva em {fig_path}")

    corr_path = Path("reports/eda/correlation_matrix.json")
    corr_path.write_text(corr.to_json(), encoding="utf-8")
    print(f"📊 Matriz de correlação salva em {corr_path}")
    
    # Plota valores nulos
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("✅ Nenhum valor nulo encontrado.")
        return

    plt.figure(figsize=(8, 4))
    missing.sort_values(ascending=False).plot(kind="bar")
    plt.title("Valores nulos por coluna")
    plt.ylabel("Quantidade de nulos")
    plt.xlabel("Coluna")
    fig_path = "reports/figures/missing_values.png"
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Gráfico de valores nulos salvo em {fig_path}")

    missing_path = Path("reports/eda/missing_values.json")
    missing_path.write_text(missing.to_json(), encoding="utf-8")
    print(f"📊 Resumo de nulos salvo em {missing_path}")
    
    # Outliers
    num_df = df.select_dtypes(include=["int64", "float64"])
    outlier_counts = {}

    for col in num_df.columns:
        Q1 = num_df[col].quantile(0.25)
        Q3 = num_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (num_df[col] < lower) | (num_df[col] > upper)
        outlier_counts[col] = int(mask.sum())

    outliers_path = Path("reports/eda/outliers_iqr.json")
    outliers_path.write_text(json.dumps(outlier_counts, indent=2), encoding="utf-8")
    print(f"📊 Contagem de outliers (IQR) salva em {outliers_path}")
    

# Pré-processamento do dado de entrada
def preprocess():
    params = load_params()
    cfg_pre = params["preprocess"]

    raw_path = BASE_DIR / "data" / "raw" / "procedimentos_medicos.csv"
    df = pd.read_csv(raw_path)
     
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
