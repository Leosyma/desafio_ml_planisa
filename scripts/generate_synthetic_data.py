"""
Script para gerar dataset sintético de procedimentos médicos.
Este script cria dados realistas para o desafio técnico.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Gera dataset sintético de procedimentos médicos hospitalares.
    
    Args:
        n_samples: Número de amostras a gerar
        random_state: Seed para reprodutibilidade
    
    Returns:
        DataFrame com dados sintéticos
    """
    np.random.seed(random_state)
    
    # Tipos de unidades
    tipos_unidade = [
        'Cirúrgica', 'Emergencial', 'Ambulatorial', 
        'Internação', 'SADT Imagem', 'SADT Laboratorial', 'SADT Outros'
    ]
    
    # Regiões
    regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']
    
    # Especialidades
    especialidades = [
        'Cardiologia', 'Ortopedia', 'Pediatria', 'Clínica Médica',
        'Cirurgia Geral', 'Neurologia', 'Oncologia', 'Traumatologia'
    ]
    
    data = {
        'centro_custo_id': [f'CC_{i:06d}' for i in range(1, n_samples + 1)],
        'tipo_unidade': np.random.choice(tipos_unidade, n_samples),
        'regiao': np.random.choice(regioes, n_samples),
        'especialidade': np.random.choice(especialidades, n_samples),
    }
    
    # Gerar custos unitários por trimestre (com correlação entre trimestres)
    base_cost = np.random.lognormal(mean=6, sigma=1.5, size=n_samples)
    
    # Trimestre 1 (base)
    data['custo_unitario_trim1'] = base_cost
    
    # Trimestres seguintes com variação
    for trim in [2, 3, 4]:
        variation = np.random.normal(1.0, 0.15, n_samples)  # Variação de ±15%
        data[f'custo_unitario_trim{trim}'] = base_cost * variation
    
    # Volumes de produção (correlacionados com custos)
    base_volume = np.random.poisson(lam=100, size=n_samples)
    
    for trim in [1, 2, 3, 4]:
        variation = np.random.normal(1.0, 0.2, n_samples)
        data[f'volume_producao_trim{trim}'] = np.maximum(
            1, (base_volume * variation).astype(int)
        )
    
    # Criar variável target com lógica realista
    # Target = 1 se houver anomalias em custos ou volumes
    df = pd.DataFrame(data)
    
    # Calcular indicadores que podem indicar problemas
    df['custo_medio'] = df[[f'custo_unitario_trim{i}' for i in [1,2,3,4]]].mean(axis=1)
    df['custo_std'] = df[[f'custo_unitario_trim{i}' for i in [1,2,3,4]]].std(axis=1)
    df['volume_medio'] = df[[f'volume_producao_trim{i}' for i in [1,2,3,4]]].mean(axis=1)
    
    # Criar target: 1 se custo muito alto, muito variável, ou volume muito baixo
    high_cost = df['custo_medio'] > df['custo_medio'].quantile(0.75)
    high_variance = df['custo_std'] > df['custo_std'].quantile(0.75)
    low_volume = df['volume_medio'] < df['volume_medio'].quantile(0.25)
    
    # Adicionar algum ruído
    noise = np.random.random(n_samples) < 0.1
    
    df['target'] = ((high_cost | high_variance | low_volume) | noise).astype(int)
    
    # Remover colunas auxiliares
    df = df.drop(columns=['custo_medio', 'custo_std', 'volume_medio'])
    
    # Adicionar alguns valores faltantes (5% dos dados)
    missing_cols = ['custo_unitario_trim2', 'custo_unitario_trim3', 'volume_producao_trim3']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Reordenar colunas
    cols_order = [
        'centro_custo_id', 'tipo_unidade', 'regiao', 'especialidade',
        'custo_unitario_trim1', 'custo_unitario_trim2', 
        'custo_unitario_trim3', 'custo_unitario_trim4',
        'volume_producao_trim1', 'volume_producao_trim2',
        'volume_producao_trim3', 'volume_producao_trim4',
        'target'
    ]
    
    return df[cols_order]


def main():
    """Função principal para gerar e salvar dados."""
    # Criar diretório se não existir
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Gerar dados
    print("Gerando dataset sintético...")
    df = generate_synthetic_data(n_samples=5000)
    
    # Salvar
    output_path = data_dir / 'procedimentos_medicos.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset gerado com sucesso!")
    print(f"Salvo em: {output_path}")
    print(f"\nEstatísticas:")
    print(f"  - Total de amostras: {len(df)}")
    print(f"  - Classes target: {df['target'].value_counts().to_dict()}")
    print(f"  - Valores faltantes: {df.isnull().sum().sum()}")
    print(f"\nPrimeiras linhas:")
    print(df.head())


if __name__ == '__main__':
    main()

