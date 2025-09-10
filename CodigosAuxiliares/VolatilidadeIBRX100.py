import pandas as pd
import numpy as np

def calcular_volatilidade_anual(csv_path):
    # Ler o CSV (assumindo separador padrão vírgula decimal brasileiro)
    df = pd.read_csv(csv_path)

    # Converter coluna de data
    df['DATA'] = pd.to_datetime(df['DATA'], dayfirst=True, errors='coerce')

    # Corrigir e converter a coluna FECHAMENTO:
    # Ex: "58,642" → 58642.0 (milhares de pontos)
    df['FECHAMENTO'] = df['FECHAMENTO'].astype(str).str.replace('.', ',', regex=False).str.replace(',', '.', regex=False)
    df['FECHAMENTO'] = pd.to_numeric(df['FECHAMENTO'], errors='coerce')

    # Remover linhas inválidas
    df = df.dropna(subset=['DATA', 'FECHAMENTO'])

    # Ordenar por data (antiga → recente)
    df = df.sort_values(by='DATA')

    # Filtrar últimos 12 meses
    data_final = df['DATA'].max()
    data_inicial = data_final - pd.DateOffset(years=1)
    df = df[df['DATA'] >= data_inicial]

    # Calcular retorno logarítmico
    df['RETORNO'] = np.log(df['FECHAMENTO'] / df['FECHAMENTO'].shift(1))
    print(df['FECHAMENTO'], df['FECHAMENTO'].shift(1))
    # Calcular volatilidade anualizada
    vol_anual = df['RETORNO'].std() * np.sqrt(252)

    # Retornar em decimal e em percentual
    return vol_anual, vol_anual * 100

vol_anual, vol_anual2 = calcular_volatilidade_anual("arquivos/IBRX100.csv")

print(vol_anual, vol_anual2)