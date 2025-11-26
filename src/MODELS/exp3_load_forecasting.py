import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dlsm_lib import RecursiveLevelSetFuzzyModel

def get_real_load_data(filename='CURVA_CARGA_2000.csv'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"O arquivo {filename} não foi encontrado no diretório.")

    df = pd.read_csv(filename, sep=';')

    df['din_instante'] = pd.to_datetime(df['din_instante'])

    df_sudeste = df[df['nom_subsistema'] == 'SUDESTE'].copy()

    start_date = '2000-08-01'
    end_date = '2000-09-01' # Exclusivo
    mask = (df_sudeste['din_instante'] >= start_date) & (df_sudeste['din_instante'] < end_date)
    df_august = df_sudeste.loc[mask].sort_values('din_instante')

    if len(df_august) == 0:
        raise ValueError("Nenhum dado encontrado para o período/região especificados.")
    
    print(f"Dados carregados: {len(df_august)} amostras (horas) encontradas para SUDESTE em Agosto/2000.")

    load = df_august['val_cargaenergiahomwmed'].values

    load_min = load.min()
    load_max = load.max()
    load_norm = (load - load_min) / (load_max - load_min) # Entre 0 e 1
    load_final = load_norm * 0.8 + 0.1 # Ajuste para intervalo 0.1 - 0.9 (estético/numérico)

    return load_final

def create_lags(data, lags=2):
    """
    Entrada: y(k-1), y(k-2) -> Saída: y(k)
    """
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i][::-1]) 
        y.append(data[i])
    return np.array(X), np.array(y)

def run_load_forecast():
    try:
        data = get_real_load_data('CURVA_CARGA_2000.csv')
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # Lags = 2 (f(y[k-1], y[k-2]))
    X, y = create_lags(data, lags=2)
    
    rlsm = RecursiveLevelSetFuzzyModel(n_rules=6, lam=0.85, alpha=1000)
    
    preds_rlsm = rlsm.fit_recursive(X, y)
    
    rmse = np.sqrt(np.mean((y - preds_rlsm.flatten())**2))
    
    print("--- Experimento Previsão de Carga (Replicado com Dados Reais) ---")
    print(f"Dataset: ONS Carga Global - Subsistema Sudeste (Ago/2000)")
    print(f"RMSE RLSM (Online): {rmse:.4f}")
    
    hours_to_plot = 72
    
    plt.figure(figsize=(12, 6))
    plt.plot(y[:hours_to_plot], 'k-', linewidth=1.5, label='Carga Real (Normalizada)')
    plt.plot(preds_rlsm[:hours_to_plot], 'r-', linewidth=1.5, alpha=0.8, label='Previsão RLSM')
    
    plt.title("Previsão de Carga Horária - RLSM (Primeiras 72h de Agosto/2000)")
    plt.xlabel("Hora")
    plt.ylabel("Potência Normalizada")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    save_path = os.path.join('figures', 'exp3_load_forecast_real.png')
    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_load_forecast()