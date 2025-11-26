import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from dlsm_lib import LevelSetFuzzyModel, RecursiveLevelSetFuzzyModel

def generate_data_paper_protocol():
    # Total ~302 amostras. 200 Treino (Com Ruído), 102 Teste (Sem Ruído)
    np.random.seed(42)
    n_samples = 302
    
    # 1. Gerar a série PURA (Noise-Free) baseada na Eq (12)
    y_clean = np.zeros(n_samples)
    y_clean[0], y_clean[1] = 0.5, 0.5 
    
    for k in range(2, n_samples):
        term1 = np.cos(y_clean[k-1] / (y_clean[k-2] + 2))
        term2 = 0.8 * np.sin(y_clean[k-2]**2 + y_clean[k-1])
        y_clean[k] = term1 + term2
        
    noise = np.random.normal(0, 0.1, n_samples)
    y_noisy = y_clean + noise
    
    X_clean, targets_clean = [], []
    X_noisy, targets_noisy = [], []
    
    for k in range(2, n_samples):
        # Entradas: y(k-1), y(k-2)
        lags_clean = [y_clean[k-1], y_clean[k-2]]
        lags_noisy = [y_noisy[k-1], y_noisy[k-2]]
        
        X_clean.append(lags_clean)
        targets_clean.append(y_clean[k])
        
        X_noisy.append(lags_noisy)
        targets_noisy.append(y_noisy[k]) # Alvo com ruído
        
    X_clean = np.array(X_clean)
    targets_clean = np.array(targets_clean)
    X_noisy = np.array(X_noisy)
    targets_noisy = np.array(targets_noisy)
    
    train_size = 200
    
    X_train = X_noisy[:train_size]
    y_train = targets_noisy[:train_size]
    
    X_test = X_clean[train_size:]
    y_test = targets_clean[train_size:]
    
    return X_train, y_train, X_test, y_test

def run_dynamic_experiment():
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = generate_data_paper_protocol()
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    print(f"Treino (Com Ruído): {X_train.shape}")
    print(f"Teste (Sem Ruído): {X_test.shape}")

    SIGMA_MULT = 1.5 
    LAMBDA = 0.99 
    ALPHA = 10000.0   
    N_RULES = 6      

    # DLSM
    dlsm = LevelSetFuzzyModel(n_rules=N_RULES)
    dlsm.fit(X_train, y_train)
    
    dlsm.sigmas = dlsm.sigmas * 1.2 
    
    y_pred_dlsm_norm = dlsm.predict(X_test)
    
    y_pred_dlsm = scaler_y.inverse_transform(y_pred_dlsm_norm.reshape(-1, 1)).flatten()
    mse_dlsm = np.mean((y_test_raw - y_pred_dlsm)**2)
    
    # RLSM
    rlsm = RecursiveLevelSetFuzzyModel(n_rules=N_RULES, lam=LAMBDA, alpha=ALPHA)
    
    # Warm Start
    _ = rlsm.fit_recursive(X_train, y_train)
    
    rlsm.sigmas = rlsm.sigmas * 1.2
    
    y_pred_rlsm_norm = []
    
    for k in range(len(X_test)):
        x_k = X_test[k:k+1]      # Input atual (normalizado)
        target_k = y_test[k]     # Alvo real atual (normalizado)
        
        pred = rlsm.predict(x_k)[0]
        y_pred_rlsm_norm.append(pred)
        
        taus = rlsm._compute_membership(x_k)
        d_k = rlsm._construct_matrix_d(taus).flatten()
        d_k_vec = d_k.reshape(1, -1)
        d_k_T = d_k.reshape(-1, 1)
        
        numerator = rlsm.P @ d_k_T @ d_k_vec @ rlsm.P
        denominator = rlsm.lam + d_k_vec @ rlsm.P @ d_k_T
        rlsm.P = (1/rlsm.lam) * (rlsm.P - numerator/denominator)
        
        error = target_k - d_k_vec @ rlsm.u
        gain = rlsm.P @ d_k_T
        rlsm.u = rlsm.u + (gain.flatten() * error)

    y_pred_rlsm_norm = np.array(y_pred_rlsm_norm)
    y_pred_rlsm = scaler_y.inverse_transform(y_pred_rlsm_norm.reshape(-1, 1)).flatten()
    
    mse_rlsm = np.mean((y_test_raw - y_pred_rlsm)**2)

    print("-" * 50)
    print(f"RESULTADOS FINAIS (Escala Real):")
    print(f"MSE DLSM: {mse_dlsm:.5f} (Artigo: ~0.0264)")
    print(f"MSE RLSM: {mse_rlsm:.5f} (Artigo: ~0.0038)")
    print("-" * 50)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_raw, 'k-', lw=2, alpha=0.6, label='Real (Noise-Free)')
    plt.plot(y_pred_dlsm, 'b--', lw=1, label=f'DLSM (Batch) MSE={mse_dlsm:.4f}')
    plt.plot(y_pred_rlsm, 'r-', lw=1.5, label=f'RLSM (Online) MSE={mse_rlsm:.4f}')
    
    plt.title("Revisão Exp 2: Dados Noise-Free no Teste + Normalização")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    save_path = os.path.join('figures', 'exp2_dynamic_system.png')
    plt.savefig(save_path)
    print(f"Figura salva em: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_dynamic_experiment()