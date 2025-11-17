import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os 

# Geração de Dados

def f(x):
    return math.exp(-x/5.0) * math.sin(3*x) + 0.5 * math.sin(x) # Função que eu quero aproximar

def gerar_dados(n_pontos=1000, x_min=0.0, x_max=10.0):   # 1000 pontos entre 0 e 10
    """dados de entrada (xs) e saída (ys) """
    xs = np.linspace(x_min, x_max, n_pontos)
    ys = [f(x) for x in xs]
    return xs, ys

def fp_gaussiana(x, centro, sigma): # Usando uma função de pertinencia gaussiana
  """Função de pertinência gaussiana (Antecedente)."""
  return math.exp(-0.5 * ((x - centro) / sigma) ** 2)

# TSK Tradicional

def criar_regras_tsk(n_regras, x_min, x_max):
  centros = np.linspace(x_min, x_max, n_regras)
  sigma = (x_max - x_min) / (n_regras * 1.5)
  regras = []
  for c in centros:
    p = random.uniform(-0.5, 0.5)
    q = random.uniform(-0.5, 0.5)
    regras.append({'centro': c, 'sigma': sigma, 'p': p, 'q': q})
  return regras

def prever_tsk(x, regras):
  forcas = [fp_gaussiana(x, r['centro'], r['sigma']) for r in regras] # Ativação
  saidas = []

  total_forca = sum(forcas)
  if total_forca == 0:
    return 0, [], []

  ponderado = 0.0
  for i in range(len(regras)):
    # Consequente TSK: p*x + q
    saida = regras[i]['p'] * x + regras[i]['q']
    saidas.append(saida)
    ponderado += forcas[i] * saida

  return ponderado / total_forca, saidas, forcas

def treinar_tsk_gd(regras, xs, ys, taxa_aprendizado=0.01, epocas=1000):
  """Treina parâmetros (p, q) via Gradiente Descendente (GD)."""
  print(f"Iniciando treinamento TSK (GD) por {epocas} épocas...")
  n_regras = len(regras)
  n_dados = len(xs)

  for epoca in range(epocas):
    grad_p = [0.0] * n_regras
    grad_q = [0.0] * n_regras
    erro_quadratico = 0.0

    for x, y in zip(xs, ys):
      y_pred, _, forcas = prever_tsk(x, regras)
      total_forca = sum(forcas)
      if total_forca == 0:
        continue

      erro = y_pred - y
      erro_quadratico += erro * erro
      for i in range(n_regras):
        coef = forcas[i] / total_forca
        grad_p[i] += 2 * erro * coef * x
        grad_q[i] += 2 * erro * coef

    # Atualização
    for i in range(n_regras):
      regras[i]['p'] -= taxa_aprendizado * grad_p[i] / n_dados
      regras[i]['q'] -= taxa_aprendizado * grad_q[i] / n_dados

  final_mse = erro_quadratico / n_dados
  print(f"TSK (GD) EQM final: {final_mse:.6f}")
  return final_mse

# TSK DLSM 

def criar_regras_dlsm(n_regras, x_min, x_max):
  """Inicializa regras DLSM com consequentes v*tau + w."""
  centros = np.linspace(x_min, x_max, n_regras)
  sigma = (x_max - x_min) / (n_regras * 1.5) # Mesmo antecedente
  regras = []
  for c in centros:
    # Parâmetros v e w serão DETERMINADOS pelo 'treinar_dlsm_batch'
    regras.append({'centro': c, 'sigma': sigma, 'v': 0.0, 'w': 0.0})
  return regras

def prever_dlsm(x, regras):
  # ativação de cada regra (tau_i)
  forcas_tau = [fp_gaussiana(x, r['centro'], r['sigma']) for r in regras]

  saidas_m_i = []
  # ponto médio m_i (m_i = v*tau + w)
  for i in range(len(regras)):
    tau_i = forcas_tau[i]
    r = regras[i]
    # Consequente DLSM => m_i = v_i*tau_i + w_i
    m_i = r['v'] * tau_i + r['w']
    saidas_m_i.append(m_i)

  # saída do modelo (y_hat)
  total_tau = sum(forcas_tau)
  if total_tau == 0:
    return 0, [], []

  ponderado = 0.0
  for i in range(len(regras)):
    ponderado += forcas_tau[i] * saidas_m_i[i]

  return ponderado / total_tau, saidas_m_i, forcas_tau

def treinar_dlsm_batch(regras, xs, ys):
  """Treina parâmetros (v, w) via Pseudo-Inversa (Eq. 7-10 do artigo)."""
  print("Iniciando treinamento TSK (DLSM - Batch)...")
  n_regras = len(regras)
  K = len(xs) # Número de pontos (K)
  N = n_regras

  # D é a matriz (K x 2N)
  D = np.zeros((K, 2 * N))
  # y_vec é o vetor de saídas reais 
  y_vec = np.array(ys)

  for k in range(K): # Para cada par (x^k, y^k)
    x = xs[k]
    forcas_tau = [fp_gaussiana(x, r['centro'], r['sigma']) for r in regras]
    s_k = sum(forcas_tau)

    if s_k == 0:
      continue

    # Construir a linha d^k
    row_d_k = []
    for i in range(N):
      tau_i = forcas_tau[i]
      row_d_k.append((tau_i**2) / s_k) # Coeficiente de v_i
      row_d_k.append(tau_i / s_k)   # Coeficiente de w_i

    D[k, :] = row_d_k

  # Resolver u = D+ z (Eq. 9, 10 )
  # u = [v_1, w_1, v_2, w_2, ..., v_N, w_N]^T
  print("Calculando a pseudo-inversa (pode levar um momento)...")
  D_plus = np.linalg.pinv(D)
  u = D_plus @ y_vec

  #  Atualizar as regras com os parâmetros ótimos
  for i in range(N):
    regras[i]['v'] = u[2*i]
    regras[i]['w'] = u[2*i + 1]

  print("Treinamento DLSM (Batch) concluído.")

  # predições e MSE final
  preds_dlsm = [prever_dlsm(x, regras)[0] for x in xs]
  erros = np.array(preds_dlsm) - y_vec
  final_mse = np.mean(erros**2)

  print(f"TSK (DLSM) EQM final: {final_mse:.6f}")
  return final_mse, preds_dlsm

if __name__ == '__main__':
  N_REGRAS = 25
  X_MIN, X_MAX = 0.0, 10.0

  xs, ys = gerar_dados(n_pontos=500, x_min=X_MIN, x_max=X_MAX)

  # TSK Classico 
  start_time_tsk = time.time()
  regras_tsk = criar_regras_tsk(N_REGRAS, X_MIN, X_MAX)
  mse_tsk = treinar_tsk_gd(regras_tsk, xs, ys, taxa_aprendizado=0.02, epocas=1000)
  preds_tsk = [prever_tsk(x, regras_tsk)[0] for x in xs]
  time_tsk = time.time() - start_time_tsk

  print(f"Tempo TSK (GD): {time_tsk:.4f} segundos")
  print("-" * 30)

  #  TSK (DLSM - Batch) 
  start_time_dlsm = time.time()
  regras_dlsm = criar_regras_dlsm(N_REGRAS, X_MIN, X_MAX)
  mse_dlsm, preds_dlsm = treinar_dlsm_batch(regras_dlsm, xs, ys)
  time_dlsm = time.time() - start_time_dlsm

  print(f"Tempo TSK (DLSM): {time_dlsm:.4f} segundos")
  print("-" * 30)

  print("\n--- Comparação Final ---")
  print(f"Número de Regras: {N_REGRAS}")
  print(f"EQM TSK (GD):   {mse_tsk:.8f} (Tempo: {time_tsk:.4f}s)")
  print(f"EQM TSK (DLSM):  {mse_dlsm:.8f} (Tempo: {time_dlsm:.4f}s)")

  OUTPUT_DIR = "output_images"

  plt.figure(figsize=(12, 7))
  plt.plot(xs, ys, label='Função Verdadeira', color='black', linewidth=2.5)
  plt.plot(xs, preds_tsk, '--', label=f'TSK (GD) - EQM={mse_tsk:.6f}', color='blue')
  plt.plot(xs, preds_dlsm, ':', label=f'TSK (DLSM) - EQM={mse_dlsm:.6f}', color='red')
  plt.title('Comparação de Modelos TSK vs TSK (DLSM)')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.grid(True)
  
  plt.savefig(os.path.join(OUTPUT_DIR, 'comparacao_modelos.png'))

  erros_tsk = np.array(preds_tsk) - np.array(ys)
  erros_dlsm = np.array(preds_dlsm) - np.array(ys)

  plt.figure(figsize=(12, 5))
  plt.plot(xs, erros_tsk, label=f'Erro TSK (GD)', color='blue', alpha=0.7)
  plt.plot(xs, erros_dlsm, label=f'Erro TSK (DLSM)', color='red', alpha=0.7)
  plt.title('Erro de Aproximação (Predito - Verdadeiro)')
  plt.xlabel('x')
  plt.ylabel('Erro')
  plt.legend()
  plt.grid(True)

  plt.savefig(os.path.join(OUTPUT_DIR, 'comparacao_erros.png'))

  print(f"\nGRAFICOS SALVOS NA PASTA: '{OUTPUT_DIR}'.")