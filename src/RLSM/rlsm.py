import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
import os  

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans') # Não ter warning

def load_and_preprocess_adult(path):
    """
    Carrega e pré-processa o Adult Dataset.
    O pré-processamento é crucial para modelos fuzzy,
    pois as funções de pertinência dependem de distâncias.
    """
    print(f"Carregando dados de '{path}'...")
    
    # Colunas do Adult dataset
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # ? => Nan
    data = pd.read_csv(
        path,
        names=columns,
        na_values=' ?',
        skipinitialspace=True
    )
    
    # Limpar Dados
    data.dropna(inplace=True)
    
    # Mapear o Alvo (Target) para 0 e 1
    # Esta é a adaptação de Regressão para Classificação
    data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
    
    # Encoding Categórico (One-Hot Encoding)
    categorical_cols = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Normalização de Features
    # Separar alvo das features
    target = data.pop('income').values
    
    feature_names = data.columns.tolist()
    
    # Normalizar features para o intervalo [0, 1]; Melhor p/ Gaussianas !!!!!!
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data)
    
    print(f"Pré-processamento concluído.")
    print(f"Forma dos dados (Amostras, Features): {X_scaled.shape}")
    
    return X_scaled, target, feature_names

class RLSM:
    """
    Implementa o Recursive Level Set Modeling (RLSM)
    Baseado nos artigos "Data Driven Fuzzy Modeling".
    
    Este modelo atualiza os parâmetros do CONSEQUENTE (v_i, w_i)
    de forma recursiva usando RLS (Recursive Least Squares).
    Os ANTECEDENTES (centros das regras) são fixos após a inicialização.
    """
    
    def __init__(self, n_rules, n_features, forgetting_factor=0.99, P_alpha=1000.0, sigma=1.0):
        self.N = n_rules                 # Número de regras (N)
        self.n_features = n_features
        self.lambda_ = forgetting_factor # Fator de esquecimento (λ)
        self.sigma = sigma               # Largura (global) da gaussiana
        
        # Init Parâmetros 
        
        # Parâmetros do Consequente (u)
        # u = [v_1, w_1, v_2, w_2, ..., v_N, w_N]^T
        # Tamanho é (2 * N, 1)
        self.u = np.zeros((2 * self.N, 1)) # Inicializar com zeros
        
        # Matriz de Ganho (P)
        # Tamanho é (2N, 2N)
        # P^0 = alpha * I, com alpha grande
        self.P = P_alpha * np.identity(2 * self.N)
        
        # Parâmetros do Antecedente (Centros)
        # Serão definidos pelo método define_antecedents
        self.centers = np.zeros((self.N, self.n_features))
        print(f"Modelo RLSM inicializado com N={n_rules} regras.")

    def define_antecedents(self, X_sample):
        """
        Define os centros das N regras usando K-Means.
        No RLSM, os antecedentes A_i são fixos e o aprendizado
        ocorre nos consequentes.
        [Baseado na ideia de 'Cluster the data' - cite: 542, 521]
        """
        print(f"Definindo {self.N} centros de regras usando K-Means...")
        kmeans = KMeans(n_clusters=self.N, random_state=42, n_init=10)
        kmeans.fit(X_sample)
        self.centers = kmeans.cluster_centers_
        print("Centros das regras definidos.")

    def _calculate_activations(self, x_k):
        """
        Calcula o vetor de ativação 'd_k' para uma amostra x_k.
        """
        # x_k deve ser um vetor (n_features,)
        
        # --- Passo 1: Calcular Níveis de Ativação (tau_i) ---
        # tau_i = A_i(x_k)
        # [cite: 132]
        # Usamos uma MF Gaussiana multivariada (baseada em dist. Euclidiana)
        taus = np.zeros(self.N)
        for i in range(self.N):
            dist_sq = np.sum((x_k - self.centers[i])**2)
            taus[i] = np.exp(-dist_sq / (2 * self.sigma**2))
            
        # --- Passo 2: Calcular s_k e d_k ---
        # s_k = sum(tau_i)
        # [cite: 133]
        s_k = np.sum(taus)
        
        if s_k < 1e-9: # Evitar divisão por zero se nenhuma regra ativar
            return np.zeros((2 * self.N, 1)), 0.0
            
        # d_k = [(tau_1^2)/s_k, tau_1/s_k, ..., (tau_N^2)/s_k, tau_N/s_k]^T
        # [cite: 133]
        d_k = np.zeros(2 * self.N)
        for i in range(self.N):
            d_k[2*i]     = (taus[i]**2) / s_k # Coeficiente de v_i
            d_k[2*i + 1] = taus[i] / s_k      # Coeficiente de w_i
            
        # Retorna d_k como um vetor coluna
        return d_k.reshape(-1, 1), s_k

    def predict(self, x_k):
        """
        Prevê a saída y_hat para uma amostra x_k usando o estado ATUAL
        do vetor de parâmetros 'u'.
        [cite: 140]
        """
        d_k, s_k = self._calculate_activations(x_k)
        
        if s_k == 0.0:
            return 0.0 # Nenhuma regra ativada
            
        # y_hat = d_k * u (em notação vetorial, d_k^T * u)
        # [cite: 140]
        y_hat = d_k.T @ self.u
        
        return y_hat[0, 0] # Retorna o valor escalar

    def train_online(self, x_k, y_k):
        """
        Atualiza os parâmetros 'u' e a matriz de ganho 'P'
        usando uma nova amostra (x_k, y_k) via RLS.
        """
        # Obter vetor de ativação d_k
        d_k, s_k = self._calculate_activations(x_k)
        
        if s_k == 0.0:
            return # Nenhuma regra ativada, não é possível aprender

        d_k_T = d_k.T # Transposta (vetor linha)
        
        # Calcular o Erro de Predição (a priori)
        # erro = y_k - d_k * u^(k-1)
        # [parte de cite: 138]
        prediction_error = y_k - (d_k_T @ self.u)
        
        # Atualizar a Matriz de Ganho P
        # P^k = (1/λ) * (P^(k-1) - ...)
        # [cite: 136]
        P_num = self.P @ d_k @ d_k_T @ self.P
        P_den = self.lambda_ + (d_k_T @ self.P @ d_k)
        
        self.P = (1.0 / self.lambda_) * (self.P - (P_num / P_den[0, 0]))
        
        # Atualizar o Vetor de Parâmetros u
        # u^k = u^(k-1) + P^k * d_k * erro
        # [cite: 138]
        self.u = self.u + self.P @ d_k * prediction_error
        
        return prediction_error[0, 0]

if __name__ == "__main__":
    
    # Configuracoes
    DATA_PATH = 'datasets/adult.data'             
    N_RULES = 50                            # Número de regras fuzzy (hiperparâmetro)
    FORGET_FACTOR = 0.998       # Fator de esquecimento, próximo de 1 p/ estabilidade
    SIGMA = 0.5                             # Largura da MF Gaussiana (hiperparâmetro)
    INIT_SAMPLES = 1000                     # Amostras para definir os centros das regras
    
    # Carga e Pré-processamento 
    try:
        X, y, features = load_and_preprocess_adult(DATA_PATH)
        n_samples, n_features = X.shape
    except FileNotFoundError:
        print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
        print("Por favor, baixe o 'Adult Data Set' (adult.data) do repositório UCI.")
        exit()
    except Exception as e:
        print(f"Ocorreu um erro ao processar os dados: {e}")
        exit()

    # Inicialização do Modelo RLSM 
    model = RLSM(
        n_rules=N_RULES,
        n_features=n_features,
        forgetting_factor=FORGET_FACTOR,
        sigma=SIGMA
    )
    
    # Definir antecedentes (centros) usando as primeiras amostras
    model.define_antecedents(X[:INIT_SAMPLES])
    
    # Simulação Online (Preditiva-Corretiva) 
    print(f"\nIniciando simulação online para {n_samples - INIT_SAMPLES} amostras...")
    
    y_true_online = []
    y_pred_online = []
    squared_errors = []
    
    # Começamos a treinar/testar *após* os dados de inicialização
    for k in range(INIT_SAMPLES, n_samples):
        x_k = X[k]
        y_k = y[k]
        
        # PREVER com o modelo atual (antes de ver y_k)
        y_hat_raw = model.predict(x_k)
        
        # Aplicar limiar para classificação
        y_hat_class = 1 if y_hat_raw > 0.5 else 0
        
        # Armazenar resultados
        y_true_online.append(y_k)
        y_pred_online.append(y_hat_class)
        squared_errors.append((y_hat_raw - y_k)**2)
        
        # TREINAR o modelo com a amostra (x_k, y_k)
        model.train_online(x_k, y_k)
        
        if (k - INIT_SAMPLES + 1) % 5000 == 0:
            print(f"Processadas {k - INIT_SAMPLES + 1} amostras...")

    print("Simulação online concluída.")
    
    # Resultados 
    print("\n--- Relatório de Classificação Online ---")
    accuracy = accuracy_score(y_true_online, y_pred_online)
    print(f"Acurácia Online: {accuracy * 100:.2f}%")
    print(classification_report(y_true_online, y_pred_online, target_names=['<=50K', '>50K']))
    
    # Erro
    print("Gerando gráfico do erro de predição...")
    
    # Média Móvel do Erro Quadrático
    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
        
    mse_ma = moving_average(squared_errors, n=500)
    
    plt.figure(figsize=(12, 6))
    plt.plot(mse_ma)
    plt.title(f'Erro Quadrático Médio Móvel (Janela=500)\nModelo RLSM (N={N_RULES}, λ={FORGET_FACTOR})')
    plt.xlabel('Amostras (tempo)')
    plt.ylabel('EQM Móvel')
    plt.grid(True)
    plt.ylim(bottom=0)
    
    output_folder = 'output_images'
    file_name = 'rlsm_mse_adult_dataset.png'
    output_path = os.path.join(output_folder, file_name)
    
    try:
        plt.savefig(output_path)
        print(f"Gráfico salvo com sucesso em: {output_path}")
    except Exception as e:
        print(f"Erro ao salvar o gráfico em '{output_path}': {e}")
        print("Certifique-se de que a pasta 'output_images' existe no mesmo diretório.")