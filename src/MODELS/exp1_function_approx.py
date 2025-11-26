import numpy as np
import matplotlib.pyplot as plt
import os 
from dlsm_lib import LevelSetFuzzyModel

def f1(x1, x2):
    term1 = np.exp(x1) * np.sin(13 * (x1 - 0.6)**2)
    term2 = np.exp(-x2) * np.sin(7 * x2)
    return 1.9 * (1.35 + term1 * term2)

def f2(x1, x2):
    return np.sin(x1) * np.sin(x2)

def run_experiment_f1():
    np.random.seed(42)
    X_train = np.random.rand(100, 2)
    y_train = f1(X_train[:, 0], X_train[:, 1])
    
    x = np.linspace(0, 1, 31)
    xx, yy = np.meshgrid(x, x)
    X_test = np.column_stack((xx.ravel(), yy.ravel()))
    y_test = f1(X_test[:, 0], X_test[:, 1])

    model = LevelSetFuzzyModel(n_rules=25)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = np.mean((y_test - y_pred)**2)
    print(f"--- Experimento F1 --- MSE: {mse:.5f}")
    
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx, yy, y_test.reshape(xx.shape), cmap='viridis', alpha=0.8)
    ax1.set_title('Superfície Real f1')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, y_pred.reshape(xx.shape), cmap='plasma', alpha=0.8)
    ax2.set_title(f'Estimativa DLSM (MSE={mse:.4f})')
    
    save_path = os.path.join('figures', 'exp1_f1_approx.png')
    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")
    plt.close() 

def run_experiment_f2():
    np.random.seed(42)
    X_train = np.random.uniform(-3, 3, (400, 2))
    y_train = f2(X_train[:, 0], X_train[:, 1])
    
    x = np.linspace(-3, 3, 30)
    xx, yy = np.meshgrid(x, x)
    X_test = np.column_stack((xx.ravel(), yy.ravel()))
    y_test = f2(X_test[:, 0], X_test[:, 1])

    model = LevelSetFuzzyModel(n_rules=4) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = np.mean((y_test - y_pred)**2)
    print(f"--- Experimento F2 --- MSE: {mse:.5f}")

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx, yy, y_test.reshape(xx.shape), cmap='viridis', alpha=0.8)
    ax1.set_title('Superfície Real f2')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, y_pred.reshape(xx.shape), cmap='plasma', alpha=0.8)
    ax2.set_title(f'Estimativa DLSM f2 (MSE={mse:.4f})')

    save_path = os.path.join('figures', 'exp1_f2_approx.png')
    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('figures'):
        os.makedirs('figures')
    run_experiment_f1()
    run_experiment_f2()