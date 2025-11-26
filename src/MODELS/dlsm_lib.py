import numpy as np
from sklearn.cluster import KMeans

class LevelSetFuzzyModel:
    def __init__(self, n_rules=5, membership_type='gaussian'):
        self.n_rules = n_rules
        self.centers = None
        self.sigmas = None
        self.params = None 

    def _compute_membership(self, X):
        N, _ = X.shape
        taus = np.zeros((N, self.n_rules))
        for i in range(self.n_rules):
            diff = X - self.centers[i]
            sq_dist = np.sum(diff**2, axis=1)
            taus[:, i] = np.exp(-sq_dist / (2 * self.sigmas[i]**2))
        return taus

    def _construct_matrix_d(self, taus):
        N, _ = taus.shape
        sum_taus = np.sum(taus, axis=1, keepdims=True)
        sum_taus[sum_taus == 0] = 1e-10 
        
        D = np.zeros((N, 2 * self.n_rules))
        for i in range(self.n_rules):
            tau_i = taus[:, i:i+1]
            D[:, 2*i] = (tau_i**2 / sum_taus).flatten()
            D[:, 2*i+1] = (tau_i / sum_taus).flatten()
        return D

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.n_rules, n_init=10, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
        self.sigmas = np.zeros(self.n_rules)
        for i in range(self.n_rules):
            dists = np.linalg.norm(X - self.centers[i], axis=1)
            if len(dists) > 0:
                self.sigmas[i] = np.std(dists) + 1e-6
            else:
                self.sigmas[i] = 1.0

        taus = self._compute_membership(X)
        D = self._construct_matrix_d(taus)
        self.params = np.linalg.pinv(D) @ y

    def predict(self, X):
        taus = self._compute_membership(X)
        D = self._construct_matrix_d(taus)
        return D @ self.params

class RecursiveLevelSetFuzzyModel(LevelSetFuzzyModel):
    def __init__(self, n_rules=6, lam=0.98, alpha=1000.0):
        super().__init__(n_rules=n_rules)
        self.lam = lam
        self.P = None
        self.u = None 
        
    def init_structure(self, X_init):
        kmeans = KMeans(n_clusters=self.n_rules, n_init=10, random_state=42)
        kmeans.fit(X_init)
        self.centers = kmeans.cluster_centers_
        
        self.sigmas = np.zeros(self.n_rules)
        for i in range(self.n_rules):
            dists = np.linalg.norm(X_init - self.centers[i], axis=1)
            self.sigmas[i] = np.std(dists) + 1e-6

        self.u = np.zeros(2 * self.n_rules)
        self.P = self.alpha_val * np.eye(2 * self.n_rules)

    def fit_recursive(self, X, y, alpha_val=1000.0):
        self.alpha_val = alpha_val
        if self.centers is None:
            self.init_structure(X)
            
        preds = []
        for k in range(len(X)):
            x_k = X[k:k+1]
            y_k = y[k]
            
            taus = self._compute_membership(x_k)
            d_k = self._construct_matrix_d(taus).flatten() 
            
            d_k_T = d_k.reshape(-1, 1)
            d_k_vec = d_k.reshape(1, -1)
            
            numerator = self.P @ d_k_T @ d_k_vec @ self.P
            denominator = self.lam + d_k_vec @ self.P @ d_k_T
            self.P = (1/self.lam) * (self.P - numerator/denominator)
            
            error = y_k - d_k_vec @ self.u
            gain = self.P @ d_k_T
            self.u = self.u + (gain.flatten() * error)
            
            preds.append(d_k_vec @ self.u)
            
        self.params = self.u
        return np.array(preds)