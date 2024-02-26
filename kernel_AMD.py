import numpy as np

class Kernel_Ridge_AMD:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        
        self.S = np.cov(X, rowvar=False)
        L = np.linalg.cholesky(self.S)      ## Cholesky decomposition
        L_inv = np.linalg.inv(L)
        self.S_inv = np.dot(L_inv.T, L_inv)
        
        self.K = np.eye(self.n)
        for i in range(self.n):
            for j in range(i+1, self.n):
                self.K[i, j] = self.kernel_rbf(self.X[i], self.X[j])
              
        self.K = self.K + self.K.T - np.eye(self.n)
        
        
    def mahalanobis_dist(self, Xi, Xj):
        diff = Xi - Xj
        return np.dot(np.dot(diff, self.S_inv), diff.T)
    
    
    def kernel_rbf(self, Xi, Xj):
        return np.exp(-self.beta * self.mahalanobis_dist(Xi, Xj))
    
    
    def partial_matrix(self):
        partial_12 = 2 * self.S_inv[0,1]
        matrix = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(i, self.n):
                diff = self.X[i] - self.X[j]
                partial_1 = 2 * np.dot(self.S_inv[0], diff.T)
                partial_2 = 2 * np.dot(self.S_inv[1], diff.T)
                matrix[i, j] = (np.square(self.beta) * partial_1 * partial_2 - self.beta * partial_12) * self.K[i, j]
            
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))
        return matrix
    
    
    def KRR_est(self, X_test):
        A = self.K + self.alpha * np.eye(self.n)
        A_inv = np.linalg.inv(A)
        
        m = X_test.shape[0]
        K_test = np.zeros((m, self.n))
        for i in range(m):
            for j in range(self.n):
                K_test[i, j] = self.kernel_rbf(X_test[i], self.X[j])
                
        Y_pred = np.dot(K_test, np.dot(A_inv, self.Y.T))
        return Y_pred
    
    
    def AMD(self):
        A = self.K + self.alpha * np.eye(self.n)
        L = np.linalg.cholesky(A)
        L_inv = np.linalg.inv(L)
        A_inv = np.dot(L_inv.T, L_inv)
        AMD = np.mean(np.dot(self.partial_matrix(), np.dot(A_inv, self.Y.T)))
        
        return AMD
    
    
    def AMD_truncate(self, p=0.1):
        A = self.K + self.alpha * np.eye(self.n)
        L = np.linalg.cholesky(A)
        L_inv = np.linalg.inv(L)
        A_inv = np.dot(L_inv.T, L_inv)
        est = np.dot(self.partial_matrix(), np.dot(A_inv, self.Y.T))
        
        n = round(self.n * p)
        max_n = np.partition(est, -n)[-n]
        min_n = np.partition(est, n-1)[n-1]
        
        est_truncate = np.clip(est, min_n, max_n)
        AMD_truncate = np.mean(est_truncate)
        
        return AMD_truncate
