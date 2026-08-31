import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import MinMaxScaler

# n_classes: Total number of classes
# n_labels: Average number of active classes assigned to each instance
def generate_multilabel_data(n_samples=1000, n_features=20, n_classes=5, n_labels=2, random_state=42):
    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        allow_unlabeled=False,
        random_state=random_state
    )
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, Y

def init_params(m, n, data_dim):
    return np.random.randn(m, n, data_dim)

def decay(param, epoch, T, decay_method='lin'):
    if decay_method == 'lin':
        return param * (1 - (epoch / T))
    else:
        return param * np.exp(-(epoch / T))

def decay_params(alpha_0, sigma_0, epoch, T, decay_method='exp'):
    alpha_decayed, sigma_decayed = decay(np.array([alpha_0, sigma_0]), epoch, T, decay_method=decay_method)
    return alpha_decayed, sigma_decayed

def get_bmu_idx(grid, instance):
    dist = np.linalg.norm(grid - instance, axis=-1)
    return np.unravel_index(np.argmin(dist), dist.shape)

def neighbor_func(bmu_idx, m, n, sigma_decayed):
    I, J = np.indices((m, n))
    u_c, v_c = bmu_idx
    dist_sq = (I - u_c)**2 + (J - v_c)**2
    return np.exp(-dist_sq / (2 * (sigma_decayed**2) + 1e-8))

def update_weights(grid, alpha_decayed, h_ci, x):
    h_ci_expanded = h_ci[:, :, np.newaxis]
    grid += alpha_decayed * h_ci_expanded * (x - grid)

def som_train(x_train, m, n, alpha_0, sigma_0, T, train_mode='sequential', batch_size=10, decay_method='exp'):
    n_samples, input_dim = x_train.shape
    grid = init_params(m, n, input_dim)

    if train_mode == 'batch' and (batch_size is None or batch_size <= 0):
        raise ValueError("batch mode requires batch_size > 0.")

    for epoch in range(T):
        alpha_decayed, sigma_decayed = decay_params(alpha_0, sigma_0, epoch, T, decay_method=decay_method)
        
        if train_mode == 'sequential':
            for x in x_train:
                bmu_idx = get_bmu_idx(grid, x)
                h_ci = neighbor_func(bmu_idx, m, n, sigma_decayed)
                update_weights(grid, alpha_decayed, h_ci, x)

        elif train_mode == 'batch':
            for start_idx in range(0, n_samples, batch_size):
                batch = x_train[start_idx: start_idx + batch_size]

                numerator = np.zeros((m, n, input_dim))
                denominator = np.zeros((m, n, 1))

                for x in batch:
                    bmu_idx = get_bmu_idx(grid, x)
                    h_ci = neighbor_func(bmu_idx, m, n, sigma_decayed)

                    numerator += h_ci * x
                    denominator += h_ci
                
                mask = denominator > 0
                grid = np.where(mask, numerator / denominator, grid)
        
        else:
            raise ValueError(f"train_mode '{train_mode}' inválido. Use 'sequential' ou 'batch'.")

    return grid


if __name__ == "__main__":
    n_classes = 5
    m, n = 10, 10
    X, Y = generate_multilabel_data(n_samples=600, n_features=12, n_classes=n_classes, n_labels=2)

    T = Y.T @ Y

    N = Y.shape[0]
    T_diag = np.diag(T)
    P = np.divide(T, T_diag, out=np.zeros_like(T, dtype=float), where=T_diag!=0)
    marginal_prob = T_diag / N
    np.fill_diagonal(P, marginal_prob)

    class_datasets = {
        c: X[Y[:, c] == 1] 
        for c in range(n_classes)
    }

    soms = {}
    for label in range(n_classes):
        x_subset = class_datasets[label]
        
        trained_grid = som_train(
            x_train=x_subset,
            m=m,
            n=n,
            alpha_0=0.5,
            sigma_0=max(m, n) / 2.0,
            T=100,
            train_mode='batch',
            decay_method='exp',
            batch_size=10
        )
        soms[label] = trained_grid
    
    print("Treinamento concluído!")