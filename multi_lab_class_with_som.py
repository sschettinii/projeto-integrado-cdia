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
    return np.unravel_index(np.nanargmin(dist), dist.shape)

def neighbor_func(bmu_idx, m, n, sigma_decayed):
    I, J = np.indices((m, n))
    u_c, v_c = bmu_idx
    dist_sq = (I - u_c)**2 + (J - v_c)**2
    return np.exp(-dist_sq / (2 * (sigma_decayed**2) + 1e-8))

def update_weights(grid, alpha_decayed, h_ci, x):
    h_ci_expanded = h_ci[:, :, np.newaxis]
    grid += alpha_decayed * h_ci_expanded * (x - grid)

def som_train(x_train, m, n, alpha_0, T, sigma_0=None, train_mode='sequential', decay_method='exp', tol=1e-6):
    if sigma_0 is None:
        sigma_0 = max(m, n) / 2
    n_samples, input_dim = x_train.shape
    grid = init_params(m, n, input_dim)

    for epoch in range(T):
        alpha_decayed, sigma_decayed = decay_params(alpha_0, sigma_0, epoch, T, decay_method=decay_method)
        grid_prev = grid.copy()

        if train_mode == 'sequential':
            for x in x_train:
                bmu_idx = get_bmu_idx(grid, x)
                h_ci = neighbor_func(bmu_idx, m, n, sigma_decayed)
                update_weights(grid, alpha_decayed, h_ci, x)

        elif train_mode == 'batch':
            sub_lists = [[[] for _ in range(n)] for _ in range(m)]

            for x in x_train:
                bmu_i, bmu_j = get_bmu_idx(grid, x)
                sub_lists[bmu_i][bmu_j].append(x)
                
            for i in range(m):
                for j in range(n):
                    samples = []

                    for di in range(m):
                        for dj in range(n):
                            dist_grid = np.sqrt((i - di)**2 + (j - dj)**2)

                            if dist_grid <= sigma_decayed:
                                samples.extend(sub_lists[di][dj])
                                
                    if len(samples) > 0:
                        grid[i, j] = np.mean(samples, axis=0)
        
        else:
            raise ValueError(f"train_mode '{train_mode}' inválido. Use 'sequential' ou 'batch'.")

        if np.allclose(grid, grid_prev, atol=tol):
            print(f"Convergência atingida na época {epoch + 1}.")
            break

    return grid


def prune_neurons(grid, x_train, min_instances=4):
    m, n, _ = grid.shape
    counts = np.zeros((m, n), dtype=int)

    # count how many data points each neuron is BMU
    for x in x_train:
        bmu_i, bmu_j = get_bmu_idx(grid, x)
        counts[bmu_i][bmu_j] += 1

    valid_mask = counts >= min_instances

    grid_pruned = grid.copy().astype(float)
    grid_pruned[~valid_mask] = np.nan

    return grid_pruned, valid_mask


def compute_neuron_thresholds(grid, valid_mask, x_train, class_idx, P, n_classes):
    m, n, _ = grid.shape
    aver_out   = np.full((m, n), np.nan)
    thresholds = np.full((m, n), np.nan)

    sub_lists = [[[] for _ in range(n)] for _ in range(m)]
    for x in x_train:
        bmu_i, bmu_j = get_bmu_idx(grid, x)
        sub_lists[bmu_i][bmu_j].append(x)

    j = class_idx
    p_yj = P[j, j]

    conditional_prod = 1.0
    for k in range(n_classes):
        if k == j:
            continue
        p_yk_given_yj = P[k, j]
        if p_yk_given_yj > 0:
            conditional_prod *= p_yk_given_yj

    for i in range(m):
        for jj in range(n):
            if not valid_mask[i, jj] or len(sub_lists[i][jj]) == 0:
                continue

            mb = grid[i, jj]
            Xb = np.array(sub_lists[i][jj])

            dists = np.linalg.norm(Xb - mb, axis=1)
            aver_out[i, jj] = np.mean(np.exp(-dists))

            thresholds[i, jj] = p_yj * conditional_prod * aver_out[i, jj]

    return aver_out, thresholds


if __name__ == "__main__":
    n_classes = 5
    m, n = 10, 10

    # generate synthetic multi-class data with sklearn.datasets
    X, Y = generate_multilabel_data(n_samples=600, n_features=12, n_classes=n_classes, n_labels=2)

    # build T
    T = Y.T @ Y

    # build P
    N = Y.shape[0]
    T_diag = np.diag(T)
    P = np.divide(T, T_diag, out=np.zeros_like(T, dtype=float), where=T_diag!=0)
    marginal_prob = T_diag / N
    np.fill_diagonal(P, marginal_prob)

    class_datasets = {
        c: X[Y[:, c] == 1] 
        for c in range(n_classes)
    }

    soms         = {}
    valid_masks  = {}
    aver_outs    = {}
    thresholds   = {}

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
        )

        trained_grid, valid_mask = prune_neurons(trained_grid, x_subset, min_instances=4)

        aver_out, thresh = compute_neuron_thresholds(
            grid       = trained_grid,
            valid_mask = valid_mask,
            x_train    = x_subset,
            class_idx  = label,
            P          = P,
            n_classes  = n_classes,
        )

        soms[label]        = trained_grid
        valid_masks[label] = valid_mask
        aver_outs[label]   = aver_out
        thresholds[label]  = thresh

        n_valid = valid_mask.sum()
        print(f"  Classe {label}: {n_valid}/{m*n} neurônios válidos | "
              f"threshold médio = {np.nanmean(thresh):.4f}")

    print("\nTreinamento concluído!")