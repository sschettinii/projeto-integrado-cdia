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
    if np.all(np.isnan(dist)):
        return (0, 0)
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

    for x in x_train:
        bmu_i, bmu_j = get_bmu_idx(grid, x)
        counts[bmu_i][bmu_j] += 1

    valid_mask = counts >= min_instances
    if not np.any(valid_mask):
        max_count = counts.max()
        valid_mask = (counts == max_count) if max_count > 0 else np.ones((m, n), dtype=bool)

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


def update_thresholds(valid_masks, aver_outs, P, n_classes):
    thresholds = {}
    for j in range(n_classes):
        p_yj = P[j, j]
        conditional_prod = 1.0
        for k in range(n_classes):
            if k != j and P[k, j] > 0:
                conditional_prod *= P[k, j]

        thresholds[j] = np.where(
            valid_masks[j],
            p_yj * conditional_prod * aver_outs[j],
            np.nan
        )
    return thresholds

def sort_neurons(grid, instance, valid_mask=None):
    dists = np.linalg.norm(grid - instance, axis=-1)
    if valid_mask is None:
        valid_mask = ~np.isnan(dists)

    valid_indices = np.argwhere(valid_mask)
    neuron_list = []
    for i, j in valid_indices:
        neuron_list.append({
            'idx': (int(i), int(j)),
            'dist': float(dists[i, j]),
            'weight': grid[i, j]
        })

    neuron_list.sort(key=lambda item: item['dist'])
    return neuron_list


def get_knn(NrSort, kn):
    remaining_classes = list(NrSort.keys())
    win_classes = []

    while len(remaining_classes) > 1:
        candidates = []
        for c in remaining_classes:
            for neuron in NrSort[c]:
                candidates.append((neuron['dist'], c))

        candidates.sort(key=lambda x: x[0])

        k = min(kn, len(candidates))
        top_k = candidates[:k]

        votes = {c: 0 for c in remaining_classes}
        for _, c in top_k:
            votes[c] += 1

        winner = max(
            remaining_classes,
            key=lambda c: (votes[c], -min((d for d, cls in candidates if cls == c), default=float('inf')))
        )

        win_classes.append(winner)
        remaining_classes.remove(winner)

    if remaining_classes:
        win_classes.append(remaining_classes[0])

    return win_classes

def predict_classes(WinClasses, outputWinNr, WinNr, P, thresholds, z):
    c1 = WinClasses[0]
    Y_pred = [c1]

    max_candidates = min(int(np.ceil(z)), len(WinClasses))

    for k in range(1, max_candidates):
        c = WinClasses[k]
        p_yc = P[c, c]
        p_xi_given_yc = outputWinNr[c]

        p_yd_given_yc = 1.0
        for d in Y_pred:
            if P[d, c] > 0:
                p_yd_given_yc *= P[d, c]

        p_bayes = p_yc * p_yd_given_yc * p_xi_given_yc

        tr = thresholds[c][WinNr[c]]

        if p_bayes >= tr:
            Y_pred.append(c)

    return Y_pred

class MultiLabelDataStream:
    def __init__(self, n_features=12, n_classes=5, p_multilabel=0.7, seed=42):
        np.random.seed(seed)
        self.n_features = n_features
        self.n_classes = n_classes
        self.p_multilabel = p_multilabel
        self.t = 0
        
        self.centers = np.random.uniform(0.2, 0.8, size=(n_classes, n_features))
        self.sigmas = np.random.uniform(0.05, 0.10, size=n_classes)
        self.radii = np.random.uniform(0.25, 0.40, size=n_classes)

    def get_sample(self):
        if np.random.rand() < self.p_multilabel and self.n_classes >= 2:
            n_active = np.random.choice([2, min(3, self.n_classes)])
        else:
            n_active = 1

        active_indices = np.random.choice(self.n_classes, size=n_active, replace=False)
        
        center = self.centers[active_indices].mean(axis=0)
        sigma = self.sigmas[active_indices].mean()
        
        x = np.random.normal(loc=center, scale=sigma, size=self.n_features)
        x = np.clip(x, 0.0, 1.0)
        
        y = np.zeros(self.n_classes, dtype=int)
        y[active_indices] = 1
        
        dists = np.linalg.norm(self.centers - x, axis=1)
        y[dists <= self.radii] = 1
        
        self.t += 1
        return x, y

    def stream_samples(self, interval=1.5):
        import time
        while True:
            yield self.get_sample()
            if interval > 0:
                time.sleep(interval)

if __name__ == "__main__":
    n_classes = 5
    m, n = 10, 10
    n_features = 6

    X, Y = generate_multilabel_data(n_samples=600, n_features=n_features, n_classes=n_classes, n_labels=2)

    T = Y.T @ Y
    N = Y.shape[0]
    T_diag = np.diag(T)
    P = np.divide(T, T_diag, out=np.zeros_like(T, dtype=float), where=T_diag!=0)
    marginal_prob = T_diag / N
    np.fill_diagonal(P, marginal_prob)

    z = float(Y.sum(axis=1).mean())

    class_datasets = {c: X[Y[:, c] == 1] for c in range(n_classes)}

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
        print(f"  Classe {label}: {n_valid}/{m*n} neurônios válidos | threshold médio = {np.nanmean(thresh):.4f}")

    min_neurons = min(mask.sum() for mask in valid_masks.values())
    kn = min_neurons if min_neurons % 2 != 0 else min_neurons - 1
    kn = max(1, kn)

    stream_gen = MultiLabelDataStream(n_features=n_features, n_classes=n_classes, seed=42)

    try:
        for x_t, y_t in stream_gen.stream_samples(interval=1.0):
            labels_ativas = np.where(y_t == 1)[0].tolist()

            NrSort = {}
            WinNr = {}
            outputWinNr = {}

            for j in range(n_classes):
                NrSort[j] = sort_neurons(soms[j], x_t, valid_mask=valid_masks[j])

                WinNr[j] = NrSort[j][0]['idx']

                outputWinNr[j] = np.exp(-NrSort[j][0]['dist'])

            WinClasses = get_knn(NrSort, kn)

            Y_pred = predict_classes(WinClasses, outputWinNr, WinNr, P, thresholds, z)

            print(f"[t={stream_gen.t:04d}] Labels reais: {labels_ativas} | Predição Y: {Y_pred}")

            eta = 0.05
            for label in Y_pred:
                bmu_idx = WinNr[label]
                soms[label][bmu_idx] += eta * (x_t - soms[label][bmu_idx])

            N += 1
            z = ((N - 1) * z + len(Y_pred)) / N

            for label in Y_pred:
                bmu_idx = WinNr[label]
                m_updated = soms[label][bmu_idx]
                aver_outs[label][bmu_idx] += np.exp(-np.linalg.norm(x_t - m_updated))

            for j in Y_pred:
                for k in Y_pred:
                    T[j, k] += 1

            T_diag = np.diag(T)
            P = np.divide(T, T_diag, out=np.zeros_like(T, dtype=float), where=T_diag != 0)
            marginal_prob = T_diag / N
            np.fill_diagonal(P, marginal_prob)

            thresholds = update_thresholds(valid_masks, aver_outs, P, n_classes)

    except KeyboardInterrupt:
        print(f"\nStream interrompido pelo usuário no tempo t={stream_gen.t}.")