import numpy as np
from collections import deque
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

class MOASphericalStream:
    def __init__(
        self, 
        n_features=2, 
        n_classes=5, 
        n_samples=95000, 
        sd=1500,
        seed=42, 
        drift_speed=0.02
    ):
        self.rng = np.random.RandomState(seed)
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.sd = sd
        self.drift_speed = drift_speed
        self.t = 0

        self.centers = self.rng.uniform(0.15, 0.85, size=(n_classes, n_features))
        self.radii = self._compute_radii()
        self.sigmas = self.radii * 0.4

        self.velocities = self.rng.randn(n_classes, n_features) * drift_speed

        self.substream_size = n_samples // 4
        self._init_label_relationships()

    def _compute_radii(self):
        if self.n_classes <= 2:
            return self.rng.uniform(0.25, 0.40, size=self.n_classes)

        dists = []
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                dists.append(np.linalg.norm(self.centers[i] - self.centers[j]))

        mean_dist = np.mean(dists) if dists else 0.5
        base_radius = mean_dist * 0.45
        return self.rng.uniform(base_radius * 0.8, base_radius * 1.2, size=self.n_classes)

    def _init_label_relationships(self):
        self.P_base = np.zeros((self.n_classes, self.n_classes))

        for j in range(self.n_classes):
            for k in range(self.n_classes):
                if j == k:
                    continue
                dist_jk = np.linalg.norm(self.centers[j] - self.centers[k])
                overlap = max(0, (self.radii[j] + self.radii[k]) - dist_jk)
                max_overlap = self.radii[j] + self.radii[k]
                self.P_base[k, j] = overlap / max_overlap if max_overlap > 0 else 0

        self.P_perturbed = {}
        for sub_idx in range(4):
            P_sub = self.P_base.copy()
            if sub_idx in (1, 2):
                frac = 0.10
            elif sub_idx == 3:
                frac = 0.30
            else:
                self.P_perturbed[sub_idx] = P_sub
                continue

            off_diag = [(k, j) for k in range(self.n_classes)
                        for j in range(self.n_classes) if k != j]
            n_perturb = max(1, int(len(off_diag) * frac))
            chosen = self.rng.choice(len(off_diag), size=n_perturb, replace=False)

            for idx in chosen:
                k, j = off_diag[idx]
                p_yk = self.P_base[k, k] if k < self.n_classes else 0.5
                noise = self.rng.normal(loc=self.P_base[k, j], scale=0.3)
                P_sub[k, j] = np.clip(noise, 0.0, 1.0)

            self.P_perturbed[sub_idx] = P_sub

    def _get_substream_index(self):
        idx = self.t // self.substream_size
        return min(idx, 3)

    def _apply_spatial_drift(self):
        if self.sd > 0 and self.t > 0 and self.t % self.sd == 0:
            self.centers += self.velocities
            self.centers = np.clip(self.centers, 0.05, 0.95)

            self.velocities += self.rng.randn(self.n_classes, self.n_features) * (self.drift_speed * 0.3)
            self.velocities = np.clip(self.velocities, -self.drift_speed * 2, self.drift_speed * 2)

    def get_sample(self):
        self._apply_spatial_drift()

        primary_class = self.rng.randint(0, self.n_classes)
        x = self.rng.normal(loc=self.centers[primary_class],
                            scale=self.sigmas[primary_class],
                            size=self.n_features)
        x = np.clip(x, 0.0, 1.0)

        y = np.zeros(self.n_classes, dtype=int)
        dists = np.linalg.norm(self.centers - x, axis=1)
        y[dists <= self.radii] = 1

        if y.sum() == 0:
            y[primary_class] = 1

        sub_idx = self._get_substream_index()
        P_current = self.P_perturbed[sub_idx]
        active = np.where(y == 1)[0]
        for c in range(self.n_classes):
            if y[c] == 1:
                continue
            for d in active:
                if P_current[c, d] > 0 and self.rng.rand() < P_current[c, d] * 0.15:
                    y[c] = 1
                    break

        self.t += 1
        return x, y

    def get_offline_data(self, fraction=0.10):
        n_offline = int(self.n_samples * fraction)
        X_list, Y_list = [], []
        for _ in range(n_offline):
            x, y = self.get_sample()
            X_list.append(x)
            Y_list.append(y)
        return np.array(X_list), np.array(Y_list)

    def stream_samples(self, interval=0.1):
        import time
        while self.t < self.n_samples:
            yield self.get_sample()
            if interval > 0:
                time.sleep(interval)


class StreamEvaluator:
    def __init__(self, n_classes, window_size=50):
        self.n_classes = n_classes
        self.window_size = window_size
        self.y_true_window = deque(maxlen=window_size)
        self.y_pred_window = deque(maxlen=window_size)

    def update(self, y_true, y_pred):
        if isinstance(y_true, (list, np.ndarray)) and len(y_true) == self.n_classes and set(y_true).issubset({0, 1}):
            true_vec = np.array(y_true, dtype=int)
        else:
            true_vec = np.zeros(self.n_classes, dtype=int)
            true_vec[list(y_true)] = 1

        if isinstance(y_pred, (list, np.ndarray)) and len(y_pred) == self.n_classes and set(y_pred).issubset({0, 1}):
            pred_vec = np.array(y_pred, dtype=int)
        else:
            pred_vec = np.zeros(self.n_classes, dtype=int)
            pred_vec[list(y_pred)] = 1

        self.y_true_window.append(true_vec)
        self.y_pred_window.append(pred_vec)

    def compute_jaccard_accuracy(self):
        if len(self.y_true_window) == 0:
            return 0.0

        jaccards = []
        for y_t, y_p in zip(self.y_true_window, self.y_pred_window):
            intersection = np.logical_and(y_t, y_p).sum()
            union = np.logical_or(y_t, y_p).sum()
            jaccards.append(1.0 if union == 0 else intersection / union)

        return float(np.mean(jaccards))

    def compute_macro_f1(self):
        if len(self.y_true_window) == 0:
            return 0.0

        Y_true_mat = np.array(self.y_true_window)
        Y_pred_mat = np.array(self.y_pred_window)

        f1_scores = []
        for c in range(self.n_classes):
            tp = np.logical_and(Y_true_mat[:, c] == 1, Y_pred_mat[:, c] == 1).sum()
            fp = np.logical_and(Y_true_mat[:, c] == 0, Y_pred_mat[:, c] == 1).sum()
            fn = np.logical_and(Y_true_mat[:, c] == 1, Y_pred_mat[:, c] == 0).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 1.0 if (tp + fp + fn) == 0 else 0.0

            f1_scores.append(f1)

        return float(np.mean(f1_scores))

    def get_metrics(self):
        return {
            'jaccard': self.compute_jaccard_accuracy(),
            'macro_f1': self.compute_macro_f1(),
            'window_size': len(self.y_true_window)
        }


if __name__ == "__main__":
    n_features = 2
    n_classes = 5
    n_samples = 95000
    sd = 1500
    m, n = 2, 2

    stream_gen = MOASphericalStream(
        n_features=n_features,
        n_classes=n_classes,
        n_samples=n_samples,
        sd=sd,
        seed=42,
    )

    X, Y = stream_gen.get_offline_data(fraction=0.10)

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

    evaluator = StreamEvaluator(n_classes=n_classes, window_size=50)

    try:
        for x_t, y_t in stream_gen.stream_samples(interval=0.1):
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

            evaluator.update(y_true=y_t, y_pred=Y_pred)
            metrics = evaluator.get_metrics()

            reais_str = str(labels_ativas)
            pred_str = str(Y_pred)

            print(
                f"[t={stream_gen.t:04d}] "
                f"Real: {reais_str:<16} | "
                f"Pred: {pred_str:<14} | "
                f"Jaccard (W={metrics['window_size']:02d}): {metrics['jaccard']*100:5.1f}% | "
                f"Macro F1: {metrics['macro_f1']:.3f}"
            )

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