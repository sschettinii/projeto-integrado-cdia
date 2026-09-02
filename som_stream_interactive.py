"""
som_stream_interactive.py

Interactive multi-label 2D stream visualization for the Medical Domain
integrating real-time SOM (Self-Organizing Maps) training.

X-axis: Heart Rate (bpm)
Y-axis: Blood Oxygenation (SpO2 %)

Controls:
  || Pausar / > Retomar   — toggle stream generation
  + Nova Classe           — add a predefined new medical condition (without init SOM)
  Velocidade slider       — points generated per animation frame (1–30)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.widgets as mwidgets
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from collections import deque

# ── Dark theme ────────────────────────────────────────────────────────────────
BG       = '#0A0D14'
PANEL_BG = '#111520'
GRID_C   = '#1C2235'
TEXT_C   = '#C8D4F0'
MUTED_C  = '#5A6A8A'
BTN_C    = '#1A2035'
BTN_H    = '#263050'
MULTI_EDGE = '#FFD700'   # gold border → multi-label point
NOISE_C    = '#888899'   # no-label point color

plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'axes.facecolor'  : PANEL_BG,
    'figure.facecolor': BG,
    'axes.edgecolor'  : GRID_C,
    'xtick.color'     : MUTED_C,
    'ytick.color'     : MUTED_C,
    'grid.color'      : GRID_C,
    'grid.linewidth'  : 0.5,
    'text.color'      : TEXT_C,
    'axes.titlecolor' : TEXT_C,
})

PALETTE = [
    '#5B9CF6', '#FB923C', '#4ADE80', '#C084FC',
    '#F43F5E', '#EAB308', '#06B6D4', '#A3E635',
    '#E879F9', '#F97316',
]

# Domain Mapping (internal coordinates in [0,1])
HR_MIN, HR_MAX = 40, 220
SPO2_MIN, SPO2_MAX = 60, 100

def map_to_norm(hr, spo2):
    x = (hr - HR_MIN) / (HR_MAX - HR_MIN)
    y = (spo2 - SPO2_MIN) / (SPO2_MAX - SPO2_MIN)
    return x, y


# ══════════════════════════════════════════════════════════════════════════════
# SOM Engine
# ══════════════════════════════════════════════════════════════════════════════

def init_params(m, n, data_dim, center_bias=None):
    """Inicializa os pesos do SOM."""
    if center_bias is not None:
        # Inicia a malha ao redor de um ponto central específico
        grid = np.random.randn(m, n, data_dim) * 0.05 + center_bias
    else:
        # Inicia a malha com ruído aleatório em [0.4, 0.6]
        grid = np.random.rand(m, n, data_dim) * 0.2 + 0.4
    return grid

def get_bmu_idx(grid, instance):
    dist = np.linalg.norm(grid - instance, axis=-1)
    return np.unravel_index(np.nanargmin(dist), dist.shape)

def neighbor_func(bmu_idx, m, n, sigma):
    I, J = np.indices((m, n))
    u_c, v_c = bmu_idx
    dist_sq = (I - u_c)**2 + (J - v_c)**2
    return np.exp(-dist_sq / (2 * (sigma**2) + 1e-8))

def update_weights(grid, alpha, h_ci, x):
    h_ci_expanded = h_ci[:, :, np.newaxis]
    grid += alpha * h_ci_expanded * (x - grid)

class SOMMap:
    """Um mapa SOM atrelado a uma classe específica (treino online contínuo)."""
    def __init__(self, idx, m=4, n=4, input_dim=2, initial_center=None):
        self.idx = idx
        self.m = m
        self.n = n
        self.grid = init_params(m, n, input_dim, center_bias=initial_center)
        # Parâmetros de aprendizado fixos para aprendizado contínuo (online sem decay)
        self.alpha = 0.05
        self.sigma = 1.0  # Raio topológico na malha

    def online_train_step(self, x):
        """Atualiza a rede com um novo exemplo (sequencial)."""
        bmu_idx = get_bmu_idx(self.grid, x)
        h_ci = neighbor_func(bmu_idx, self.m, self.n, self.sigma)
        update_weights(self.grid, self.alpha, h_ci, x)


# ══════════════════════════════════════════════════════════════════════════════
# Cluster (Data Generator)
# ══════════════════════════════════════════════════════════════════════════════

class LabelCluster:
    def __init__(self, idx, name, center, sigma, radius, color):
        self.idx    = idx
        self.name   = name
        self.center = np.array(center, dtype=float)
        self.sigma  = sigma
        self.radius = radius
        self.color  = color
        self.rgba   = np.array(to_rgba(color), dtype=float)
        
        # Drift velocity vector
        self.drift_vec = np.random.randn(2) * 0.0005
        
        # UI Elements
        self.patch_fill = None
        self.patch_edge = None
        self.text = None

    def membership(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X - self.center, axis=1) <= self.radius

    def sample(self, n: int) -> np.ndarray:
        cov = self.sigma**2 * np.eye(2)
        return np.random.multivariate_normal(self.center, cov, n)

    def update_drift(self):
        """Aplica concept drift movendo o centro sutilmente."""
        self.center += self.drift_vec
        # Rebate nas bordas se passar de 5% a 95% do espaço
        for i in range(2):
            if self.center[i] < 0.05 or self.center[i] > 0.95:
                self.drift_vec[i] *= -1

    def update_visuals(self):
        """Atualiza a posição dos patches na animação."""
        if self.patch_fill:
            self.patch_fill.center = self.center
        if self.patch_edge:
            self.patch_edge.center = self.center
        if self.text:
            self.text.set_position((self.center[0], self.center[1] + self.radius + 0.02))


# ══════════════════════════════════════════════════════════════════════════════
# Interactive App
# ══════════════════════════════════════════════════════════════════════════════

class StreamApp:
    MAX_POINTS   = 900
    INIT_SPEED   = 8

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.clusters : list[LabelCluster] = []
        # Para suportar clusters sem SOM, usaremos um dicionário {cluster_idx: SOMMap}
        self.soms : dict = {}
        
        self.paused   = False
        self.speed    = self.INIT_SPEED
        self.t        = 0
        
        self.new_diagnoses = [
            ("Bradicardia Severa", map_to_norm(45, 95), 0.08, 0.15),
            ("Choque Séptico", map_to_norm(140, 88), 0.10, 0.20),
            ("Embolia Pulmonar", map_to_norm(120, 82), 0.09, 0.18),
            ("Fibrilação Atrial", map_to_norm(160, 96), 0.12, 0.20),
            ("Depressão Respiratória", map_to_norm(70, 75), 0.10, 0.18),
            ("Arritmia Ventricular", map_to_norm(180, 85), 0.11, 0.19)
        ]

        self.buf : deque = deque(maxlen=self.MAX_POINTS)

        self._build_figure()

        # Adiciona as classes iniciais COM inicialização de SOM
        self._add_class("Taquicardia", map_to_norm(140, 96), 0.08, 0.18, init_som=True)
        self._add_class("Hipóxia", map_to_norm(80, 85), 0.09, 0.20, init_som=True)
        self._add_class("Parada Respiratória\nIminente", map_to_norm(160, 75), 0.08, 0.18, init_som=True)

        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=40, blit=False, cache_frame_data=False
        )

    # ── Figure ─────────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 9.5), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title('Integração SOM + Multi-label Stream')
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.08, 0.20, 0.88, 0.74])
        self.ax.set_xlim(-0.08, 1.08)
        self.ax.set_ylim(-0.08, 1.08)
        
        x_ticks = np.linspace(0, 1, 7)
        y_ticks = np.linspace(0, 1, 5)
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        self.ax.set_xticklabels([f"{int(HR_MIN + x*(HR_MAX-HR_MIN))}" for x in x_ticks])
        self.ax.set_yticklabels([f"{int(SPO2_MIN + y*(SPO2_MAX-SPO2_MIN))}" for y in y_ticks])
        
        self.ax.set_xlabel('Frequência Cardíaca (bpm)', color=TEXT_C, fontsize=11)
        self.ax.set_ylabel('Oxigenação do Sangue (SpO2 %)', color=TEXT_C, fontsize=11)

        self.ax.tick_params(labelsize=9, colors=MUTED_C)
        self.ax.grid(True, lw=0.5, alpha=0.5)
        self.ax.set_facecolor(PANEL_BG)

        self._title = self.ax.set_title(
            't = 0  |  0 classes  |  > gerando',
            fontsize=13, color=TEXT_C, pad=10, fontweight='bold'
        )

        self._sc = self.ax.scatter([], [], s=30, linewidths=0.6, zorder=5)
        
        # Para desenhar as malhas SOM (linhas horizontais e verticais de cada SOM)
        # Agora é um dicionário {cluster_idx: list_of_lines}
        self._som_lines = {}

        self.ax_pause = self.fig.add_axes([0.08, 0.065, 0.15, 0.075])
        self.btn_pause = mwidgets.Button(self.ax_pause, '|| Pausar', color=BTN_C, hovercolor=BTN_H)
        self.btn_pause.label.set_color(TEXT_C)
        self.btn_pause.label.set_fontsize(11)
        self.btn_pause.on_clicked(self._on_pause)

        self.ax_new = self.fig.add_axes([0.25, 0.065, 0.20, 0.075])
        self.btn_new = mwidgets.Button(self.ax_new, '+ Nova Classe', color=BTN_C, hovercolor=BTN_H)
        self.btn_new.label.set_color('#4ADE80')
        self.btn_new.label.set_fontsize(11)
        self.btn_new.on_clicked(self._on_new_class)

        ax_sl = self.fig.add_axes([0.55, 0.08, 0.35, 0.040])
        ax_sl.set_facecolor(PANEL_BG)
        self.slider = mwidgets.Slider(
            ax_sl, 'Velocidade', 1, 30,
            valinit=self.INIT_SPEED, valstep=1, color='#5B9CF6'
        )
        self.slider.label.set_color(TEXT_C)
        self.slider.label.set_fontsize(10)
        self.slider.valtext.set_color('#5B9CF6')
        self.slider.on_changed(lambda v: setattr(self, 'speed', int(v)))

        self._legend = None

    # ── Add Class and SOM ─────────────────────────────────────────────────────

    def _add_class(self, name=None, center=None, sigma=None, radius=None, init_som=True):
        if len(self.clusters) >= len(PALETTE):
            print("Número máximo de classes atingido!")
            return
            
        idx    = len(self.clusters)
        color  = PALETTE[idx]
        
        if name is None:
            if not self.new_diagnoses:
                name = f"Doença Nova {idx+1}"
                center = np.random.uniform(0.12, 0.88, 2)
                sigma = np.random.uniform(0.06, 0.12)
                radius = np.random.uniform(0.15, 0.25)
            else:
                name, center, sigma, radius = self.new_diagnoses.pop(0)

        # Cria gerador
        c = LabelCluster(idx, name, center, sigma, radius, color)
        self.clusters.append(c)

        if init_som:
            # Inicia rede SOM de tamanho 4x4 no centro da tela para visualizarmos a migração
            som = SOMMap(idx, m=4, n=4, input_dim=2, initial_center=np.array([0.5, 0.5]))
            self.soms[idx] = som
            
            # Cria linhas vazias para desenhar a malha desse SOM
            color_rgba = to_rgba(color)
            mesh_lines = []
            for _ in range(som.m):
                line, = self.ax.plot([], [], color=color_rgba, lw=1.2, alpha=0.7, zorder=8, marker='s', markersize=3)
                mesh_lines.append(line)
            for _ in range(som.n):
                line, = self.ax.plot([], [], color=color_rgba, lw=1.2, alpha=0.7, zorder=8, marker='s', markersize=3)
                mesh_lines.append(line)
            self._som_lines[idx] = mesh_lines

        c.patch_fill = plt.Circle(center, radius, color=color, fill=True, alpha=0.08, zorder=1)
        c.patch_edge = plt.Circle(center, radius, color=color, fill=False, alpha=0.4, lw=1.5, zorder=2)
        self.ax.add_patch(c.patch_fill)
        self.ax.add_patch(c.patch_edge)
        
        c.text = self.ax.text(
            center[0], center[1] + radius + 0.02, c.name,
            ha='center', va='bottom', fontsize=9,
            color=color, fontweight='bold', zorder=7
        )
        self._rebuild_legend()
        self.fig.canvas.draw_idle()

    # ── Controls ─────────────────────────────────────────────────────────────

    def _on_pause(self, event):
        self.paused = not self.paused
        self.btn_pause.label.set_text('> Retomar' if self.paused else '|| Pausar')
        self.fig.canvas.draw_idle()

    def _on_new_class(self, event):
        # Novas classes NÃO inicializam SOM, simulando o desafio de novelty detection.
        self._add_class(init_som=False)

    # ── Data generation and online SOM train ──────────────────────────────────

    def _generate_batch(self, n: int):
        # Atualiza a posição (concept drift) para todas as classes
        for c in self.clusters:
            c.update_drift()

        k      = len(self.clusters)
        counts = np.random.multinomial(n, [1.0 / k] * k)
        parts  = [c.sample(cnt) for c, cnt in zip(self.clusters, counts) if cnt > 0]
        X      = np.vstack(parts)
        np.random.shuffle(X)

        Y        = np.column_stack([c.membership(X) for c in self.clusters])
        n_active = Y.sum(axis=1)

        for i in range(len(X)):
            x  = X[i]
            y  = Y[i]
            na = int(n_active[i])

            # Processamento de Treinamento SOM (Sequencial Online)
            # Treina apenas se a classe possui um SOM associado
            for j in range(k):
                if y[j] and j in self.soms:
                    self.soms[j].online_train_step(x)

            # Preparação visual
            if na == 0:
                fc       = (0.53, 0.53, 0.60, 0.40)
                is_multi = False
            elif na == 1:
                j        = int(np.argmax(y))
                rgba     = self.clusters[j].rgba.copy()
                rgba[3]  = 0.82
                fc       = tuple(rgba)
                is_multi = False
            else:
                active  = np.array([self.clusters[j].rgba for j in range(k) if y[j]])
                blended = active.mean(axis=0)
                blended[3] = 0.88
                fc       = tuple(blended)
                is_multi = True

            self.buf.append((x[0], x[1], fc, is_multi))

        self.t += n

    # ── Animation frame ───────────────────────────────────────────────────────

    def _update_som_meshes(self):
        # Atualiza a posição das linhas e neurônios
        for idx, som in self.soms.items():
            grid = som.grid
            m, n, _ = grid.shape
            lines = self._som_lines[idx]
            
            # Linhas horizontais (fixando m)
            for i in range(m):
                xs = grid[i, :, 0]
                ys = grid[i, :, 1]
                lines[i].set_data(xs, ys)
            
            # Linhas verticais (fixando n)
            for j in range(n):
                xs = grid[:, j, 0]
                ys = grid[:, j, 1]
                lines[m + j].set_data(xs, ys)

    def _update(self, frame):
        if not self.paused and self.clusters:
            self._generate_batch(self.speed)

        # Update circle patches to reflect concept drift
        for c in self.clusters:
            c.update_visuals()

        n = len(self.buf)
        if n == 0:
            return

        arr      = list(self.buf)
        xs       = np.array([p[0] for p in arr])
        ys       = np.array([p[1] for p in arr])
        fcs      = [p[2] for p in arr]
        is_multi = [p[3] for p in arr]

        self._sc.set_offsets(np.column_stack([xs, ys]))
        self._sc.set_facecolor(fcs)
        self._sc.set_edgecolor([MULTI_EDGE if m else 'white' for m in is_multi])
        self._sc.set_linewidths([2.0 if m else 0.4 for m in is_multi])

        n_pts  = len(arr)
        alphas = np.linspace(0.25, 1.0, n_pts)
        fcs_alpha = []
        for idx_pt, (fc, a) in enumerate(zip(fcs, alphas)):
            r, g, b, _ = fc
            fcs_alpha.append((r, g, b, _ * a))
        self._sc.set_facecolor(fcs_alpha)

        # Atualiza o desenho das malhas
        self._update_som_meshes()

        state = '|| pausado' if self.paused else '> gerando'
        self._title.set_text(
            f't = {self.t:,}  |  {len(self.clusters)} classes  |  {state}'
        )

    # ── Legend ────────────────────────────────────────────────────────────────

    def _rebuild_legend(self):
        handles = [mpatches.Patch(color=c.color, label=c.name) for c in self.clusters]
        handles += [
            mpatches.Patch(color=NOISE_C, alpha=0.5, label='Saudável / Sem Diagnóstico'),
            mpatches.Patch(facecolor='none', edgecolor=MULTI_EDGE, linewidth=2.5, label='Comorbidade (Multi-label)'),
        ]
        if self._legend:
            self._legend.remove()
        self._legend = self.ax.legend(
            handles=handles, fontsize=9, framealpha=0.22,
            loc='upper right', facecolor=PANEL_BG,
            edgecolor=GRID_C, labelcolor=TEXT_C
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        plt.show()

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = StreamApp(seed=42)
    app.run()
