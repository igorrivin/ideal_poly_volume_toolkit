"""
Hierarchical RL-guided optimizer for ideal polytope volume maximization.

The RL controller allocates search effort across regions of (S^{d-1})^N,
while classical optimizers (CMA-ES, LBFGS) handle local exploitation.

Architecture:
  - Gymnasium environment wrapping the volume optimization problem
  - Search tree partitioning the configuration space
  - Node-level features for the RL policy
  - Actions: sample, CMA-ES, LBFGS, split, perturb, snap-symmetric, prune

The task family: sample (N, d) and optimize ideal polytope volume.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import ConvexHull


# =====================================================================
# Region tree
# =====================================================================

@dataclass
class RegionNode:
    """A region of the configuration space (S^{d-1})^N.

    Rather than axis-aligned boxes in R^{Nd} (which don't respect the
    sphere constraint), each region is represented by a center
    configuration + radius on (S^{d-1})^N, measured in chordal distance.
    """
    center: np.ndarray          # (N, d) — center configuration on S^{d-1}
    radius: float               # search radius (chordal)
    depth: int = 0
    parent: Optional['RegionNode'] = None
    children: List['RegionNode'] = field(default_factory=list)

    # Statistics
    n_evals: int = 0
    best_value: float = -np.inf
    best_config: Optional[np.ndarray] = None
    mean_value: float = 0.0
    var_value: float = 0.0
    recent_improvement: float = 0.0
    values_history: List[float] = field(default_factory=list)

    # Local optimizer state
    local_state: Optional[Dict] = None
    is_active: bool = True

    def features(self) -> np.ndarray:
        """Encode this node as a feature vector for the RL policy."""
        return np.array([
            self.depth,
            np.log1p(self.n_evals),
            self.best_value,
            self.mean_value,
            self.var_value,
            self.recent_improvement,
            self.radius,
            1.0 if self.local_state is not None else 0.0,
        ], dtype=np.float32)

    def update_stats(self, values: List[float]):
        """Update running statistics from new evaluations."""
        old_best = self.best_value
        self.values_history.extend(values)
        self.n_evals += len(values)
        self.best_value = max(self.best_value, max(values))
        arr = np.array(self.values_history)
        self.mean_value = float(arr.mean())
        self.var_value = float(arr.var()) if len(arr) > 1 else 0.0
        self.recent_improvement = float(self.best_value - old_best)


NODE_FEATURE_DIM = 8
MAX_FRONTIER = 16  # max active nodes tracked by RL


# =====================================================================
# Volume evaluation helpers
# =====================================================================

def euclidean_volume(pts: np.ndarray) -> float:
    """Euclidean convex hull volume of points on S^{d-1}."""
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def random_on_sphere(N: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample N random points on S^{d-1}."""
    pts = rng.standard_normal((N, d))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def perturb_on_sphere(pts: np.ndarray, radius: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Perturb points on S^{d-1} by approximately `radius` in chordal distance."""
    noise = rng.standard_normal(pts.shape) * radius
    pts_new = pts + noise
    return pts_new / np.linalg.norm(pts_new, axis=1, keepdims=True)


# =====================================================================
# Local optimizers
# =====================================================================

class LocalCMAES:
    """CMA-ES optimizer restricted to a region around a center on S^{d-1}."""

    def __init__(self):
        try:
            import cma
            self.cma = cma
        except ImportError:
            raise ImportError("CMA-ES requires the 'cma' package")

    def run(self, center: np.ndarray, objective: Callable,
            budget: int = 8, sigma: float = 0.2,
            seed: int = 0) -> Tuple[np.ndarray, List[float]]:
        """Run CMA-ES for `budget` evaluations starting from center.

        Returns (best_config, list_of_values).
        """
        N, d = center.shape

        def neg_obj(x):
            pts = x.reshape(N, d)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            return -objective(pts)

        es = self.cma.CMAEvolutionStrategy(
            center.ravel(), sigma,
            {'maxfevals': budget, 'verbose': -9, 'seed': seed}
        )
        values = []
        while not es.stop():
            solutions = es.ask()
            fitnesses = [neg_obj(x) for x in solutions]
            es.tell(solutions, fitnesses)
            values.extend([-f for f in fitnesses])

        best_x = es.result.xbest.reshape(N, d)
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
        return best_x, values


class LocalLBFGS:
    """Torch LBFGS optimizer using exact hyperbolic volume gradients."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def run(self, center: np.ndarray, simplices,
            steps: int = 5) -> Tuple[np.ndarray, float]:
        """Run LBFGS for `steps` on hyperbolic volume.

        Returns (optimized_config, final_volume).
        """
        import torch

        if self.dimension == 6:
            from .murakami_torch import ideal_h6_polytope_volume
            from .li2_torch import li2 as li2_pure

            pts_opt = torch.tensor(center, dtype=torch.float64, requires_grad=True)
            simps_t = torch.tensor(simplices, dtype=torch.long)
            opt = torch.optim.LBFGS(
                [pts_opt], lr=0.01, max_iter=3,
                line_search_fn='strong_wolfe'
            )
            for _ in range(steps):
                def closure():
                    opt.zero_grad()
                    normed = pts_opt / torch.norm(pts_opt, dim=1, keepdim=True)
                    vol = ideal_h6_polytope_volume(normed, simps_t, li2_fn=li2_pure)
                    (-vol).backward()
                    return -vol
                opt.step(closure)
                with torch.no_grad():
                    pts_opt.data /= torch.norm(pts_opt.data, dim=1, keepdim=True)

            final = pts_opt.detach().numpy()
            final /= np.linalg.norm(final, axis=1, keepdims=True)
            with torch.no_grad():
                vol = ideal_h6_polytope_volume(
                    torch.tensor(final, dtype=torch.float64),
                    simps_t, li2_fn=li2_pure
                ).item()
            return final, vol

        elif self.dimension == 5:
            from .h5_volume_torch import schlafli_surrogate_vectorized
            import torch

            pts_opt = torch.tensor(center, dtype=torch.float64, requires_grad=True)
            simps_t = torch.tensor(simplices, dtype=torch.long)
            opt = torch.optim.LBFGS(
                [pts_opt], lr=0.01, max_iter=3,
                line_search_fn='strong_wolfe'
            )
            for _ in range(steps):
                def closure():
                    opt.zero_grad()
                    normed = pts_opt / torch.norm(pts_opt, dim=1, keepdim=True)
                    s = schlafli_surrogate_vectorized(normed, simps_t)
                    (-s).backward()
                    return -s
                opt.step(closure)
                with torch.no_grad():
                    pts_opt.data /= torch.norm(pts_opt.data, dim=1, keepdim=True)

            final = pts_opt.detach().numpy()
            final /= np.linalg.norm(final, axis=1, keepdims=True)
            return final, euclidean_volume(final)

        elif self.dimension == 4:
            # d=4: use scipy Powell on H^4 volume
            from .geometry_4d import ideal_polytope_volume_4d
            from scipy.optimize import minimize

            def neg_h4(x):
                pts = x.reshape(center.shape)
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
                try:
                    vol, _ = ideal_polytope_volume_4d(pts)
                    return -vol if np.isfinite(vol) else 0.0
                except Exception:
                    return 0.0

            res = minimize(neg_h4, center.ravel(), method='Powell',
                           options={'maxiter': steps * 50, 'maxfev': steps * 20})
            final = res.x.reshape(center.shape)
            final /= np.linalg.norm(final, axis=1, keepdims=True)
            vol, _ = ideal_polytope_volume_4d(final)
            return final, vol

        elif self.dimension == 3:
            # d=3: direct H^3 volume via Bloch-Wigner + scipy L-BFGS-B
            from scipy.special import spence
            from scipy.spatial import Delaunay
            from scipy.optimize import minimize

            def h3_vol(pts_S2):
                N = len(pts_S2)
                north_dist = [abs(p[2] - 1.0) for p in pts_S2]
                inf_idx = int(np.argmin(north_dist))
                z_arr = []
                for i in range(N):
                    if i == inf_idx: continue
                    p = pts_S2[i]
                    d = max(1.0 - p[2], 1e-12)
                    z_arr.append(complex(p[0]/d, p[1]/d))
                z_arr = np.array(z_arr)
                pts_2d = np.column_stack([z_arr.real, z_arr.imag])
                try:
                    tri = Delaunay(pts_2d)
                except: return 0.0
                total = 0.0
                for sx in tri.simplices:
                    z1, z2, z3 = z_arr[sx[0]], z_arr[sx[1]], z_arr[sx[2]]
                    cr = (z1-z3)/(z2-z3) if abs(z2-z3) > 1e-15 else 0.0
                    li2 = spence(1.0 - cr)
                    total += abs(li2.imag + np.angle(1.0-cr)*np.log(max(abs(cr),1e-300)))
                return total

            def neg_h3(x):
                pts = x.reshape(center.shape)
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
                try: return -h3_vol(pts)
                except: return 0.0

            res = minimize(neg_h3, center.ravel(), method='Powell',
                           options={'maxiter': steps * 50, 'maxfev': steps * 30})
            final = res.x.reshape(center.shape)
            final /= np.linalg.norm(final, axis=1, keepdims=True)
            vol = h3_vol(final)
            return final, vol

        else:
            return center, euclidean_volume(center)


# =====================================================================
# Triangulation helper
# =====================================================================

def triangulate(pts: np.ndarray) -> list:
    """Stereographic Delaunay + cone from vertex 0."""
    from scipy.spatial import Delaunay
    N, d = pts.shape
    pole = pts[0]
    others = np.delete(pts, 0, axis=0)
    basis = []
    for i in range(d):
        e = np.zeros(d); e[i] = 1.0
        e -= np.dot(e, pole) * pole
        for b in basis:
            e -= np.dot(e, b) * b
        n = np.linalg.norm(e)
        if n > 1e-10:
            basis.append(e / n)
        if len(basis) == d - 1:
            break
    basis = np.array(basis)
    proj = []
    for p in others:
        dd = max(1 - np.dot(p, pole), 1e-14)
        proj.append(basis @ ((p - np.dot(p, pole) * pole) / dd))
    proj = np.array(proj)
    tri = Delaunay(proj)
    return [[0] + [s + 1 for s in simp] for simp in tri.simplices]


# =====================================================================
# Gymnasium Environment
# =====================================================================

# Action types
ACT_SAMPLE = 0       # Random sampling in region
ACT_CMA_EUCL = 1     # CMA-ES on Euclidean volume
ACT_LBFGS_HYP = 2    # LBFGS on hyperbolic volume (exact gradients)
ACT_SPLIT = 3         # Split region into children
ACT_PERTURB = 4       # Random perturbation + retriangulate
ACT_PRUNE = 5         # Remove region from frontier
NUM_ACTIONS = 6


class IdealPolytopeOptEnv(gym.Env):
    """
    Gymnasium environment for RL-guided ideal polytope volume optimization.

    Observation: (MAX_FRONTIER, NODE_FEATURE_DIM + task_meta_dim) matrix
    Action: (node_index, operation_type)
    Reward: improvement in best-so-far volume minus evaluation cost
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        d_range: Tuple[int, int] = (4, 6),
        N_range: Tuple[int, int] = (7, 25),
        horizon: int = 50,
        eval_cost_weight: float = 0.001,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.d_range = d_range
        self.N_range = N_range
        self.horizon = horizon
        self.eval_cost_weight = eval_cost_weight
        self._rng = np.random.default_rng(seed)

        # Task metadata: [d, N, log(N*d)]
        task_meta_dim = 3
        obs_dim = MAX_FRONTIER * (NODE_FEATURE_DIM + task_meta_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([MAX_FRONTIER, NUM_ACTIONS])

        self._cma = LocalCMAES()
        self._lbfgs = {}  # dimension -> LocalLBFGS

    def _get_lbfgs(self, d: int) -> LocalLBFGS:
        if d not in self._lbfgs:
            self._lbfgs[d] = LocalLBFGS(d)
        return self._lbfgs[d]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample task
        self.d = int(self._rng.integers(self.d_range[0], self.d_range[1] + 1))
        self.N = int(self._rng.integers(
            self.N_range[0],
            min(self.N_range[1], 5 * self.d) + 1
        ))
        self.task_meta = np.array([
            self.d, self.N, np.log(self.N * self.d)
        ], dtype=np.float32)

        # Initial configuration: random points on S^{d-1}
        init_pts = random_on_sphere(self.N, self.d, self._rng)

        # Root node
        self.root = RegionNode(center=init_pts, radius=2.0, depth=0)
        self.frontier = [self.root]

        # Best so far
        self.best_value = euclidean_volume(init_pts)
        self.best_config = init_pts.copy()
        self.root.best_value = self.best_value
        self.root.best_config = init_pts.copy()
        self.root.n_evals = 1
        self.root.values_history = [self.best_value]

        self.total_evals = 1
        self.step_count = 0

        return self._encode_obs(), self._info()

    def step(self, action):
        node_idx, op_type = action[0], action[1]

        # Validate node index
        if node_idx >= len(self.frontier):
            node_idx = len(self.frontier) - 1
        node = self.frontier[node_idx]

        old_best = self.best_value
        evals_used = 0

        try:
            if op_type == ACT_SAMPLE:
                evals_used = self._do_sample(node, n=4)

            elif op_type == ACT_CMA_EUCL:
                evals_used = self._do_cma(node, budget=12)

            elif op_type == ACT_LBFGS_HYP:
                evals_used = self._do_lbfgs(node, steps=3)

            elif op_type == ACT_SPLIT:
                self._do_split(node)

            elif op_type == ACT_PERTURB:
                evals_used = self._do_perturb(node)

            elif op_type == ACT_PRUNE:
                self._do_prune(node)

        except Exception:
            pass  # robustness: failed actions are no-ops

        # Update global best
        for n in self.frontier:
            if n.best_value > self.best_value and n.best_config is not None:
                self.best_value = n.best_value
                self.best_config = n.best_config.copy()

        self.total_evals += evals_used
        self.step_count += 1

        # Reward
        improvement = max(0.0, self.best_value - old_best)
        cost = self.eval_cost_weight * evals_used
        reward = float(improvement - cost)

        terminated = self.step_count >= self.horizon
        truncated = False

        return self._encode_obs(), reward, terminated, truncated, self._info()

    # --- Action implementations ---

    def _do_sample(self, node: RegionNode, n: int = 4) -> int:
        """Sample n random configurations near the node center."""
        values = []
        for _ in range(n):
            pts = perturb_on_sphere(node.center, node.radius * 0.3, self._rng)
            v = euclidean_volume(pts)
            values.append(v)
            if v > node.best_value:
                node.best_value = v
                node.best_config = pts.copy()
        node.update_stats(values)
        return n

    def _do_cma(self, node: RegionNode, budget: int = 12) -> int:
        """Run CMA-ES on Euclidean volume from node center."""
        best_config, values = self._cma.run(
            node.center if node.best_config is None else node.best_config,
            euclidean_volume,
            budget=budget,
            sigma=node.radius * 0.2,
            seed=int(self._rng.integers(0, 2**31)),
        )
        node.update_stats(values)
        best_v = euclidean_volume(best_config)
        if best_v > node.best_value:
            node.best_value = best_v
            node.best_config = best_config.copy()
            node.center = best_config.copy()
        return len(values)

    def _do_lbfgs(self, node: RegionNode, steps: int = 3) -> int:
        """Run LBFGS on hyperbolic volume with exact gradients."""
        if node.best_config is None:
            return 0
        simps = triangulate(node.best_config)
        lbfgs = self._get_lbfgs(self.d)
        try:
            final, vol = lbfgs.run(node.best_config, simps, steps=steps)
            node.update_stats([vol])
            if vol > node.best_value:
                node.best_value = vol
                node.best_config = final.copy()
                node.center = final.copy()
        except Exception:
            pass
        return steps  # approximate eval count

    def _do_split(self, node: RegionNode):
        """Split node into two children at smaller radius."""
        if len(self.frontier) >= MAX_FRONTIER - 1:
            return

        child_radius = node.radius * 0.6

        # Child 1: perturb center in random direction
        offset = self._rng.standard_normal(node.center.shape) * child_radius * 0.3
        c1 = node.center + offset
        c1 /= np.linalg.norm(c1, axis=1, keepdims=True)

        # Child 2: perturb in opposite direction
        c2 = node.center - offset
        c2 /= np.linalg.norm(c2, axis=1, keepdims=True)

        child1 = RegionNode(c1, child_radius, depth=node.depth + 1, parent=node)
        child2 = RegionNode(c2, child_radius, depth=node.depth + 1, parent=node)

        # Inherit best from parent
        if node.best_config is not None:
            child1.best_config = node.best_config.copy()
            child1.best_value = node.best_value
            child2.best_config = node.best_config.copy()
            child2.best_value = node.best_value

        node.children = [child1, child2]
        node.is_active = False
        if node in self.frontier:
            self.frontier.remove(node)
        self.frontier.extend([child1, child2])

    def _do_perturb(self, node: RegionNode) -> int:
        """Large perturbation to escape local basin."""
        if node.best_config is None:
            return 0
        pts = perturb_on_sphere(node.best_config, 0.3, self._rng)
        v = euclidean_volume(pts)
        node.update_stats([v])
        if v > node.best_value:
            node.best_value = v
            node.best_config = pts.copy()
            node.center = pts.copy()
        return 1

    def _do_prune(self, node: RegionNode):
        """Remove node from frontier."""
        if len(self.frontier) > 1 and node in self.frontier:
            self.frontier.remove(node)
            node.is_active = False

    # --- Observation encoding ---

    def _encode_obs(self) -> np.ndarray:
        """Encode frontier as a flat 1D vector (SB3-compatible)."""
        per_node = NODE_FEATURE_DIM + len(self.task_meta)
        obs = np.zeros(MAX_FRONTIER * per_node, dtype=np.float32)
        for i, node in enumerate(self.frontier[:MAX_FRONTIER]):
            features = node.features()
            offset = i * per_node
            obs[offset:offset + NODE_FEATURE_DIM] = features
            obs[offset + NODE_FEATURE_DIM:offset + per_node] = self.task_meta
        return obs

    def _info(self) -> dict:
        return {
            'best_value': self.best_value,
            'total_evals': self.total_evals,
            'n_frontier': len(self.frontier),
            'd': self.d,
            'N': self.N,
        }


# =====================================================================
# Training utilities
# =====================================================================

def train_rl_controller(
    total_timesteps: int = 500_000,
    d_range: Tuple[int, int] = (4, 6),
    N_range: Tuple[int, int] = (7, 20),
    horizon: int = 40,
    seed: int = 42,
    verbose: int = 1,
):
    """Train the RL controller using SB3 PPO.

    Returns the trained model.
    """
    from stable_baselines3 import PPO

    env = IdealPolytopeOptEnv(
        d_range=d_range,
        N_range=N_range,
        horizon=horizon,
        seed=seed,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=verbose,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        seed=seed,
        device='cpu',
    )

    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_controller(model, d: int, N: int, n_episodes: int = 10,
                         horizon: int = 50, seed: int = 0):
    """Evaluate a trained controller on a specific (d, N) task.

    Returns dict with statistics.
    """
    env = IdealPolytopeOptEnv(
        d_range=(d, d),
        N_range=(N, N),
        horizon=horizon,
        seed=seed,
    )

    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append({
            'best_value': info['best_value'],
            'total_evals': info['total_evals'],
            'total_reward': total_reward,
        })

    values = [r['best_value'] for r in results]
    return {
        'mean_value': np.mean(values),
        'std_value': np.std(values),
        'max_value': np.max(values),
        'mean_evals': np.mean([r['total_evals'] for r in results]),
        'results': results,
    }


def compare_with_baseline(d: int, N: int, n_trials: int = 10,
                           budget_per_trial: int = 500, seed: int = 0):
    """Compare RL controller against pure CMA-ES baseline.

    Returns dict with both results.
    """
    rng = np.random.default_rng(seed)

    # Baseline: pure CMA-ES
    import cma
    baseline_values = []
    for trial in range(n_trials):
        pts = random_on_sphere(N, d, np.random.default_rng(seed + trial))
        def neg_vol(x):
            p = x.reshape(N, d)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            try: return -ConvexHull(p).volume
            except: return 0.0

        es = cma.CMAEvolutionStrategy(pts.ravel(), 0.3, {
            'maxfevals': budget_per_trial, 'verbose': -9,
            'seed': seed + trial,
        })
        while not es.stop():
            sols = es.ask()
            fits = [neg_vol(x) for x in sols]
            es.tell(sols, fits)
        baseline_values.append(-es.result.fbest)

    return {
        'baseline_mean': np.mean(baseline_values),
        'baseline_std': np.std(baseline_values),
        'baseline_max': np.max(baseline_values),
    }
