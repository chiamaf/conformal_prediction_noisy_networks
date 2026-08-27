import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import networkx as nx

"""
simulate.py

Core functions for simulating a bipartite compound-protein network with
measurement error, and running split conformal prediction on top of it.

Pipeline (per the sample-first-then-noise setup):
  1. Generate a true bipartite graph (bigraphon-based).
  2. Draw a sample from the true graph using an invariant selector
     (e.g. ego sampling around a seed protein).
  3. Apply Bernoulli measurement error ONLY within the sampled subgraph.
  4. Compute network statistics (Z) from both the true and noisy subgraphs.
  5. Fit a predictor, compute nonconformity scores, run split CP.
  6. Compare true vs noisy quantiles / coverage.
"""




# ---------------------------------------------------------------------------
# 1. TRUE GRAPH GENERATION
# ---------------------------------------------------------------------------
def generate_graph(n, sparsity=0.5, lam=1.0, seed=None):
    rng = np.random.default_rng(seed)

    # Latent node positions
    latent_node_pos = rng.uniform(0, 1, n)
    eta_s = rng.uniform(0,1,(n,n))
    w = 3.0 * np.abs(latent_node_pos[:, None] - latent_node_pos[None, :])
    threshold = np.minimum(sparsity * w, 1.0)

    A = (eta_s <= threshold).astype(int)
    np.fill_diagonal(A, 0)

    upper = np.triu(A,k=1)
    A = upper + upper.T

    return A, latent_node_pos


def generate_data(A,latent_node_pos, beta_0 = 0.0, beta_x=1.0, beta_z=2.0,
                  noise_x=1.0, noise_y=1, seed=None):

    rng = np.random.default_rng(seed)
    n = len(latent_node_pos)

    # Individual covariate
    X = latent_node_pos + rng.normal(0, noise_x, n)

    # True neighborhood average
    degree = A.sum(axis=1)
    Z = np.divide(
        A @ X, # neighborhood total
        degree,
        out=np.zeros(n),
        where=degree > 0
    )

    # Response depends on BOTH individual and network information
    Y = (
        beta_0 +
        beta_x * X
        + beta_z * Z
        + rng.normal(0, noise_y, n)
    )

    return X, Z, Y

# ---------------------------------------------------------------------------
# 2. SAMPLING (INVARIANT SELECTOR)
# ---------------------------------------------------------------------------

def ego_sample(A, ego_node):
    neighbors = np.nonzero(A[ego_node])
    return neighbors

# ---------------------------------------------------------------------------
# 2b. WAVE SAMPLING (exact hop-distance k, excludes all closer waves)
# ---------------------------------------------------------------------------

def wave_sample(A, seed_nodes, k):
    """
    Return node indices at EXACT hop-distance k from seed_nodes.
    wave 0 = seed_nodes themselves; wave k excludes waves 0..k-1.
    """
    if np.isscalar(seed_nodes):
        seed_nodes = [seed_nodes]

    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            neighbors = np.nonzero(A[node])[0]
            next_frontier.update(neighbors.tolist())
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    return sorted(frontier)


# ---------------------------------------------------------------------------
# 2c. UNION-OF-HOPS SAMPLING (cumulative ball of radius k)
# ---------------------------------------------------------------------------

def union_hop_sample(A, seed_nodes, k, include_seed=True):
    """
    Return node indices within hop-distance <= k from seed_nodes
    (union of waves 0 through k).
    """
    if np.isscalar(seed_nodes):
        seed_nodes = [seed_nodes]

    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            neighbors = np.nonzero(A[node])[0]
            next_frontier.update(neighbors.tolist())
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    if not include_seed:
        visited -= set(seed_nodes)

    return sorted(visited)


# ---------------------------------------------------------------------------
# 3. MEASUREMENT ERROR MODEL (APPLIED ONLY WITHIN THE SAMPLE)
# ---------------------------------------------------------------------------

def apply_measurement_error(A_sample, gamma, beta, seed=None):
    """
    Apply homogeneous Bernoulli measurement error to a SAMPLED subgraph.

    - True edges (A_ij = 1) are retained w.p. (1-beta), flipped to 0 w.p. beta
      (beta = Type II error / false negative rate)
    - True non-edges (A_ij = 0) are retained w.p. (1-alpha), flipped to 1
      w.p. alpha (alpha = Type I error / false positive rate)

    Returns:
        A_hat : noisy version of A_sample, same shape
    """
    rng = np.random.default_rng(seed)
    flip_prob = np.where(A_sample == 1, beta, gamma)
    flips = rng.binomial(1, flip_prob)
    A_hat = np.where(flips == 1, 1 - A_sample, A_sample)
    np.fill_diagonal(A_hat,0)
    return A_hat


# ---------------------------------------------------------------------------
# 4. NETWORK STATISTICS (ZETA FUNCTION)
# ---------------------------------------------------------------------------

def compute_node_degree(A_sub):
    return A_sub.sum(axis=1)

def compute_neighborhood_average_covariates(A_sub, X):
    degree = A_sub.sum(axis=1)
    n=len(X)
    Z = np.divide(
        A_sub @ X,  # neighborhood total
        degree,
        out=np.zeros(n),
        where=degree > 0
    )
    return Z


# ---------------------------------------------------------------------------
# 6. SPLIT CONFORMAL PREDICTION
# ---------------------------------------------------------------------------

def split_conformal_quantile(residuals, alpha_level):
    """
    Compute the (1-alpha_level)(1 + 1/n) empirical quantile of calibration
    residuals, per standard split conformal prediction.
    """
    n = len(residuals)
    q_level = np.ceil((1 - alpha_level) * (n + 1)) / n
    q_level = min(q_level, 1.0)  # clip in case of small n
    return np.quantile(residuals, q_level, method="higher")


def run_split_cp(mu_hat, X_cal, Y_cal, Z_cal, X_test, Y_test, Z_test, alpha_level=0.1):
    """
    Run split conformal prediction given fitted mu_hat. Uses absolute residuals.

    Returns:
        q_hat    : calibration quantile
        covered  : bool, whether Y_test falls inside the prediction interval
        width    : interval width (2 * q_hat)
    """
    cal_preds = mu_hat.predict(np.column_stack([X_cal, Z_cal]))
    residuals = np.abs(Y_cal - cal_preds)
    q_hat = split_conformal_quantile(residuals, alpha_level)

    test_pred = mu_hat.predict(np.column_stack([X_test, Z_test]))
    covered = np.abs(Y_test - test_pred) <= q_hat
    width = 2 * q_hat

    return q_hat, covered, width


# ---------------------------------------------------------------------------
# 7. ONE FULL REPLICATE (TRUE vs NOISY PIPELINE)
# ---------------------------------------------------------------------------

def run_one_replicate(A, latent_node_pos, n, lam, gamma, beta, ego, wave, union_hop, wave_number,
                       seed_nodes, alpha_level=0.1, seed=None,
                      train_frac = 0.45, cal_frac=0.3,
                      ):
    """
    Run a single Monte Carlo replicate:
      - fixed true graph (should be generated ONCE outside this function
        if you want to condition on a=fixed; see run_experiment.py)
      - one draw of measurement error
      - one calibration/test split
      - returns (q_true, q_noisy, covered_true, covered_noisy)

    NOTE: for the conditional-on-true-graph design discussed earlier,
    pass in a pre-generated A rather than regenerating it here.
    """
    rng=np.random.default_rng(seed)
    #Generate data
    beta_x, beta_z = rng.normal(0,1,2)

    X,Z,Y = generate_data(A, latent_node_pos,beta_x, beta_z, seed)

    # --- sample (invariant selector applied to TRUE graph) ---
    sample_idx =[]
    if ego:
        sample_idx = ego_sample(A, seed_nodes[0])[0].tolist()
    elif wave:
        sample_idx = wave_sample(A,seed_nodes,wave_number)
    elif union_hop:
        sample_idx = union_hop_sample(A, seed_nodes,wave_number, False)

    A_sample = A[np.ix_(sample_idx,sample_idx)]
    X_sample = X[sample_idx]
    Y_sample = Y[sample_idx]

    # --- apply measurement error ONLY within the sample ---
    A_hat_sample = apply_measurement_error(A_sample, gamma, beta, seed)

    #STATISTICS
    true_neighborhood_avg = compute_neighborhood_average_covariates(A_sample,X_sample)
    noisy_neighborhood_avg = compute_neighborhood_average_covariates(A_hat_sample,X_sample)

    true_degree = compute_node_degree(A_sample)
    noisy_degree = compute_node_degree(A_hat_sample)

    # --- train/cal/test split, within sample---
    n_sample = len(X_sample)
    idx = np.arange(n_sample)

    idx_train, idx_rest = train_test_split(idx, train_size=train_frac, random_state=seed)

    rel_cal_frac = cal_frac / (1-train_frac)
    cal_idx, test_idx = train_test_split(idx_rest, train_size=rel_cal_frac, random_state=seed)

    # mu 1: correctly specified model.
    mu_1_true = LinearRegression().fit(
        np.column_stack([X_sample[idx_train], true_neighborhood_avg[idx_train]]),
        Y_sample[idx_train]
    )
    mu_1_noisy = LinearRegression().fit(
        np.column_stack([X_sample[idx_train], noisy_neighborhood_avg[idx_train]]),
        Y_sample[idx_train]
    )

    # mu 2: incorrectly specified, uses degree as statistic instead
    mu_2_true = LinearRegression().fit(
        np.column_stack([X_sample[idx_train], true_degree[idx_train]]),
        Y_sample[idx_train]
    )
    mu_2_noisy = LinearRegression().fit(
        np.column_stack([X_sample[idx_train], noisy_degree[idx_train]]),
        Y_sample[idx_train]
    )

    q_true, covered_true, width_true = run_split_cp(
        mu_1_true, X_sample[cal_idx], Y_sample[cal_idx], true_neighborhood_avg[cal_idx],
        X_sample[test_idx], Y_sample[test_idx], true_neighborhood_avg[test_idx], alpha_level
    )

    q_noisy, covered_noisy, width_noisy = run_split_cp(
        mu_1_noisy, X_sample[cal_idx], Y_sample[cal_idx], noisy_neighborhood_avg[cal_idx],
        X_sample[test_idx], Y_sample[test_idx], noisy_neighborhood_avg[test_idx], alpha_level
    )

    return {
        "q_true": q_true, "q_noisy": q_noisy,
        "covered_true": covered_true, "covered_noisy": covered_noisy,
        "width_true": width_true, "width_noisy": width_noisy,
    }


