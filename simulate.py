import numpy as np
import pandas as pd
import sklearn
import matplotlib as plt
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
    U = rng.uniform(0, 1, n)

    # Graphon probabilities
    dist = np.abs(U[:, None] - U[None, :])
    P = sparsity * np.exp(-lam * dist)

    # No self-loops
    np.fill_diagonal(P, 0)

    # Generate symmetric adjacency matrix
    A = np.zeros((n, n), dtype=int)
    upper = rng.binomial(1, P)
    upper = np.triu(upper, k=1)
    A = upper + upper.T

    return A, U


def generate_data(A, U, beta_x=1.0, beta_z=2.0,
                  noise_x=0.5, noise_y=1.0, seed=None):

    rng = np.random.default_rng(seed)
    n = len(U)

    # Individual covariate
    X = U + rng.normal(0, noise_x, n)

    # True neighborhood average
    degree = A.sum(axis=1)
    Z = np.divide(
        A @ X,
        degree,
        out=np.zeros(n),
        where=degree > 0
    )

    # Response depends on BOTH individual and network information
    Y = (
        beta_x * X
        + beta_z * Z
        + rng.normal(0, noise_y, n)
    )

    return X, Z, Y

# ---------------------------------------------------------------------------
# 2. SAMPLING (INVARIANT SELECTOR)
# ---------------------------------------------------------------------------

#TODO: rewrite for non bipartite graph
def ego_sample_proteins(A, seed_protein_idx):
    """
    Ego sampling: given a seed protein, sample all compounds connected to it
    (its "ego network" on the true graph).

    This is applied to the TRUE adjacency matrix A, since in simulation we
    have full access to it (unlike in real life). This mirrors Lunde's
    invariant selector definition.

    Returns:
        compound_idx : indices of sampled compounds (neighbors of seed)
        protein_idx  : indices of sampled proteins (just the seed, for now;
                       extend to snowball sampling if needed)
    """
    compound_idx = np.where(A[:, seed_protein_idx] == 1)[0]
    protein_idx = np.array([seed_protein_idx])
    return compound_idx, protein_idx


# ---------------------------------------------------------------------------
# 3. MEASUREMENT ERROR MODEL (APPLIED ONLY WITHIN THE SAMPLE)
# ---------------------------------------------------------------------------

def apply_measurement_error(A_sample, alpha, beta, seed=None):
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
    flip_prob = np.where(A_sample == 1, beta, alpha)
    flips = rng.binomial(1, flip_prob)
    A_hat = np.where(flips == 1, 1 - A_sample, A_sample)
    return A_hat


# ---------------------------------------------------------------------------
# 4. NETWORK STATISTICS (ZETA FUNCTION)
# ---------------------------------------------------------------------------

def compute_degree_statistic(A_sub):
    """
    Simple network statistic: compound degree within the sampled subgraph.
    Replace/extend this with richer statistics (e.g. weighted neighbor
    averages) as needed -- this is your `zeta` function.

    Returns:
        Z : array of degree values, one per compound in the sample
    """
    return A_sub.sum(axis=1)


# ---------------------------------------------------------------------------
# 5. RESPONSE GENERATION
# ---------------------------------------------------------------------------

#TODO: delete
def generate_response(X, Z, noise_std=1.0, seed=None):
    """
    Generate a synthetic response Y as a function of covariates X and the
    TRUE network statistic Z, plus independent mean-zero noise.

    Y = mu*(X, Z) + epsilon

    Using a simple linear form here; swap in something more complex later.
    """
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(0, noise_std, size=len(X))
    Y = 1.0 * X + 0.5 * Z + epsilon
    return Y


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
    Run split conformal prediction given a fitted predictor mu_hat.

    Returns:
        q_hat    : calibration quantile
        covered  : bool, whether Y_test falls inside the prediction interval
        width    : interval width (2 * q_hat)
    """
    cal_preds = mu_hat(X_cal, Z_cal)
    residuals = np.abs(Y_cal - cal_preds)
    q_hat = split_conformal_quantile(residuals, alpha_level)

    test_pred = mu_hat(X_test, Z_test)
    covered = np.abs(Y_test - test_pred) <= q_hat
    width = 2 * q_hat

    return q_hat, covered, width


# ---------------------------------------------------------------------------
# 7. ONE FULL REPLICATE (TRUE vs NOISY PIPELINE)
# ---------------------------------------------------------------------------

def run_one_replicate(n_compounds, n_proteins, lam, alpha, beta,
                       seed_protein_idx, alpha_level=0.1, seed=None):
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
    rng_seed = seed

    # --- true graph (regenerate here only if NOT conditioning on fixed A) ---
    A, U, W = generate_bipartite_graphon(n_compounds, n_proteins, lam=lam, seed=rng_seed)

    # --- sample (invariant selector applied to TRUE graph) ---
    compound_idx, protein_idx = ego_sample_proteins(A, seed_protein_idx)
    A_sample = A[np.ix_(compound_idx, protein_idx)]

    # --- apply measurement error ONLY within the sample ---
    A_hat_sample = apply_measurement_error(A_sample, alpha, beta, seed=rng_seed)

    # --- true and noisy network statistics ---
    Z_true = compute_degree_statistic(A_sample)
    Z_hat = compute_degree_statistic(A_hat_sample)

    # --- covariates and response (uses TRUE Z, since Y depends on true network) ---
    X = U[compound_idx]
    Y = generate_response(X, Z_true, seed=rng_seed)

    # --- calibration / test split ---
    n_sample = len(compound_idx)
    idx = np.arange(n_sample)
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(idx)
    split = n_sample // 2
    cal_idx, test_idx = idx[:split], idx[split:]

    # --- fixed predictor mu* (fit once on true data conceptually;
    #     here just a placeholder linear function -- replace with sklearn model) ---
    def mu_star(x, z):
        return 1.0 * x + 0.5 * z

    q_true, covered_true, width_true = run_split_cp(
        mu_star, X[cal_idx], Y[cal_idx], Z_true[cal_idx],
        X[test_idx], Y[test_idx], Z_true[test_idx], alpha_level
    )

    q_noisy, covered_noisy, width_noisy = run_split_cp(
        mu_star, X[cal_idx], Y[cal_idx], Z_hat[cal_idx],
        X[test_idx], Y[test_idx], Z_hat[test_idx], alpha_level
    )

    return {
        "q_true": q_true, "q_noisy": q_noisy,
        "covered_true": covered_true, "covered_noisy": covered_noisy,
        "width_true": width_true, "width_noisy": width_noisy,
    }