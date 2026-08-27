import numpy as np
import pandas as pd
from simulate import generate_graph, run_one_replicate


def run_experiment(n_reps, ego, wave, union_hop,wave_number, n=200, sparsity=0.4, lam=1.0,
                    gamma=0.05, beta=0.1, seed_node=0,
                    alpha_level=0.1, base_seed=0, fixed_graph=True):
    """
    Run n_reps Monte Carlo replicates and return a tidy DataFrame of results.
    If fixed_graph=True, one true graph is generated once and conditioned on
    (measurement error / splits vary across reps).
    """
    if fixed_graph:
        A, pos = generate_graph(n, sparsity, lam, seed=base_seed)

    rows = []
    for r in range(n_reps):
        rep_seed = base_seed + r
        if not fixed_graph:
            A, pos = generate_graph(n, sparsity, lam, seed=rep_seed)

        try:
            res = run_one_replicate(
                A, pos, n, lam, gamma, beta,ego,wave,union_hop, wave_number, seed_node,
                alpha_level=alpha_level, seed=rep_seed
            )
        except Exception as e:
            print(f"rep {r} failed: {e}")
            continue

        rows.append({
            "rep": r,
            "gamma": gamma, "beta": beta,
            "q_true": res["q_true"], "q_noisy": res["q_noisy"],
            "covered_true": res["covered_true"],
            "covered_noisy": res["covered_noisy"],
            "width_true": res["width_true"], "width_noisy": res["width_noisy"],
        })

    return pd.DataFrame(rows)


def run_grid(n_reps, gammas, betas, ego, wave, union_hop, wave_number,
             n=200, seed_node=50, alpha_level=0.1, base_seed=1008, fixed_graph=True):
    """Sweep over (gamma, beta) measurement-error grid, n_reps per cell."""
    all_results = []
    for g in gammas:
        for b in betas:
            df = run_experiment(n_reps,ego, wave, union_hop,wave_number, gamma=g, beta=b, n=n, seed_node=seed_node,
                                alpha_level=alpha_level, base_seed=base_seed, fixed_graph=fixed_graph)
            all_results.append(df)
    full = pd.concat(all_results, ignore_index=True)

    def summarize(group):
        return pd.Series({
            "coverage_true": np.mean(np.concatenate(group["covered_true"].values)),
            "coverage_noisy": np.mean(np.concatenate(group["covered_noisy"].values)),
            "width_true": group["width_true"].mean(),
            "width_noisy": group["width_noisy"].mean(),
            "n": len(group),
        })

    summary = full.groupby(["gamma", "beta"]).apply(summarize).reset_index()
    return full, summary


# if __name__ == "__main__":
#     alphas = [0.0, 0.05, 0.1, 0.2]
#     betas = [0.0, 0.05, 0.1, 0.2]
#
#     full, summary = run_grid(
#         n_reps=200, alphas=alphas, betas=betas,
#         n=200, seed_node=0, alpha_level=0.1, base_seed=42, fixed_graph=True
#     )
#
#     print(summary.to_string(index=False))
#     full.to_csv("cp_replicates.csv", index=False)
#     summary.to_csv("cp_summary.csv", index=False)