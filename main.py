# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#
import numpy as np
import simulate
from sklearn.linear_model import LinearRegression

# n=20
# sparsity = .15
# lam = 1
# A, latent_node_pos = simulate.generate_graph(n,sparsity,lam,10)
# X,Z,Y= simulate.generate_data(A,latent_node_pos)
# print("X")
# print(X)
# print("Z")
# print(Z)
# print("Y")
# print(Y)
# print(A)
#
# sample_idx = simulate.union_hop_sample(A,[1,4],2,False)
# print("sample_idx")
# print(sample_idx)


# print(sample_idx)
# X_sample= X[sample_idx]
# print(X_sample)
# Z_sample= Z[sample_idx]
# print(Z_sample)
# X_Z_sample= np.array([X_sample,Z_sample])
# print(np.shape(X_Z_sample))
# print(np.column_stack([X_sample,Z_sample]))


from run_experiment import run_grid

full, summary = run_grid(
    n_reps=1000,
    gammas=[0.0,0.01,0.025, 0.05, 0.1, 0.2],
    betas=[0.0,0.01,0.025, 0.05, 0.1, 0.2],
    ego=False,
    wave=False,
    union_hop=True,
    wave_number=2,
    n=200, seed_node=50, alpha_level=0.1, base_seed=1, fixed_graph=True
)

print(summary)
full.to_csv("cp_replicates.csv", index=False)
summary.to_csv("cp_summary.csv", index=False)




