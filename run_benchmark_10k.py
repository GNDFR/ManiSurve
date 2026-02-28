import networkx as nx
import numpy as np
import time
from manisurve_core import solve_manisurve

def run_10k_test():
    n = 10000
    p_edge = 0.001
    k = 50
    
    G = nx.fast_gnp_random_graph(n, p_edge, seed=42)
    edges = np.array(list(G.edges()))

    start_time = time.time()
    labels, steps = solve_manisurve(n, edges, k_colors=k)
    elapsed = time.time() - start_time

    print(f"Steps: {steps}")
    print(f"Time: {elapsed:.4f}s")
    
    violations = sum(1 for u, v in edges if labels[u] == labels[v])
    print(f"Violations: {violations}")
    
    if violations == 0:
        print("Validated.")

if __name__ == "__main__":
    run_10k_test()
