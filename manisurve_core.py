import numpy as np

def solve_manisurve(n, edges, k_colors=50, max_steps=1000):
    embed_dim = k_colors + 10
    V = np.random.randn(n, embed_dim).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    
    beams = np.eye(k_colors, embed_dim).astype(np.float32)
    
    u_nodes = edges[:, 0]
    v_nodes = edges[:, 1]

    for step in range(max_steps):
        scores = V @ beams.T
        labels = np.argmax(scores, axis=1)
        
        mask = (labels[u_nodes] == labels[v_nodes])
        violation_edges = edges[mask]
        num_v = len(violation_edges)
        
        if num_v == 0:
            return labels, step + 1

        lr = max(0.01, 0.2 * (1.0 - step / max_steps))
        
        for u, v in violation_edges[:500]:
            diff = V[u] - V[v]
            V[u] += lr * diff
            V[v] -= lr * diff
            
            c = labels[u]
            beams[c] -= (lr * 0.1) * (V[u] + V[v])
            beams[c] /= np.linalg.norm(beams[c])

        V /= np.linalg.norm(V, axis=1, keepdims=True)
        
    return labels, max_steps
