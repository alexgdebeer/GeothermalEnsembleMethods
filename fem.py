import numpy as np
from scipy import sparse

# Define gradients of basis functions
GRAD_2D = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

def generate_fem_matrices_2D(nodes, boundary_nodes, elements, L=None):
    """
    Generates matrices required for generating Matern fields using FEM.

    Parameters:
    ----------------
    nodes: np.array 
        Array of node locations. The ith column contains the coordinates of 
        node i.
    boundary_nodes: np.array
        Array of indices of nodes on mesh boundary.
    elements: np.array 
        Array of indices of nodes in each element. The ith column contains the 
        indices of the nodes of element i.
    L: np.array
        Array containing the squared lengthscale in each direction on the 
        diagonal (default: 1 in each direction).

    Returns:
    ----------------
    M: scipy.sparse.lil_matrix
        Mass matrix.
    K: scipy.sparse.lil_matrix
        Stiffness matrix.
    N: scipy.sparse.lil_matrix
        Inner products of each pair of basis functions over the mesh boundary
        (used when implementing Robin boundary conditions).
    """
    
    n_nodes = len(nodes)

    if L is None:
        L = np.eye(2)

    M = sparse.lil_matrix((n_nodes, n_nodes))
    K = sparse.lil_matrix((n_nodes, n_nodes))
    N = sparse.lil_matrix((n_nodes, n_nodes))

    for element in elements:
        nodes_e = nodes[element, :]
        
        for i in range(3):
            
            pi = element[i]

            # Define transformation to reference element
            T = np.array([nodes_e[(i+1)%3] - nodes_e[i],
                          nodes_e[(i+2)%3] - nodes_e[i]]).T
            
            detT = np.abs(np.linalg.det(T))
            invT = np.linalg.inv(T)

            for j in range(3):
                
                pj = element[j]
                
                if pi == pj:
                    M[pi, pj] += detT * 1/12
                else: 
                    M[pi, pj] += detT * 1/24
                
                K[pi, pj] += 0.5 * detT * \
                    GRAD_2D[0]@ L @ invT @ invT.T @ GRAD_2D[(j-i)%3].T
                
                # TODO: check whether this changes when lengthscale matrix used
                if set([pi, pj]).issubset(boundary_nodes) and pi != pj:
                    n = np.linalg.norm(nodes[pi] - nodes[pj])
                    N[pi, pi] += n * 1/3
                    N[pi, pj] += n * 1/6

    return M, K, N