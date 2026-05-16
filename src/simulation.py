import numpy as np
import scipy as sp


HBAR = 6.582119569e-16  # eV*s
REST_ENERGY_E = 0.51999895069e6  # eV
C = 2.997924580e18  # A/S
M_E = REST_ENERGY_E / (C**2)

def generate_g_vectors(lattice_constant, num_vectors=200):
    """
    Generates reciprocal lattice vectors (G) for an FCC/Diamond lattice.
    """
    # Pre-factor for reciprocal space (2 pi / a[A]) ( x, y ,z )
    prefactor = 2 * np.pi / lattice_constant
    
    # Primitive reciprocal vectors for FCC
    b1 = prefactor * np.array([-1, 1, 1])
    b2 = prefactor * np.array([1, -1, 1])
    b3 = prefactor * np.array([1, 1, -1]) 
    
    # Range of integer indices to search
    # A range of 5 (-5 to 5) is plenty to find the first 200 vectors
    n_range = np.arange(-6, 7)
    
    g_list = []
    
    for h in n_range:
        for k in n_range:
            for l in n_range:
                # G = h*b1 + k*b2 + l*b3
                g = h*b1 + k*b2 + l*b3
                mag = np.linalg.norm(g)
                g_list.append((g, mag))
                
    # Sort vectors by their magnitude (energy cutoff logic)
    g_list.sort(key=lambda x: x[1])
    
    # Extract the top 'num_vectors'
    sorted_g = np.array([item[0] for item in g_list[0:num_vectors]])
    
    return sorted_g
# Constants for Silicon
a_ge = 5.43  # Angstroms
G = generate_g_vectors(a_ge, 300)

print(f"Generated {len(G)} G-vectors.")
#print("First 5 vectors (including origin):\n", G)


# G_i - G_j =  deltaG[i,j]
q = G[:, np.newaxis, :] - G[np.newaxis, :, :]

G_squared_unit = (2 * np.pi / a_ge)**2
q_squared = np.sum(q**2, axis=-1)
q_sq = q_squared / G_squared_unit

tau = a_ge/8 * np.array([1,1,1])
V_matrix = np.zeros_like(q_sq, dtype=float)

S_q = 2*np.cos(np.dot(q, tau))

v3 = -1.43
v8 = 0.27
v11 = 0.54

mask_3 = np.isclose(q_sq, 3.0)
V_matrix[mask_3] = v3 * S_q[mask_3]

mask_8 = np.isclose(q_sq, 8.0)
V_matrix[mask_8] = v8 * S_q[mask_8]

mask_11 = np.isclose(q_sq, 11.0)
V_matrix[mask_11] = v11 * S_q[mask_11]

prefactor = 2 * np.pi / a_ge
L_point = np.array([0.5, 0.5, 0.5]) * prefactor
Gamma_point = np.array([0.0, 0.0, 0.0]) * prefactor
X_point = np.array([1.0, 0.0, 0.0]) * prefactor

num_points = 100
k_path_L_G = np.linspace(L_point, Gamma_point, num_points)
# Path 2: Gamma to X (skip the first point to avoid duplicate Gamma)
k_path_G_X = np.linspace(Gamma_point, X_point, num_points)[1:]
k_path = np.vstack((k_path_L_G, k_path_G_X))

distances_L_G = -np.linalg.norm(k_path_L_G - Gamma_point, axis=1) / prefactor
distances_G_X = np.linalg.norm(k_path_G_X - Gamma_point, axis=1) / prefactor
k_dist = np.concatenate((distances_L_G, distances_G_X))

bands = []
for k_vec in k_path:
    # Calculate kinetic energy for this specific k
    # V_matrix is k-independent, so we only update T_matrix
    gpk = G + k_vec[np.newaxis, :]
    g_mags = np.linalg.norm(gpk, axis=1)
    T_matrix = np.diag((HBAR**2 * g_mags**2) / (2 * M_E))
    
    H = T_matrix + V_matrix
    
    # Solve and store the first 8 bands
    eigvals, _ = sp.linalg.eigh(H)
    bands.append(eigvals[0:8])

bands = np.array(bands)

gamma_index = num_points - 1  # Index where k = 0
vbm = bands[gamma_index, 3]   # 4th band at Gamma
shifted_bands = bands - vbm