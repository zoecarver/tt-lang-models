"""Force field parameters for CG micelle simulation.

Bead types: HEAD (charged lipid head), TAIL (hydrophobic tail),
WATER (CG solvent), NA (sodium), CL (chloride), CA (calcium).

LJ parameters drive self-assembly: tails attract tails strongly (hydrophobic),
heads prefer water (hydrophilic), creating micelle geometry.
"""
import numpy as np

TILE = 32
N_NBR = 27

# Bead type IDs
HEAD = 0
TAIL = 1
WATER = 2
NA = 3
CL = 4
CA = 5
N_TYPES = 6

TYPE_NAMES = ["HEAD", "TAIL", "WATER", "NA", "CL", "CA"]

# LJ sigma (bead diameter) per type pair - symmetric 6x6
# Units: reduced (sigma ~ 1.0 for CG beads)
_SIGMA = np.ones((N_TYPES, N_TYPES), dtype=np.float32)
_SIGMA[HEAD, HEAD] = 1.0
_SIGMA[TAIL, TAIL] = 1.0
_SIGMA[HEAD, TAIL] = _SIGMA[TAIL, HEAD] = 1.0
_SIGMA[WATER, WATER] = 1.0
_SIGMA[HEAD, WATER] = _SIGMA[WATER, HEAD] = 1.0
_SIGMA[TAIL, WATER] = _SIGMA[WATER, TAIL] = 1.0
_SIGMA[NA, :] = _SIGMA[:, NA] = 0.8
_SIGMA[CL, :] = _SIGMA[:, CL] = 0.9
_SIGMA[CA, :] = _SIGMA[:, CA] = 0.7

# LJ epsilon (interaction strength) per type pair
# Key physics: tail-tail strong, head-water strong, tail-water weak
_EPSILON = np.zeros((N_TYPES, N_TYPES), dtype=np.float32)
_EPSILON[TAIL, TAIL] = 2.0      # strong hydrophobic attraction
_EPSILON[HEAD, HEAD] = 0.5      # moderate head-head
_EPSILON[HEAD, TAIL] = 0.3      # weak head-tail (drives separation)
_EPSILON[TAIL, HEAD] = 0.3
_EPSILON[WATER, WATER] = 1.0    # standard water-water
_EPSILON[HEAD, WATER] = 1.5     # heads like water (hydrophilic)
_EPSILON[WATER, HEAD] = 1.5
_EPSILON[TAIL, WATER] = 0.2     # tails dislike water (hydrophobic effect)
_EPSILON[WATER, TAIL] = 0.2
# Ion interactions
_EPSILON[NA, WATER] = _EPSILON[WATER, NA] = 1.0
_EPSILON[CL, WATER] = _EPSILON[WATER, CL] = 1.0
_EPSILON[CA, WATER] = _EPSILON[WATER, CA] = 1.2
_EPSILON[NA, HEAD] = _EPSILON[HEAD, NA] = 1.2
_EPSILON[CL, HEAD] = _EPSILON[HEAD, CL] = 1.2
_EPSILON[CA, HEAD] = _EPSILON[HEAD, CA] = 1.5
_EPSILON[NA, NA] = 0.3
_EPSILON[CL, CL] = 0.3
_EPSILON[CA, CA] = 0.3

# Charges per type
CHARGES = {HEAD: 0.5, TAIL: 0.0, WATER: 0.0, NA: 1.0, CL: -1.0, CA: 2.0}

# Bond parameters (harmonic: F = -k*(r - r0)*r_hat)
BOND_K = 100.0    # spring constant (stiff bonds to keep lipids connected)
BOND_R0 = 1.0     # equilibrium distance

# Ewald parameters
ALPHA = 1.0
K_GRID = 32
N_GAUSS = 16

# Erfc approximation (Abramowitz & Stegun)
ERFC_A1 = 0.254829592
ERFC_A2 = -0.284496736
ERFC_A3 = 1.421413741
ERFC_A4 = -1.453152027
ERFC_A5 = 1.061405429
ERFC_P = 0.3275911


def get_lj_tables():
    """Return flat epsilon and sigma^2 tables for kernel consumption.

    Returns (N_TYPES*N_TYPES,) arrays laid out so that
    table[type_i * N_TYPES + type_j] gives the pair parameters.
    We store sigma^6 and sigma^12 pre-computed for the kernel.
    """
    sig2 = (_SIGMA * _SIGMA).flatten().astype(np.float32)
    eps = _EPSILON.flatten().astype(np.float32)
    return eps, sig2
