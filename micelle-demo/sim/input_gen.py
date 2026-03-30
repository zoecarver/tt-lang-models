"""CG micelle system builder.

Builds a coarse-grained lipid + water system for micelle self-assembly.
Pure Python, no external files needed.

Each lipid: 1 HEAD bead + 4 TAIL beads in a chain.
HEAD beads carry charge (+/-0.5 alternating), tails are neutral.
Water beads fill the remaining volume.
"""
import numpy as np
try:
    from .params import (HEAD, TAIL, WATER, NA, CL, CA, CHARGES, BOND_R0)
except ImportError:
    from params import (HEAD, TAIL, WATER, NA, CL, CA, CHARGES, BOND_R0)


def make_lipid_system(n_lipids=80, n_water=2000, density=0.3, seed=42):
    """Build a CG lipid + water system.

    Returns:
        positions: (N, 3) float64
        types: (N,) int, bead type IDs
        charges: (N,) float64
        bonds: (n_bonds, 2) int, pairs of bonded atom indices
        box_length: float
    """
    np.random.seed(seed)

    beads_per_lipid = 9  # 1 head + 8 tail
    n_lipid_beads = n_lipids * beads_per_lipid
    n_total = n_lipid_beads + n_water
    box_length = (n_total / density) ** (1.0 / 3.0)

    types = np.zeros(n_total, dtype=int)
    charges = np.zeros(n_total)
    bonds = []

    # Place lipids in a loose cloud near box center — spread enough to see
    # them come together, close enough to aggregate within ~5k steps
    center = box_length / 2.0
    lip_side = int(np.ceil(n_lipid_beads ** (1.0 / 3.0)))
    lip_spacing = 2.0  # wider spacing = looser initial cloud
    lip_positions = []
    lip_origin = center - lip_side * lip_spacing / 2.0
    for ix in range(lip_side):
        for iy in range(lip_side):
            for iz in range(lip_side):
                if len(lip_positions) < n_lipid_beads:
                    lip_positions.append([
                        lip_origin + (ix + 0.5) * lip_spacing,
                        lip_origin + (iy + 0.5) * lip_spacing,
                        lip_origin + (iz + 0.5) * lip_spacing,
                    ])
    # Add extra jitter so it doesn't look like a perfect grid
    lip_positions = np.array(lip_positions[:n_lipid_beads])
    lip_positions += np.random.normal(0, 0.3, lip_positions.shape)
    lip_positions = lip_positions.tolist()

    # Water: cubic lattice filling the full box
    wat_side = int(np.ceil(n_water ** (1.0 / 3.0)))
    wat_spacing = box_length / wat_side
    wat_positions = []
    for ix in range(wat_side):
        for iy in range(wat_side):
            for iz in range(wat_side):
                if len(wat_positions) < n_water:
                    wat_positions.append([
                        (ix + 0.5) * wat_spacing,
                        (iy + 0.5) * wat_spacing,
                        (iz + 0.5) * wat_spacing,
                    ])

    positions = np.array(lip_positions[:n_lipid_beads] + wat_positions[:n_water])
    positions += np.random.normal(0, 0.05, positions.shape)
    positions %= box_length

    # Assign types: first n_lipid_beads are lipids, rest water
    for i in range(n_lipids):
        base_idx = i * beads_per_lipid
        types[base_idx] = HEAD
        charges[base_idx] = 0.5 if i % 2 == 0 else -0.5
        for t in range(8):
            tail_idx = base_idx + 1 + t
            types[tail_idx] = TAIL
            bonds.append([base_idx + t, tail_idx])

    water_start = n_lipid_beads
    types[water_start:] = WATER

    bonds = np.array(bonds, dtype=np.int32)

    # Verify charge neutrality
    total_charge = charges.sum()
    if abs(total_charge) > 1e-6:
        charges[0] -= total_charge

    return positions, types, charges, bonds, box_length


def add_ions(positions, types, charges, bonds, box_length,
             n_na=0, n_cl=0, n_ca=0, seed=None):
    """Inject ions into an existing system.

    Appends ion beads to the arrays. Bond list is unchanged (ions are unbonded).
    Returns new (positions, types, charges, bonds, box_length).
    """
    if seed is not None:
        np.random.seed(seed)

    ion_specs = [(NA, CHARGES[NA], n_na),
                 (CL, CHARGES[CL], n_cl),
                 (CA, CHARGES[CA], n_ca)]

    new_pos = []
    new_types = []
    new_charges = []
    for type_id, charge, count in ion_specs:
        if count > 0:
            pos = np.random.uniform(0, box_length, (count, 3))
            new_pos.append(pos)
            new_types.extend([type_id] * count)
            new_charges.extend([charge] * count)

    if not new_pos:
        return positions, types, charges, bonds, box_length

    new_pos = np.concatenate(new_pos, axis=0)
    positions = np.concatenate([positions, new_pos], axis=0)
    types = np.concatenate([types, np.array(new_types, dtype=int)])
    charges = np.concatenate([charges, np.array(new_charges)])

    return positions, types, charges, bonds, box_length


def reference_lj_forces(positions, types, eps_table, sig2_table, box_length, n_types):
    """O(N^2) reference LJ forces for validation."""
    n = len(positions)
    dr = positions[:, None, :] - positions[None, :, :]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(r2, 1e10)

    # Per-pair parameters from type table
    pair_idx = types[:, None] * n_types + types[None, :]
    eps = eps_table[pair_idx]
    s2 = sig2_table[pair_idx]

    s2_over_r2 = s2 / r2
    s6 = s2_over_r2 ** 3
    s12 = s6 * s6

    # F_ij = 24 * eps / r^2 * (2*s12 - s6) * dr_ij
    f_mag = 24.0 * eps / r2 * (2.0 * s12 - s6)
    forces = np.sum(f_mag[:, :, None] * dr, axis=1)
    return forces


def reference_bond_forces(positions, bonds, k, r0, box_length):
    """Reference harmonic bond forces for validation."""
    if len(bonds) == 0:
        return np.zeros_like(positions)
    forces = np.zeros_like(positions)
    for b in bonds:
        i, j = b[0], b[1]
        dr = positions[i] - positions[j]
        dr -= box_length * np.floor(dr / box_length + 0.5)
        r = np.linalg.norm(dr)
        if r < 1e-10:
            continue
        f_mag = -k * (r - r0)
        f_vec = f_mag * dr / r
        forces[i] += f_vec
        forces[j] -= f_vec
    return forces
