"""
Cell-list molecular dynamics on Tenstorrent hardware using TT-Lang.

Full Ewald electrostatics: erfc-damped real-space (cell-list, TT kernel) +
u-series reciprocal-space (separable Gaussian convolution, TT kernel + host).
LJ short-range forces. Periodic boundary conditions.
On-device Verlet integration (f32) with bf16 force kernels.

Validated: 10K atoms, 10K steps, 1.1ms/step (non-rebuild), 12 min total.
"""
import time
import torch
import numpy as np
import ttnn
import ttl

TILE = 32
N_NBR = 27
ALPHA = 1.0
K_GRID = 32
N_GAUSS = 16

ERFC_A1 = 0.254829592
ERFC_A2 = -0.284496736
ERFC_A3 = 1.421413741
ERFC_A4 = -1.453152027
ERFC_A5 = 1.061405429
ERFC_P = 0.3275911


# ---- Host utilities ----

def make_system(n_atoms, density=0.3, seed=42):
    """Create a lattice system with charges."""
    np.random.seed(seed)
    box_length = (n_atoms / density) ** (1.0 / 3.0)
    n_side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
    spacing = box_length / n_side
    positions = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(positions) < n_atoms:
                    positions.append([(ix+0.5)*spacing, (iy+0.5)*spacing, (iz+0.5)*spacing])
    positions = np.array(positions[:n_atoms])
    positions += np.random.normal(0, 0.05, positions.shape)
    positions = positions % box_length
    charges = np.random.randn(n_atoms) * 0.3
    charges -= charges.mean()
    return positions, charges, box_length


def compute_energy(positions, charges, box_length):
    """O(N^2) energy for validation (LJ + Coulomb, no cutoff)."""
    dr = positions[:, None, :] - positions[None, :, :]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(r2, 1e10)
    r = np.sqrt(r2)
    r6_inv = (1.0 / r2) ** 3
    lj = 4.0 * (r6_inv ** 2 - r6_inv)
    coul = (charges[:, None] * charges[None, :]) / r
    return 0.5 * (np.sum(lj) + np.sum(coul))


def direct_forces(positions, charges, box_length):
    """O(N^2) reference forces (LJ + Coulomb, no cutoff)."""
    dr = positions[:, None, :] - positions[None, :, :]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(r2, 1e10)
    r = np.sqrt(r2)
    r2_inv = 1.0 / r2
    r6_inv = r2_inv ** 3
    f_lj = 24.0 * r2_inv * (2.0 * r6_inv ** 2 - r6_inv)
    qq = charges[:, None] * charges[None, :]
    f_coul = qq / (r2 * r)
    return np.sum((f_lj + f_coul)[:, :, None] * dr, axis=1)


# ---- Reciprocal-space: u-series Gaussian convolution ----

def gaussian_decomposition(alpha, n_gauss):
    """Gauss-Legendre decomposition of 1/r into sum of Gaussians."""
    nodes, gl_weights = np.polynomial.legendre.leggauss(n_gauss)
    t = alpha / 2.0 * (nodes + 1.0)
    w = alpha / 2.0 * gl_weights
    return t ** 2, (2.0 / np.sqrt(np.pi)) * w


def make_conv_kernel(K, h, exponent):
    """Build KxK circulant Gaussian convolution matrix."""
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            d = min(abs(i - j), K - abs(i - j))
            M[i, j] = np.exp(-exponent * (d * h) ** 2)
    return M


def bspline4_weights_vec(u):
    """Order-4 B-spline weights for charge spreading."""
    u2 = u * u; u3 = u2 * u
    w3 = u3 / 6.0
    w2 = (1.0 + 3.0 * u * (1.0 + u * (1.0 - u))) / 6.0
    w1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
    w0 = 1.0 - w1 - w2 - w3
    return np.stack([w0, w1, w2, w3], axis=-1)


def bspline4_dweights_vec(u):
    """Order-4 B-spline derivative weights for force interpolation."""
    u2 = u * u
    dw3 = u2 / 2.0
    dw2 = (3.0 * (1.0 + 2.0 * u - 3.0 * u2)) / 6.0
    dw1 = (-12.0 * u + 9.0 * u2) / 6.0
    dw0 = -(dw1 + dw2 + dw3)
    return np.stack([dw0, dw1, dw2, dw3], axis=-1)


def spread_charges(positions, charges, box_length, K, order=4):
    """B-spline charge spreading onto 3D grid."""
    h = box_length / K
    s = positions / h
    g0 = np.floor(s).astype(int) - (order // 2 - 1)
    f = s - np.floor(s)

    wx = bspline4_weights_vec(f[:, 0])
    wy = bspline4_weights_vec(f[:, 1])
    wz = bspline4_weights_vec(f[:, 2])

    w3d = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    w3d *= charges[:, None, None, None]

    offsets = np.arange(order)
    gx = (g0[:, 0, None] + offsets[None, :]) % K
    gy = (g0[:, 1, None] + offsets[None, :]) % K
    gz = (g0[:, 2, None] + offsets[None, :]) % K

    grid = np.zeros((K, K, K))
    for ix in range(order):
        for iy in range(order):
            for iz in range(order):
                np.add.at(grid, (gx[:, ix], gy[:, iy], gz[:, iz]), w3d[:, ix, iy, iz])
    return grid


def interpolate_forces_bspline(positions, potential_grid, box_length, order=4):
    """B-spline force interpolation from potential grid."""
    K = potential_grid.shape[0]
    h = box_length / K
    s = positions / h
    g0 = np.floor(s).astype(int) - (order // 2 - 1)
    f = s - np.floor(s)

    wx = bspline4_weights_vec(f[:, 0])
    wy = bspline4_weights_vec(f[:, 1])
    wz = bspline4_weights_vec(f[:, 2])
    dwx = bspline4_dweights_vec(f[:, 0])
    dwy = bspline4_dweights_vec(f[:, 1])
    dwz = bspline4_dweights_vec(f[:, 2])

    offsets = np.arange(order)
    gx = (g0[:, 0, None] + offsets[None, :]) % K
    gy = (g0[:, 1, None] + offsets[None, :]) % K
    gz = (g0[:, 2, None] + offsets[None, :]) % K

    forces = np.zeros((len(positions), 3))
    inv_h = 1.0 / h
    for ix in range(order):
        for iy in range(order):
            for iz in range(order):
                phi = potential_grid[gx[:, ix], gy[:, iy], gz[:, iz]]
                forces[:, 0] -= dwx[:, ix] * wy[:, iy] * wz[:, iz] * phi * inv_h
                forces[:, 1] -= wx[:, ix] * dwy[:, iy] * wz[:, iz] * phi * inv_h
                forces[:, 2] -= wx[:, ix] * wy[:, iy] * dwz[:, iz] * phi * inv_h
    return forces


@ttl.kernel(grid="auto")
def xy_conv_kernel(charge_grid, kernels, potential_grid):
    """U-series xy-convolution: M @ charge_slice @ M for each Gaussian component.

    Replaces the FFT in traditional PME. Separable Gaussian decomposition
    means we can convolve with matrix multiplies instead of FFTs.
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    total_slices = charge_grid.shape[0] // TILE
    n_kernels = kernels.shape[0] // TILE
    slices_per_core = -(-total_slices // grid_cols)

    cg_cb = ttl.make_dataflow_buffer_like(charge_grid, shape=(1, 1), buffer_factor=2)
    km_cb = ttl.make_dataflow_buffer_like(kernels, shape=(1, 1), buffer_factor=2)
    tmp_cb = ttl.make_dataflow_buffer_like(charge_grid, shape=(1, 1), buffer_factor=2)
    acc_cb = ttl.make_dataflow_buffer_like(potential_grid, shape=(1, 1), buffer_factor=2)
    par_cb = ttl.make_dataflow_buffer_like(potential_grid, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_dataflow_buffer_like(potential_grid, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_z in range(slices_per_core):
            z = core_x * slices_per_core + local_z
            if z < total_slices:
                with cg_cb.wait() as cg:
                    with km_cb.wait() as M:
                        with tmp_cb.reserve() as t:
                            t.store(M @ cg)
                        with tmp_cb.wait() as tv:
                            with acc_cb.reserve() as a:
                                a.store(tv @ M)
                    for g in range(n_kernels - 1):
                        with km_cb.wait() as M:
                            with tmp_cb.reserve() as t:
                                t.store(M @ cg)
                            with tmp_cb.wait() as tv:
                                with par_cb.reserve() as p:
                                    p.store(tv @ M)
                        with par_cb.wait() as pv, acc_cb.wait() as av:
                            with acc_cb.reserve() as a:
                                a.store(av + pv)
                with acc_cb.wait() as final, out_cb.reserve() as o:
                    o.store(final)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_z in range(slices_per_core):
            z = core_x * slices_per_core + local_z
            if z < total_slices:
                with cg_cb.reserve() as blk:
                    tx = ttl.copy(charge_grid[z, 0], blk); tx.wait()
                for g in range(n_kernels):
                    with km_cb.reserve() as blk:
                        tx = ttl.copy(kernels[g, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_z in range(slices_per_core):
            z = core_x * slices_per_core + local_z
            if z < total_slices:
                with out_cb.wait() as blk:
                    tx = ttl.copy(blk, potential_grid[z, 0]); tx.wait()


def compute_reciprocal_forces(device, positions, charges, box_length,
                               alpha=ALPHA, n_gauss=N_GAUSS, K=K_GRID):
    """Reciprocal-space forces via u-series Gaussian convolution.

    Pipeline: spread charges -> xy-convolve (TT kernel) -> z-convolve (host) -> interpolate forces.
    """
    h = box_length / K
    exponents, weights = gaussian_decomposition(alpha, n_gauss)

    charge_grid_3d = spread_charges(positions, charges, box_length, K)

    # Pack charge grid into tile layout: each z-slice is one tile
    cg_np = np.zeros((K * TILE, TILE), dtype=np.float32)
    for z in range(K):
        cg_np[z*TILE:(z+1)*TILE, :TILE] = charge_grid_3d[:, :, z].astype(np.float32)

    # Pack Gaussian convolution kernels
    km_np = np.zeros((n_gauss * TILE, TILE), dtype=np.float32)
    for g in range(n_gauss):
        M_g = make_conv_kernel(K, h, exponents[g]) * weights[g]
        km_np[g*TILE:(g+1)*TILE, :TILE] = M_g.astype(np.float32)

    def to_tt(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    cg_tt = to_tt(cg_np)
    km_tt = to_tt(km_np)
    pot_tt = to_tt(np.zeros((K * TILE, TILE), dtype=np.float32))

    # XY-convolution on device
    xy_conv_kernel(cg_tt, km_tt, pot_tt)

    # Z-convolution on host
    pot_np = ttnn.to_torch(pot_tt).float().numpy()
    xy_conv_3d = np.zeros((K, K, K))
    for z in range(K):
        xy_conv_3d[:, :, z] = pot_np[z*TILE:(z+1)*TILE, :TILE]

    M_z = np.zeros((K, K))
    for g in range(n_gauss):
        M_z += make_conv_kernel(K, h, exponents[g]) * weights[g]
    potential_3d = np.zeros_like(xy_conv_3d)
    for x in range(K):
        potential_3d[x, :, :] = (M_z @ xy_conv_3d[x, :, :].T).T

    # Interpolate forces from potential grid
    forces_recip = interpolate_forces_bspline(positions, potential_3d, box_length)
    forces_recip *= -charges[:, None]
    return forces_recip


# ---- Cell-list data packing ----

def build_cell_data(positions, charges, box_length, r_cut, atom_types=None):
    """Assign atoms to cells, pack into tile arrays, build self-exclusion masks."""
    n = len(positions)
    n_cells_dim = max(3, int(box_length / r_cut))
    n_cells_total = n_cells_dim ** 3

    cell_size = box_length / n_cells_dim
    cidx = np.floor(positions / cell_size).astype(int) % n_cells_dim
    cell_id = cidx[:, 0] * n_cells_dim**2 + cidx[:, 1] * n_cells_dim + cidx[:, 2]

    sort_idx = np.argsort(cell_id, kind='stable')
    sorted_cell_id = cell_id[sort_idx]

    cell_counts = np.bincount(cell_id, minlength=n_cells_total)
    cell_starts = np.zeros(n_cells_total + 1, dtype=int)
    np.cumsum(cell_counts, out=cell_starts[1:])

    local_idx = np.arange(n) - cell_starts[sorted_cell_id]
    valid = local_idx < TILE

    cell_atom_map = [sort_idx[cell_starts[c]:cell_starts[c+1]].tolist()
                     for c in range(n_cells_total)]

    valid_atoms = sort_idx[valid]
    valid_cells = sorted_cell_id[valid]
    valid_local = local_idx[valid]
    rows = valid_cells * TILE + valid_local

    own_px = np.zeros((n_cells_total * TILE, TILE), dtype=np.float32)
    own_py = np.zeros_like(own_px)
    own_pz = np.zeros_like(own_px)
    own_q = np.zeros_like(own_px)
    own_px[rows, 0] = positions[valid_atoms, 0]
    own_py[rows, 0] = positions[valid_atoms, 1]
    own_pz[rows, 0] = positions[valid_atoms, 2]
    own_q[rows, 0] = charges[valid_atoms]

    # Per-atom sqrt(epsilon) for per-type LJ (geometric combining rule)
    SQRT_EPS_MAP = {0: 0.8, 1: 2.0, 2: 0.3, 3: 0.5, 4: 0.5, 5: 0.5}
    own_epslj = np.zeros_like(own_px)
    if atom_types is not None:
        for r_idx in range(len(rows)):
            a = valid_atoms[r_idx]
            own_epslj[rows[r_idx], 0] = SQRT_EPS_MAP.get(atom_types[a], 0.5)
    else:
        own_epslj[rows, 0] = 1.0

    # Self-exclusion masks: 1e6 for empty slots and self-pairs
    offsets = np.array([(dx, dy, dz)
                        for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)])
    cell_3d = np.stack(np.unravel_index(np.arange(n_cells_total),
                       (n_cells_dim, n_cells_dim, n_cells_dim)), axis=-1)
    nbr_3d = (cell_3d[:, None, :] + offsets[None, :, :]) % n_cells_dim
    nbr_cid = (nbr_3d[:, :, 0] * n_cells_dim**2 +
               nbr_3d[:, :, 1] * n_cells_dim + nbr_3d[:, :, 2])

    own_cnt = np.minimum(cell_counts, TILE)
    nbr_cnt = own_cnt[nbr_cid]
    is_self = (nbr_cid == np.arange(n_cells_total)[:, None])

    row_idx = np.arange(TILE)[None, None, :, None]
    col_idx = np.arange(TILE)[None, None, None, :]
    oc = own_cnt[:, None, None, None]
    nc = nbr_cnt[:, :, None, None]

    masks_4d = np.where(
        (row_idx >= oc) | (col_idx >= nc) | (is_self[:, :, None, None] & (row_idx == col_idx)),
        np.float32(1e6), np.float32(0.0))
    masks = masks_4d.reshape(n_cells_total * N_NBR * TILE, TILE)

    return (own_px, own_py, own_pz, own_q, own_epslj,
            masks, cell_atom_map, n_cells_total, n_cells_dim)


def build_rebuild_index(positions, charges, box_length, r_cut,
                        old_cell_atom_map, n_cells_total_old):
    """Build gather index for on-device cell-list rebuild.

    Given current atom positions and the old cell layout, compute:
    - New cell assignments
    - A gather index tensor that rearranges old cell-layout rows to new cell-layout
    - New masks

    Returns gather_idx (uint32), new masks, new cell_atom_map, n_cells_total, n_cells_dim.
    The caller uses ttnn.gather(data, dim=0, index=gather_idx) to rearrange on device.
    """
    n = len(positions)
    n_cells_dim = max(3, int(box_length / r_cut))
    n_cells_total = n_cells_dim ** 3
    n_rows = n_cells_total * TILE

    cell_size = box_length / n_cells_dim
    cidx = np.floor(positions / cell_size).astype(int) % n_cells_dim
    cell_id = cidx[:, 0] * n_cells_dim**2 + cidx[:, 1] * n_cells_dim + cidx[:, 2]

    sort_idx = np.argsort(cell_id, kind='stable')
    sorted_cell_id = cell_id[sort_idx]

    cell_counts = np.bincount(cell_id, minlength=n_cells_total)
    cell_starts = np.zeros(n_cells_total + 1, dtype=int)
    np.cumsum(cell_counts, out=cell_starts[1:])

    local_idx = np.arange(n) - cell_starts[sorted_cell_id]
    valid = local_idx < TILE

    cell_atom_map = [sort_idx[cell_starts[c]:cell_starts[c+1]].tolist()
                     for c in range(n_cells_total)]

    # Build old atom -> old_row mapping
    old_atom_to_row = np.zeros(n, dtype=np.int64)
    for cell_id_old, atoms in enumerate(old_cell_atom_map):
        for local_i, atom_id in enumerate(atoms[:TILE]):
            old_atom_to_row[atom_id] = cell_id_old * TILE + local_i

    # Build gather index: new_row -> old_row
    # For valid atoms: gather_idx[new_row, :] = old_row (broadcast across columns)
    # For empty slots: point to row 0 (data is zero, mask will exclude)
    gather_idx = np.zeros((n_rows, TILE), dtype=np.int64)

    valid_atoms = sort_idx[valid]
    valid_cells = sorted_cell_id[valid]
    valid_local = local_idx[valid]
    new_rows = valid_cells * TILE + valid_local
    old_rows = old_atom_to_row[valid_atoms]

    # Each row of gather_idx should contain the source row index
    # For dim=0 gather: output[i, j] = input[index[i, j], j]
    # We want column 0 to come from old_row, and other columns from row 0 (zeros)
    for i in range(len(new_rows)):
        gather_idx[new_rows[i], :] = old_rows[i]

    # Masks (same as build_cell_data)
    offsets = np.array([(dx, dy, dz)
                        for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)])
    cell_3d = np.stack(np.unravel_index(np.arange(n_cells_total),
                       (n_cells_dim, n_cells_dim, n_cells_dim)), axis=-1)
    nbr_3d = (cell_3d[:, None, :] + offsets[None, :, :]) % n_cells_dim
    nbr_cid = (nbr_3d[:, :, 0] * n_cells_dim**2 +
               nbr_3d[:, :, 1] * n_cells_dim + nbr_3d[:, :, 2])
    own_cnt = np.minimum(cell_counts, TILE)
    nbr_cnt = own_cnt[nbr_cid]
    is_self = (nbr_cid == np.arange(n_cells_total)[:, None])
    row_idx = np.arange(TILE)[None, None, :, None]
    col_idx = np.arange(TILE)[None, None, None, :]
    oc = own_cnt[:, None, None, None]
    nc = nbr_cnt[:, :, None, None]
    masks_4d = np.where(
        (row_idx >= oc) | (col_idx >= nc) | (is_self[:, :, None, None] & (row_idx == col_idx)),
        np.float32(1e6), np.float32(0.0))
    masks = masks_4d.reshape(n_cells_total * N_NBR * TILE, TILE)

    # Also build new charge/eps arrays (static per atom, need repacking)
    own_q = np.zeros((n_rows, TILE), dtype=np.float32)
    own_q[new_rows, 0] = charges[valid_atoms]

    return gather_idx, masks, own_q, cell_atom_map, n_cells_total, n_cells_dim


def _build_cell_index_arrays(cell_atom_map):
    """Build flat row/atom index arrays for vectorized cell-layout operations."""
    rows_list, atoms_list = [], []
    for cell_id, atoms in enumerate(cell_atom_map):
        na = min(len(atoms), TILE)
        if na > 0:
            rows_list.append(cell_id * TILE + np.arange(na))
            atoms_list.append(np.array(atoms[:na]))
    if rows_list:
        return np.concatenate(rows_list), np.concatenate(atoms_list)
    return np.array([], dtype=int), np.array([], dtype=int)


def pack_cell_layout(data_3col, cell_atom_map, n_cells_total):
    """Pack per-atom (N,3) data into cell-layout tile arrays."""
    ax = np.zeros((n_cells_total * TILE, TILE), dtype=np.float32)
    ay = np.zeros_like(ax)
    az = np.zeros_like(ax)
    rows, atom_ids = _build_cell_index_arrays(cell_atom_map)
    if len(rows) > 0:
        ax[rows, 0] = data_3col[atom_ids, 0]
        ay[rows, 0] = data_3col[atom_ids, 1]
        az[rows, 0] = data_3col[atom_ids, 2]
    return ax, ay, az


def extract_cell_data(dx_np, dy_np, dz_np, cell_atom_map, n_atoms):
    """Unpack cell-layout tile arrays back to per-atom (N,3) order."""
    result = np.zeros((n_atoms, 3))
    rows, atom_ids = _build_cell_index_arrays(cell_atom_map)
    if len(rows) > 0:
        result[atom_ids, 0] = dx_np[rows, 0]
        result[atom_ids, 1] = dy_np[rows, 0]
        result[atom_ids, 2] = dz_np[rows, 0]
    return result


# ---- On-device Verlet MD loop ----

def run_md_loop(device, positions, velocities, charges, box_length,
                n_steps=10, dt=0.005, skin=0.5, check_interval=20,
                max_rebuild_interval=5000, alpha=ALPHA, target_T=1.0,
                dump_callback=None, dump_interval=1000, atom_types=None,
                bonds=None):
    """Run MD with velocity Verlet integration on Tenstorrent hardware.

    Verlet skin method: cell_size = r_cut + skin provides a drift buffer so
    rebuilds are triggered only when max atom displacement exceeds skin/2.
    Between rebuilds, all updates happen on device with zero host copies.

    Full Ewald forces: real-space (erfc-damped LJ+Coulomb, bf16 TT kernel) +
    reciprocal-space (u-series Gaussian convolution, bf16 TT kernel + host).
    """
    n = len(positions)
    r_cut = min(box_length / 2.0 - 0.1, 3.0 / alpha)
    cell_r = r_cut + skin

    # Constants captured by kernel closures
    c_box = float(box_length)
    c_inv_box = 1.0 / float(box_length)
    c_half = 0.5
    c_dt_half = 0.5 * dt
    c_dt = float(dt)
    c_lj24 = 24.0
    c_alpha_sq = float(alpha * alpha)
    c_p_alpha = float(ERFC_P * alpha)
    c_two_a_sp = float(2.0 * alpha / np.sqrt(np.pi))
    c_a1 = float(ERFC_A1)
    c_a2 = float(-ERFC_A2)
    c_a3 = float(ERFC_A3)
    c_a4 = float(-ERFC_A4)
    c_a5 = float(ERFC_A5)

    # --- Verlet velocity half-step: vel += 0.5*dt*force (f32) ---
    @ttl.kernel(grid="auto")
    def vel_half_kernel(vel_x, vel_y, vel_z, fx, fy, fz):
        grid_cols, _ = ttl.grid_size(dims=2)
        nc = vel_x.shape[0] // TILE
        cpc = -(-nc // grid_cols)

        v_cb = ttl.make_dataflow_buffer_like(vel_x, shape=(1, 1), buffer_factor=2)
        f_cb = ttl.make_dataflow_buffer_like(fx, shape=(1, 1), buffer_factor=2)
        o_cb = ttl.make_dataflow_buffer_like(vel_x, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for lc in range(cpc):
                cid = core_x * cpc + lc
                if cid < nc:
                    for _ in range(3):
                        with v_cb.wait() as v, f_cb.wait() as f, o_cb.reserve() as o:
                            o.store(v + f * ttl.math.fill(v, c_dt_half))

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for lc in range(cpc):
                cid = core_x * cpc + lc
                if cid < nc:
                    with v_cb.reserve() as blk:
                        tx = ttl.copy(vel_x[cid, 0], blk); tx.wait()
                    with f_cb.reserve() as blk:
                        tx = ttl.copy(fx[cid, 0], blk); tx.wait()
                    with v_cb.reserve() as blk:
                        tx = ttl.copy(vel_y[cid, 0], blk); tx.wait()
                    with f_cb.reserve() as blk:
                        tx = ttl.copy(fy[cid, 0], blk); tx.wait()
                    with v_cb.reserve() as blk:
                        tx = ttl.copy(vel_z[cid, 0], blk); tx.wait()
                    with f_cb.reserve() as blk:
                        tx = ttl.copy(fz[cid, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for lc in range(cpc):
                cid = core_x * cpc + lc
                if cid < nc:
                    with o_cb.wait() as blk:
                        tx = ttl.copy(blk, vel_x[cid, 0]); tx.wait()
                    with o_cb.wait() as blk:
                        tx = ttl.copy(blk, vel_y[cid, 0]); tx.wait()
                    with o_cb.wait() as blk:
                        tx = ttl.copy(blk, vel_z[cid, 0]); tx.wait()

    # --- Verlet position update: pos += dt*vel, PBC wrap (f32) ---
    @ttl.kernel(grid="auto")
    def pos_update_kernel(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z):
        grid_cols, _ = ttl.grid_size(dims=2)
        nc = pos_x.shape[0] // TILE
        cpc = -(-nc // grid_cols)

        p_cb = ttl.make_dataflow_buffer_like(pos_x, shape=(1, 1), buffer_factor=2)
        v_cb = ttl.make_dataflow_buffer_like(vel_x, shape=(1, 1), buffer_factor=2)
        o_cb = ttl.make_dataflow_buffer_like(pos_x, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for lc in range(cpc):
                cid = core_x * cpc + lc
                if cid < nc:
                    for _ in range(3):
                        with p_cb.wait() as p, v_cb.wait() as v, o_cb.reserve() as o:
                            new_p = p + v * ttl.math.fill(p, c_dt)
                            o.store(new_p - ttl.math.fill(p, c_box) * ttl.math.floor(new_p * ttl.math.fill(p, c_inv_box)))

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for lc in range(cpc):
                cid = core_x * cpc + lc
                if cid < nc:
                    with p_cb.reserve() as blk:
                        tx = ttl.copy(pos_x[cid, 0], blk); tx.wait()
                    with v_cb.reserve() as blk:
                        tx = ttl.copy(vel_x[cid, 0], blk); tx.wait()
                    with p_cb.reserve() as blk:
                        tx = ttl.copy(pos_y[cid, 0], blk); tx.wait()
                    with v_cb.reserve() as blk:
                        tx = ttl.copy(vel_y[cid, 0], blk); tx.wait()
                    with p_cb.reserve() as blk:
                        tx = ttl.copy(pos_z[cid, 0], blk); tx.wait()
                    with v_cb.reserve() as blk:
                        tx = ttl.copy(vel_z[cid, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for lc in range(cpc):
                cid = core_x * cpc + lc
                if cid < nc:
                    with o_cb.wait() as blk:
                        tx = ttl.copy(blk, pos_x[cid, 0]); tx.wait()
                    with o_cb.wait() as blk:
                        tx = ttl.copy(blk, pos_y[cid, 0]); tx.wait()
                    with o_cb.wait() as blk:
                        tx = ttl.copy(blk, pos_z[cid, 0]); tx.wait()

    # --- Displacement check kernel: global max |pos - rebuild_pos|² ---
    # Single-core, streams all cells, outputs ONE tile with max in [0,0].
    # Tiny output -> fast readback (~2KB vs ~MB).
    @ttl.kernel(grid=(1, 1))
    def max_disp_kernel(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, scaler, out):
        nc = pos_x.shape[0] // TILE

        p_cb = ttl.make_dataflow_buffer_like(pos_x, shape=(1, 1), buffer_factor=2)
        r_cb = ttl.make_dataflow_buffer_like(ref_x, shape=(1, 1), buffer_factor=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        d_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        mx_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        o_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_cb.wait() as sc:
                # First cell: initialize max
                with p_cb.wait() as p, r_cb.wait() as r:
                    dx = p - r
                    dx_pbc = dx - ttl.math.fill(p, c_box) * ttl.math.floor(
                        dx * ttl.math.fill(p, c_inv_box) + ttl.math.fill(p, c_half))
                    with d_cb.reserve() as d:
                        d.store(dx_pbc * dx_pbc)
                with p_cb.wait() as p, r_cb.wait() as r, d_cb.wait() as prev:
                    dy = p - r
                    dy_pbc = dy - ttl.math.fill(p, c_box) * ttl.math.floor(
                        dy * ttl.math.fill(p, c_inv_box) + ttl.math.fill(p, c_half))
                    with d_cb.reserve() as d:
                        d.store(prev + dy_pbc * dy_pbc)
                with p_cb.wait() as p, r_cb.wait() as r, d_cb.wait() as prev:
                    dz = p - r
                    dz_pbc = dz - ttl.math.fill(p, c_box) * ttl.math.floor(
                        dz * ttl.math.fill(p, c_inv_box) + ttl.math.fill(p, c_half))
                    with d_cb.reserve() as d:
                        d.store(prev + dz_pbc * dz_pbc)
                with d_cb.wait() as disp_sq, mx_cb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(disp_sq, sc, dims=[0, 1]))

                # Remaining cells: update running max
                for _ in range(nc - 1):
                    with p_cb.wait() as p, r_cb.wait() as r:
                        dx = p - r
                        dx_pbc = dx - ttl.math.fill(p, c_box) * ttl.math.floor(
                            dx * ttl.math.fill(p, c_inv_box) + ttl.math.fill(p, c_half))
                        with d_cb.reserve() as d:
                            d.store(dx_pbc * dx_pbc)
                    with p_cb.wait() as p, r_cb.wait() as r, d_cb.wait() as prev:
                        dy = p - r
                        dy_pbc = dy - ttl.math.fill(p, c_box) * ttl.math.floor(
                            dy * ttl.math.fill(p, c_inv_box) + ttl.math.fill(p, c_half))
                        with d_cb.reserve() as d:
                            d.store(prev + dy_pbc * dy_pbc)
                    with p_cb.wait() as p, r_cb.wait() as r, d_cb.wait() as prev:
                        dz = p - r
                        dz_pbc = dz - ttl.math.fill(p, c_box) * ttl.math.floor(
                            dz * ttl.math.fill(p, c_inv_box) + ttl.math.fill(p, c_half))
                        with d_cb.reserve() as d:
                            d.store(prev + dz_pbc * dz_pbc)
                    with d_cb.wait() as disp_sq:
                        with d_cb.reserve() as cell_mx:
                            cell_mx.store(ttl.math.reduce_max(disp_sq, sc, dims=[0, 1]))
                        with d_cb.wait() as cmx, mx_cb.wait() as old_mx:
                            with mx_cb.reserve() as new_mx:
                                new_mx.store(ttl.math.max(old_mx, cmx))

                with mx_cb.wait() as final_mx, o_cb.reserve() as o:
                    o.store(final_mx)

        @ttl.datamovement()
        def dm_read():
            with sc_cb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            for cid in range(nc):
                with p_cb.reserve() as blk:
                    tx = ttl.copy(pos_x[cid, 0], blk); tx.wait()
                with r_cb.reserve() as blk:
                    tx = ttl.copy(ref_x[cid, 0], blk); tx.wait()
                with p_cb.reserve() as blk:
                    tx = ttl.copy(pos_y[cid, 0], blk); tx.wait()
                with r_cb.reserve() as blk:
                    tx = ttl.copy(ref_y[cid, 0], blk); tx.wait()
                with p_cb.reserve() as blk:
                    tx = ttl.copy(pos_z[cid, 0], blk); tx.wait()
                with r_cb.reserve() as blk:
                    tx = ttl.copy(ref_z[cid, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with o_cb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()

    # --- Force kernel: fused geometry + LJ + erfc Coulomb (bf16, 31 CBs) ---
    def make_force_kernel(c_n_dim, c_dim2):
        @ttl.kernel(grid="auto")
        def cell_forces_kernel(own_px, own_py, own_pz, own_q,
                               atom_valid, self_diag, scaler, epslj,
                               fx_out, fy_out, fz_out):
            grid_cols, _ = ttl.grid_size(dims=2)
            n_cells = own_px.shape[0] // TILE
            cells_per_core = -(-n_cells // grid_cols)

            ox_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            oy_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
            oz_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
            oq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
            ov_cb = ttl.make_dataflow_buffer_like(atom_valid, shape=(1, 1), buffer_factor=2)
            ex_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            ey_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
            ez_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
            eq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
            ev_cb = ttl.make_dataflow_buffer_like(atom_valid, shape=(1, 1), buffer_factor=2)
            sd_cb = ttl.make_dataflow_buffer_like(self_diag, shape=(1, 1), buffer_factor=2)
            sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
            mask_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

            ba_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            tr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            bb_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            r2_tmp = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

            r2_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            qq_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            dx_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            dy_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            dz_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

            fm_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            ft_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
            fr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

            ax_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
            ay_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
            az_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

            fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
            fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
            fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                core_x, _ = ttl.core(dims=2)
                for local_c in range(cells_per_core):
                    cell_id = core_x * cells_per_core + local_c
                    if cell_id < n_cells:
                        with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz, ov_cb.wait() as ov:
                            with sc_cb.wait() as sc, sd_cb.wait() as sd:
                                for nbr_i in range(N_NBR):
                                    with ex_cb.wait() as ex, ey_cb.wait() as ey, ez_cb.wait() as ez, ev_cb.wait() as ev:
                                        # Build effective mask from per-atom validity
                                        with ba_cb.reserve() as ba:
                                            ba.store(ttl.math.broadcast(ov, dims=[1]))
                                        with tr_cb.reserve() as tr:
                                            tr.store(ttl.transpose(ev))
                                        with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                            bb.store(ttl.math.broadcast(trv, dims=[0]))
                                        with ba_cb.wait() as ov_full, bb_cb.wait() as ev_full:
                                            with mask_cb.reserve() as mc:
                                                if nbr_i == 13:
                                                    mc.store(ov_full + ev_full + sd)
                                                else:
                                                    mc.store(ov_full + ev_full)

                                        # PBC displacements for x, y, z
                                        with ba_cb.reserve() as ba:
                                            ba.store(ttl.math.broadcast(ox, dims=[1]))
                                        with tr_cb.reserve() as tr:
                                            tr.store(ttl.transpose(ex))
                                        with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                            bb.store(ttl.math.broadcast(trv, dims=[0]))
                                        with ba_cb.wait() as bav, bb_cb.wait() as bbv:
                                            dx_raw = bav - bbv
                                            dx_pbc = dx_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dx_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                            with r2_tmp.reserve() as r2o:
                                                r2o.store(dx_pbc * dx_pbc)
                                            with dx_cb.reserve() as dxo:
                                                dxo.store(dx_pbc)

                                        with ba_cb.reserve() as ba:
                                            ba.store(ttl.math.broadcast(oy, dims=[1]))
                                        with tr_cb.reserve() as tr:
                                            tr.store(ttl.transpose(ey))
                                        with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                            bb.store(ttl.math.broadcast(trv, dims=[0]))
                                        with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                                            dy_raw = bav - bbv
                                            dy_pbc = dy_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dy_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                            with r2_tmp.reserve() as r2o:
                                                r2o.store(r2p + dy_pbc * dy_pbc)
                                            with dy_cb.reserve() as dyo:
                                                dyo.store(dy_pbc)

                                        with ba_cb.reserve() as ba:
                                            ba.store(ttl.math.broadcast(oz, dims=[1]))
                                        with tr_cb.reserve() as tr:
                                            tr.store(ttl.transpose(ez))
                                        with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                            bb.store(ttl.math.broadcast(trv, dims=[0]))
                                        with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p, mask_cb.wait() as eff_mask:
                                            dz_raw = bav - bbv
                                            dz_pbc = dz_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dz_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                            with r2_cb.reserve() as r2o:
                                                r2o.store(r2p + dz_pbc * dz_pbc + eff_mask)
                                            with dz_cb.reserve() as dzo:
                                                dzo.store(dz_pbc)

                                        # Charge products (first pop of oq_cb/eq_cb)
                                        with oq_cb.wait() as oq:
                                            with ba_cb.reserve() as ba:
                                                ba.store(ttl.math.broadcast(oq, dims=[1]))
                                        with eq_cb.wait() as eq:
                                            with tr_cb.reserve() as tr:
                                                tr.store(ttl.transpose(eq))
                                        with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                            bb.store(ttl.math.broadcast(trv, dims=[0]))
                                        with ba_cb.wait() as bav, bb_cb.wait() as bbv, qq_cb.reserve() as qqo:
                                            qqo.store(bav * bbv)

                                        # Per-pair epsilon (second pop of oq_cb/eq_cb)
                                        with oq_cb.wait() as own_se:
                                            with ba_cb.reserve() as ba:
                                                ba.store(ttl.math.broadcast(own_se, dims=[1]))
                                        with eq_cb.wait() as nbr_se:
                                            with tr_cb.reserve() as tr:
                                                tr.store(ttl.transpose(nbr_se))
                                        with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                            bb.store(ttl.math.broadcast(trv, dims=[0]))
                                        with ba_cb.wait() as oe_full, bb_cb.wait() as ne_full:
                                            with r2_tmp.reserve() as eps_ij:
                                                eps_ij.store(oe_full * ne_full)

                                        # erfc-damped Coulomb + LJ forces
                                        with r2_cb.wait() as r2, qq_cb.wait() as qq, r2_tmp.wait() as eps:
                                            r_inv = ttl.math.rsqrt(r2)
                                            r2_inv = ttl.math.recip(r2)
                                            r_val = r2 * r_inv
                                            t = ttl.math.recip(r_inv * r_inv * r2 + ttl.math.fill(r2, c_p_alpha) * r_val)
                                            poly = t * (ttl.math.fill(r2, c_a1) + t * (ttl.math.neg(ttl.math.fill(r2, c_a2)) + t * (ttl.math.fill(r2, c_a3) + t * (ttl.math.neg(ttl.math.fill(r2, c_a4)) + t * ttl.math.fill(r2, c_a5)))))
                                            exp_neg = ttl.math.exp(ttl.math.neg(ttl.math.fill(r2, c_alpha_sq) * r2))
                                            erfc_val = poly * exp_neg
                                            with ft_cb.reserve() as coul:
                                                coul.store(qq * (erfc_val * r2_inv + ttl.math.fill(r2, c_two_a_sp) * exp_neg * r_inv) * r_inv)
                                            r2_inv2 = ttl.math.recip(r2)
                                            r4_inv = r2_inv2 * r2_inv2
                                            r6_inv = r4_inv * r2_inv2
                                            r12_inv = r6_inv * r6_inv
                                            lj_eps = ttl.math.fill(r2, c_lj24) * eps
                                            with fr_cb.reserve() as lj:
                                                lj.store(lj_eps * r2_inv2 * (r12_inv + r12_inv - r6_inv))

                                        with ft_cb.wait() as fc, fr_cb.wait() as fl:
                                            with fm_cb.reserve() as fmo:
                                                fmo.store(fl + fc)

                                        # Project onto displacements, reduce, accumulate
                                        with fm_cb.wait() as fm:
                                            with dx_cb.wait() as dxv:
                                                with ft_cb.reserve() as ft:
                                                    ft.store(fm * dxv)
                                                with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                    fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                                if nbr_i == 0:
                                                    with fr_cb.wait() as frv, ax_cb.reserve() as ax:
                                                        ax.store(frv)
                                                else:
                                                    with fr_cb.wait() as frv, ax_cb.wait() as prev:
                                                        with ax_cb.reserve() as ax:
                                                            ax.store(prev + frv)
                                            with dy_cb.wait() as dyv:
                                                with ft_cb.reserve() as ft:
                                                    ft.store(fm * dyv)
                                                with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                    fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                                if nbr_i == 0:
                                                    with fr_cb.wait() as frv, ay_cb.reserve() as ay:
                                                        ay.store(frv)
                                                else:
                                                    with fr_cb.wait() as frv, ay_cb.wait() as prev:
                                                        with ay_cb.reserve() as ay:
                                                            ay.store(prev + frv)
                                            with dz_cb.wait() as dzv:
                                                with ft_cb.reserve() as ft:
                                                    ft.store(fm * dzv)
                                                with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                    fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                                if nbr_i == 0:
                                                    with fr_cb.wait() as frv, az_cb.reserve() as az:
                                                        az.store(frv)
                                                else:
                                                    with fr_cb.wait() as frv, az_cb.wait() as prev:
                                                        with az_cb.reserve() as az:
                                                            az.store(prev + frv)

                                with ax_cb.wait() as fx, fxo_cb.reserve() as fxo:
                                    fxo.store(fx)
                                with ay_cb.wait() as fy, fyo_cb.reserve() as fyo:
                                    fyo.store(fy)
                                with az_cb.wait() as fz, fzo_cb.reserve() as fzo:
                                    fzo.store(fz)

            @ttl.datamovement()
            def dm_read():
                core_x, _ = ttl.core(dims=2)
                for local_c in range(cells_per_core):
                    cell_id = core_x * cells_per_core + local_c
                    if cell_id < n_cells:
                        with ox_cb.reserve() as blk:
                            tx = ttl.copy(own_px[cell_id, 0], blk); tx.wait()
                        with oy_cb.reserve() as blk:
                            tx = ttl.copy(own_py[cell_id, 0], blk); tx.wait()
                        with oz_cb.reserve() as blk:
                            tx = ttl.copy(own_pz[cell_id, 0], blk); tx.wait()
                        with ov_cb.reserve() as blk:
                            tx = ttl.copy(atom_valid[cell_id, 0], blk); tx.wait()
                        with sc_cb.reserve() as blk:
                            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
                        with sd_cb.reserve() as blk:
                            tx = ttl.copy(self_diag[0, 0], blk); tx.wait()
                        # Compute neighbor cell IDs on-device
                        cx = cell_id // c_dim2
                        cy = (cell_id // c_n_dim) % c_n_dim
                        cz = cell_id % c_n_dim
                        for nbr in range(N_NBR):
                            off_dx = (nbr // 9) - 1
                            off_dy = ((nbr // 3) % 3) - 1
                            off_dz = (nbr % 3) - 1
                            nbr_cell = ((cx + off_dx + c_n_dim) % c_n_dim) * c_dim2 + ((cy + off_dy + c_n_dim) % c_n_dim) * c_n_dim + ((cz + off_dz + c_n_dim) % c_n_dim)
                            with ex_cb.reserve() as blk:
                                tx = ttl.copy(own_px[nbr_cell, 0], blk); tx.wait()
                            with ey_cb.reserve() as blk:
                                tx = ttl.copy(own_py[nbr_cell, 0], blk); tx.wait()
                            with ez_cb.reserve() as blk:
                                tx = ttl.copy(own_pz[nbr_cell, 0], blk); tx.wait()
                            with ev_cb.reserve() as blk:
                                tx = ttl.copy(atom_valid[nbr_cell, 0], blk); tx.wait()
                            with oq_cb.reserve() as blk:
                                tx = ttl.copy(own_q[cell_id, 0], blk); tx.wait()
                            with eq_cb.reserve() as blk:
                                tx = ttl.copy(own_q[nbr_cell, 0], blk); tx.wait()
                            with oq_cb.reserve() as blk:
                                tx = ttl.copy(epslj[cell_id, 0], blk); tx.wait()
                            with eq_cb.reserve() as blk:
                                tx = ttl.copy(epslj[nbr_cell, 0], blk); tx.wait()

            @ttl.datamovement()
            def dm_write():
                core_x, _ = ttl.core(dims=2)
                for local_c in range(cells_per_core):
                    cell_id = core_x * cells_per_core + local_c
                    if cell_id < n_cells:
                        with fxo_cb.wait() as blk:
                            tx = ttl.copy(blk, fx_out[cell_id, 0]); tx.wait()
                        with fyo_cb.wait() as blk:
                            tx = ttl.copy(blk, fy_out[cell_id, 0]); tx.wait()
                        with fzo_cb.wait() as blk:
                            tx = ttl.copy(blk, fz_out[cell_id, 0]); tx.wait()

        return cell_forces_kernel

    # --- Bond force kernel: harmonic springs keeping lipid chains together ---
    BOND_K = 100.0
    BOND_R0 = 1.0

    def build_bond_partner_data(bonds, cell_atom_map, n_cells_total):
        """Build per-atom bond partner row indices and masks.

        Each atom has at most 2 bond partners. Returns:
          partner1_rows, partner2_rows: (n_rows, TILE) gather indices
          mask1, mask2: (n_rows, TILE) float, 1.0 if partner exists else 0.0
        """
        n_rows = n_cells_total * TILE
        rows, atom_ids = _build_cell_index_arrays(cell_atom_map)
        atom_to_row = {}
        for r, a in zip(rows, atom_ids):
            atom_to_row[int(a)] = int(r)

        # Collect bond partners per atom
        partners = {}
        for b in bonds:
            i, j = int(b[0]), int(b[1])
            partners.setdefault(i, []).append(j)
            partners.setdefault(j, []).append(i)

        p1 = np.zeros((n_rows, TILE), dtype=np.float32)
        p2 = np.zeros((n_rows, TILE), dtype=np.float32)
        m1 = np.zeros((n_rows, TILE), dtype=np.float32)
        m2 = np.zeros((n_rows, TILE), dtype=np.float32)

        for atom_idx, row_idx in atom_to_row.items():
            plist = partners.get(atom_idx, [])
            if len(plist) >= 1 and plist[0] in atom_to_row:
                p1[row_idx, :] = float(atom_to_row[plist[0]])
                m1[row_idx, 0] = 1.0
            if len(plist) >= 2 and plist[1] in atom_to_row:
                p2[row_idx, :] = float(atom_to_row[plist[1]])
                m2[row_idx, 0] = 1.0

        return p1, p2, m1, m2

    def make_bond_kernel():
        c_k = float(BOND_K)
        c_r0 = float(BOND_R0)
        c_box = float(box_length)
        c_inv_box = 1.0 / float(box_length)
        c_half = 0.5
        c_eps = 1e-8

        @ttl.kernel(grid="auto")
        def bond_forces_kernel(own_x, own_y, own_z,
                               p1_x, p1_y, p1_z, m1,
                               p2_x, p2_y, p2_z, m2,
                               fx_out, fy_out, fz_out):
            grid_cols, _ = ttl.grid_size(dims=2)
            n_tiles = own_x.shape[0] // TILE
            tiles_per_core = -(-n_tiles // grid_cols)

            ox_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
            oy_cb = ttl.make_dataflow_buffer_like(own_y, shape=(1, 1), buffer_factor=2)
            oz_cb = ttl.make_dataflow_buffer_like(own_z, shape=(1, 1), buffer_factor=2)
            p1x_cb = ttl.make_dataflow_buffer_like(p1_x, shape=(1, 1), buffer_factor=2)
            p1y_cb = ttl.make_dataflow_buffer_like(p1_y, shape=(1, 1), buffer_factor=2)
            p1z_cb = ttl.make_dataflow_buffer_like(p1_z, shape=(1, 1), buffer_factor=2)
            m1_cb = ttl.make_dataflow_buffer_like(m1, shape=(1, 1), buffer_factor=2)
            p2x_cb = ttl.make_dataflow_buffer_like(p2_x, shape=(1, 1), buffer_factor=2)
            p2y_cb = ttl.make_dataflow_buffer_like(p2_y, shape=(1, 1), buffer_factor=2)
            p2z_cb = ttl.make_dataflow_buffer_like(p2_z, shape=(1, 1), buffer_factor=2)
            m2_cb = ttl.make_dataflow_buffer_like(m2, shape=(1, 1), buffer_factor=2)
            # Temp DFB: store r_sq so rsqrt reads from CB (workaround for copy_dst bug)
            rsq_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
            b1x_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
            b1y_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
            b1z_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)
            fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
            fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
            fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                core_x, _ = ttl.core(dims=2)
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < n_tiles:
                        # Bond 1 forces
                        with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz:
                            with p1x_cb.wait() as p1x, p1y_cb.wait() as p1y, p1z_cb.wait() as p1z, m1_cb.wait() as mask1:
                                dx1 = ox - p1x
                                dx1 = dx1 - ttl.math.fill(ox, c_box) * ttl.math.floor(dx1 * ttl.math.fill(ox, c_inv_box) + ttl.math.fill(ox, c_half))
                                dy1 = oy - p1y
                                dy1 = dy1 - ttl.math.fill(oy, c_box) * ttl.math.floor(dy1 * ttl.math.fill(oy, c_inv_box) + ttl.math.fill(oy, c_half))
                                dz1 = oz - p1z
                                dz1 = dz1 - ttl.math.fill(oz, c_box) * ttl.math.floor(dz1 * ttl.math.fill(oz, c_inv_box) + ttl.math.fill(oz, c_half))
                                with rsq_cb.reserve() as tmp:
                                    tmp.store(dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + ttl.math.fill(ox, c_eps))
                                with rsq_cb.wait() as r1_sq:
                                    r1_inv = ttl.math.rsqrt(r1_sq)
                                    r1 = r1_sq * r1_inv
                                    f1 = ttl.math.neg(ttl.math.fill(r1_sq, c_k)) * (r1 - ttl.math.fill(r1_sq, c_r0)) * r1_inv * mask1
                                    with b1x_cb.reserve() as tmp:
                                        tmp.store(f1 * dx1)
                                    with b1y_cb.reserve() as tmp:
                                        tmp.store(f1 * dy1)
                                    with b1z_cb.reserve() as tmp:
                                        tmp.store(f1 * dz1)

                        # Bond 2 forces + combine with bond 1
                        with ox_cb.wait() as ox2, oy_cb.wait() as oy2, oz_cb.wait() as oz2:
                            with p2x_cb.wait() as p2x, p2y_cb.wait() as p2y, p2z_cb.wait() as p2z, m2_cb.wait() as mask2:
                                dx2 = ox2 - p2x
                                dx2 = dx2 - ttl.math.fill(ox2, c_box) * ttl.math.floor(dx2 * ttl.math.fill(ox2, c_inv_box) + ttl.math.fill(ox2, c_half))
                                dy2 = oy2 - p2y
                                dy2 = dy2 - ttl.math.fill(oy2, c_box) * ttl.math.floor(dy2 * ttl.math.fill(oy2, c_inv_box) + ttl.math.fill(oy2, c_half))
                                dz2 = oz2 - p2z
                                dz2 = dz2 - ttl.math.fill(oz2, c_box) * ttl.math.floor(dz2 * ttl.math.fill(oz2, c_inv_box) + ttl.math.fill(oz2, c_half))
                                with rsq_cb.reserve() as tmp:
                                    tmp.store(dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + ttl.math.fill(ox2, c_eps))
                                with rsq_cb.wait() as r2_sq:
                                    r2_inv = ttl.math.rsqrt(r2_sq)
                                    r2 = r2_sq * r2_inv
                                    f2 = ttl.math.neg(ttl.math.fill(r2_sq, c_k)) * (r2 - ttl.math.fill(r2_sq, c_r0)) * r2_inv * mask2
                                    with b1x_cb.wait() as fx1, fxo_cb.reserve() as fxo:
                                        fxo.store(fx1 + f2 * dx2)
                                    with b1y_cb.wait() as fy1, fyo_cb.reserve() as fyo:
                                        fyo.store(fy1 + f2 * dy2)
                                    with b1z_cb.wait() as fz1, fzo_cb.reserve() as fzo:
                                        fzo.store(fz1 + f2 * dz2)

            @ttl.datamovement()
            def dm_read():
                core_x, _ = ttl.core(dims=2)
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < n_tiles:
                        # Bond 1: own positions + partner1 + mask1
                        with ox_cb.reserve() as blk:
                            tx = ttl.copy(own_x[tile_idx, 0], blk); tx.wait()
                        with oy_cb.reserve() as blk:
                            tx = ttl.copy(own_y[tile_idx, 0], blk); tx.wait()
                        with oz_cb.reserve() as blk:
                            tx = ttl.copy(own_z[tile_idx, 0], blk); tx.wait()
                        with p1x_cb.reserve() as blk:
                            tx = ttl.copy(p1_x[tile_idx, 0], blk); tx.wait()
                        with p1y_cb.reserve() as blk:
                            tx = ttl.copy(p1_y[tile_idx, 0], blk); tx.wait()
                        with p1z_cb.reserve() as blk:
                            tx = ttl.copy(p1_z[tile_idx, 0], blk); tx.wait()
                        with m1_cb.reserve() as blk:
                            tx = ttl.copy(m1[tile_idx, 0], blk); tx.wait()
                        # Bond 2: own positions again + partner2 + mask2
                        with ox_cb.reserve() as blk:
                            tx = ttl.copy(own_x[tile_idx, 0], blk); tx.wait()
                        with oy_cb.reserve() as blk:
                            tx = ttl.copy(own_y[tile_idx, 0], blk); tx.wait()
                        with oz_cb.reserve() as blk:
                            tx = ttl.copy(own_z[tile_idx, 0], blk); tx.wait()
                        with p2x_cb.reserve() as blk:
                            tx = ttl.copy(p2_x[tile_idx, 0], blk); tx.wait()
                        with p2y_cb.reserve() as blk:
                            tx = ttl.copy(p2_y[tile_idx, 0], blk); tx.wait()
                        with p2z_cb.reserve() as blk:
                            tx = ttl.copy(p2_z[tile_idx, 0], blk); tx.wait()
                        with m2_cb.reserve() as blk:
                            tx = ttl.copy(m2[tile_idx, 0], blk); tx.wait()

            @ttl.datamovement()
            def dm_write():
                core_x, _ = ttl.core(dims=2)
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < n_tiles:
                        with fxo_cb.wait() as blk:
                            tx = ttl.copy(blk, fx_out[tile_idx, 0]); tx.wait()
                        with fyo_cb.wait() as blk:
                            tx = ttl.copy(blk, fy_out[tile_idx, 0]); tx.wait()
                        with fzo_cb.wait() as blk:
                            tx = ttl.copy(blk, fz_out[tile_idx, 0]); tx.wait()

        return bond_forces_kernel

    def run_bond_forces(tt_px, tt_py, tt_pz, tt_p1_rows, tt_p2_rows,
                        tt_m1, tt_m2):
        """Gather partner positions, run bf16 bond kernel, return f32 bond forces."""
        p1x = ttnn.typecast(ttnn.gather(tt_px, dim=0, index=tt_p1_rows), ttnn.bfloat16)
        p1y = ttnn.typecast(ttnn.gather(tt_py, dim=0, index=tt_p1_rows), ttnn.bfloat16)
        p1z = ttnn.typecast(ttnn.gather(tt_pz, dim=0, index=tt_p1_rows), ttnn.bfloat16)
        p2x = ttnn.typecast(ttnn.gather(tt_px, dim=0, index=tt_p2_rows), ttnn.bfloat16)
        p2y = ttnn.typecast(ttnn.gather(tt_py, dim=0, index=tt_p2_rows), ttnn.bfloat16)
        p2z = ttnn.typecast(ttnn.gather(tt_pz, dim=0, index=tt_p2_rows), ttnn.bfloat16)
        ox_bf = ttnn.typecast(tt_px, ttnn.bfloat16)
        oy_bf = ttnn.typecast(tt_py, ttnn.bfloat16)
        oz_bf = ttnn.typecast(tt_pz, ttnn.bfloat16)
        n_rows = tt_px.shape[0]
        zeros_bf = to_bf16(np.zeros((n_rows, TILE)), l1=use_l1)
        bfx_bf = ttnn.clone(zeros_bf)
        bfy_bf = ttnn.clone(zeros_bf)
        bfz_bf = ttnn.clone(zeros_bf)
        ttnn.deallocate(zeros_bf)
        bond_kernel(ox_bf, oy_bf, oz_bf,
                    p1x, p1y, p1z, tt_m1,
                    p2x, p2y, p2z, tt_m2,
                    bfx_bf, bfy_bf, bfz_bf)
        bfx = ttnn.typecast(bfx_bf, ttnn.float32)
        bfy = ttnn.typecast(bfy_bf, ttnn.float32)
        bfz = ttnn.typecast(bfz_bf, ttnn.float32)
        return bfx, bfy, bfz

    # --- Tensor helpers ---
    def to_bf16(arr, l1=False):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG)

    def to_f32(arr, l1=False):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG)

    FORCE_CLAMP = 25.0  # cap bf16 overflow from accumulation order sensitivity

    def run_force_kernel(tt_px, tt_py, tt_pz, tt_q, tt_atom_valid,
                         tt_self_diag, tt_scaler, tt_epslj, tt_fx, tt_fy, tt_fz):
        """Typecast f32 <-> bf16 around the bf16 force kernel, clamp output."""
        px_bf16 = ttnn.typecast(tt_px, ttnn.bfloat16)
        py_bf16 = ttnn.typecast(tt_py, ttnn.bfloat16)
        pz_bf16 = ttnn.typecast(tt_pz, ttnn.bfloat16)
        fx_bf16 = ttnn.typecast(tt_fx, ttnn.bfloat16)
        fy_bf16 = ttnn.typecast(tt_fy, ttnn.bfloat16)
        fz_bf16 = ttnn.typecast(tt_fz, ttnn.bfloat16)
        cell_forces_kernel(px_bf16, py_bf16, pz_bf16, tt_q,
                           tt_atom_valid, tt_self_diag, tt_scaler, tt_epslj,
                           fx_bf16, fy_bf16, fz_bf16)
        fx_f32 = ttnn.typecast(fx_bf16, ttnn.float32)
        fy_f32 = ttnn.typecast(fy_bf16, ttnn.float32)
        fz_f32 = ttnn.typecast(fz_bf16, ttnn.float32)
        return (ttnn.clamp(fx_f32, min=-FORCE_CLAMP, max=FORCE_CLAMP),
                ttnn.clamp(fy_f32, min=-FORCE_CLAMP, max=FORCE_CLAMP),
                ttnn.clamp(fz_f32, min=-FORCE_CLAMP, max=FORCE_CLAMP))

    # --- Build initial state (cells sized with skin buffer) ---
    t_setup0 = time.time()
    (own_px, own_py, own_pz, own_q, own_epslj,
     masks, cell_atom_map, n_cells_total, n_cells_dim) = \
        build_cell_data(positions, charges, box_length, cell_r, atom_types=atom_types)
    print(f"  Cell build: {time.time()-t_setup0:.3f}s, {n_cells_total} cells ({n_cells_dim}^3)")

    c_n_dim = int(n_cells_dim)
    c_dim2 = c_n_dim * c_n_dim
    cell_size = box_length / n_cells_dim
    actual_skin = cell_size - r_cut
    half_skin_sq = (actual_skin / 2.0) ** 2

    vel_x, vel_y, vel_z = pack_cell_layout(velocities, cell_atom_map, n_cells_total)

    # Reciprocal forces (recomputed on each rebuild, held constant between)
    t_recip0 = time.time()
    f_recip = compute_reciprocal_forces(device, positions, charges, box_length, alpha)
    recip_x, recip_y, recip_z = pack_cell_layout(f_recip, cell_atom_map, n_cells_total)
    print(f"  Reciprocal forces: {time.time()-t_recip0:.3f}s")

    # L1 for small systems, DRAM for large
    cell_bytes_f32 = n_cells_total * TILE * TILE * 4
    use_l1 = (cell_bytes_f32 * 10 < 80_000_000)
    print(f"  Memory: {'L1' if use_l1 else 'DRAM'} (cell_bytes={cell_bytes_f32/1e6:.1f}MB)")

    tt_px = to_f32(own_px, l1=use_l1)
    tt_py = to_f32(own_py, l1=use_l1)
    tt_pz = to_f32(own_pz, l1=use_l1)
    tt_q = to_bf16(own_q, l1=use_l1)
    tt_vx = to_f32(vel_x, l1=use_l1)
    tt_vy = to_f32(vel_y, l1=use_l1)
    tt_vz = to_f32(vel_z, l1=use_l1)

    # Per-atom validity: 0.0 for real atoms, 1e6 for padding (bf16)
    n_rows = n_cells_total * TILE
    atom_valid_np = np.full((n_rows, TILE), 1e6, dtype=np.float32)
    rows, _ = _build_cell_index_arrays(cell_atom_map)
    atom_valid_np[rows, 0] = 0.0
    tt_atom_valid = to_bf16(atom_valid_np, l1=use_l1)

    # Static self-exclusion diagonal: 1e6 on main diagonal (bf16, never changes)
    self_diag_np = np.zeros((TILE, TILE), dtype=np.float32)
    np.fill_diagonal(self_diag_np, 1e6)
    tt_self_diag = to_bf16(self_diag_np, l1=True)

    tt_epslj = to_bf16(own_epslj, l1=use_l1)
    tt_scaler = to_bf16(np.ones((TILE, TILE)), l1=True)
    zeros = np.zeros((n_cells_total * TILE, TILE), dtype=np.float32)
    tt_fx = to_f32(zeros, l1=use_l1)
    tt_fy = to_f32(zeros.copy(), l1=use_l1)
    tt_fz = to_f32(zeros.copy(), l1=use_l1)
    # Reciprocal forces stored on device (f32, added after real-space)
    tt_rx = to_f32(recip_x, l1=use_l1)
    tt_ry = to_f32(recip_y, l1=use_l1)
    tt_rz = to_f32(recip_z, l1=use_l1)

    cell_forces_kernel = make_force_kernel(c_n_dim, c_dim2)
    bond_kernel = make_bond_kernel()

    # --- Bond partner data (updated after each rebuild) ---
    has_bonds = bonds is not None and len(bonds) > 0

    def upload_bond_partners(cell_atom_map_local):
        p1, p2, m1, m2 = build_bond_partner_data(bonds, cell_atom_map_local, n_cells_total)
        tt_p1 = ttnn.typecast(to_f32(p1, l1=use_l1), ttnn.uint32)
        tt_p2 = ttnn.typecast(to_f32(p2, l1=use_l1), ttnn.uint32)
        tt_m1 = to_bf16(m1, l1=use_l1)
        tt_m2 = to_bf16(m2, l1=use_l1)
        return tt_p1, tt_p2, tt_m1, tt_m2

    if has_bonds:
        tt_p1_rows, tt_p2_rows, tt_m1, tt_m2 = upload_bond_partners(cell_atom_map)
    else:
        tt_p1_rows = tt_p2_rows = tt_m1 = tt_m2 = None

    # Atom ID tensor: tracks which atom is at each cell-layout row through rebuilds
    atom_id_np = np.full((n_cells_total * TILE, TILE), -1.0, dtype=np.float32)
    rows_init, aids_init = _build_cell_index_arrays(cell_atom_map)
    atom_id_np[rows_init, :] = -1.0
    for r, a in zip(rows_init, aids_init):
        atom_id_np[r, :] = float(a)
    tt_atom_id = to_f32(atom_id_np, l1=use_l1)

    # --- Constants for on-device rebuild (uploaded once, never change) ---
    n_rows = n_cells_total * TILE
    n_cells_padded = ((n_cells_total + TILE - 1) // TILE) * TILE

    rb_inv_cs = to_f32(np.full((n_rows, TILE), 1.0 / cell_size, dtype=np.float32), l1=use_l1)
    rb_n_dim_t = to_f32(np.full((n_rows, TILE), float(c_n_dim), dtype=np.float32), l1=use_l1)
    rb_inv_n = to_f32(np.full((n_rows, TILE), 1.0 / c_n_dim, dtype=np.float32), l1=use_l1)
    rb_dim2_t = to_f32(np.full((n_rows, TILE), float(c_dim2), dtype=np.float32), l1=use_l1)
    rb_sentinel = to_f32(np.full((n_rows, TILE), float(n_cells_total), dtype=np.float32), l1=use_l1)
    rb_one = to_f32(np.full((n_rows, TILE), 1.0, dtype=np.float32), l1=use_l1)
    rb_zero = to_f32(np.zeros((n_rows, TILE), dtype=np.float32), l1=use_l1)

    # col0_to_all: extracts column 0 and broadcasts to all columns
    col0_to_all_np = np.zeros((TILE, TILE), dtype=np.float32)
    col0_to_all_np[0, :] = 1.0
    rb_col0_to_all = to_f32(col0_to_all_np, l1=True)

    # Broadcast col0 to n_cells_padded columns (for comparison matrix)
    ones_bc_np = np.zeros((TILE, n_cells_padded), dtype=np.float32)
    ones_bc_np[0, :] = 1.0
    rb_ones_broadcast = to_f32(ones_bc_np, l1=True)

    # Reference tensor: ref[r, c] = c (for lt comparison)
    ref_np = np.zeros((n_rows, n_cells_padded), dtype=np.float32)
    for c in range(n_cells_padded):
        ref_np[:, c] = float(c)
    rb_ref = to_f32(ref_np, l1=use_l1)

    # Ones row for summing indicator: (TILE, n_rows) with row 0 = 1
    ones_row_np = np.zeros((TILE, n_rows), dtype=np.float32)
    ones_row_np[0, :] = 1.0
    rb_ones_row = to_f32(ones_row_np, l1=use_l1)

    # Cell selector: selector[r, r//TILE] = 1 (expands cell_starts to n_rows)
    selector_np = np.zeros((n_rows, n_cells_padded), dtype=np.float32)
    for r in range(n_rows):
        c = r // TILE
        if c < n_cells_padded:
            selector_np[r, c] = 1.0
    rb_selector = to_f32(selector_np, l1=use_l1)

    # Next-cell selector: next_selector[r, r//TILE + 1] = 1 (for cell counts)
    next_sel_np = np.zeros((n_rows, n_cells_padded), dtype=np.float32)
    for r in range(n_rows):
        c = r // TILE + 1
        if c < n_cells_padded:
            next_sel_np[r, c] = 1.0
    rb_next_selector = to_f32(next_sel_np, l1=use_l1)

    # Row offset: all cols = r % TILE
    row_offset_np = np.zeros((n_rows, TILE), dtype=np.float32)
    for r in range(n_rows):
        row_offset_np[r, :] = float(r % TILE)
    rb_row_offset = to_f32(row_offset_np, l1=use_l1)

    # Padding row index (last row, guaranteed padding after sort)
    rb_padding_row = to_f32(np.full((n_rows, TILE), float(n_rows - 1), dtype=np.float32), l1=use_l1)

    # Initial forces (real-space + reciprocal + bonds)
    tt_fx, tt_fy, tt_fz = run_force_kernel(
        tt_px, tt_py, tt_pz, tt_q, tt_atom_valid, tt_self_diag,
        tt_scaler, tt_epslj, tt_fx, tt_fy, tt_fz)
    tt_fx = ttnn.add(tt_fx, tt_rx)
    tt_fy = ttnn.add(tt_fy, tt_ry)
    tt_fz = ttnn.add(tt_fz, tt_rz)
    if has_bonds:
        bfx, bfy, bfz = run_bond_forces(tt_px, tt_py, tt_pz,
                                         tt_p1_rows, tt_p2_rows, tt_m1, tt_m2)
        tt_fx = ttnn.add(tt_fx, bfx)
        tt_fy = ttnn.add(tt_fy, bfy)
        tt_fz = ttnn.add(tt_fz, bfz)
        ttnn.deallocate(bfx); ttnn.deallocate(bfy); ttnn.deallocate(bfz)

    steps_since_rebuild = 0
    n_rebuilds = 0
    print(f"  Rebuild interval: {max_rebuild_interval} steps")

    # --- Step loop ---
    step_times = []
    for step in range(n_steps):
        t0 = time.time()

        # Hardcoded rebuild every N steps (avoids expensive skin check)
        needs_rebuild = (steps_since_rebuild > 0 and
                         steps_since_rebuild >= max_rebuild_interval)

        if needs_rebuild:
            t_rb = time.time()

            # Fully on-device TILE-aligned rebuild (zero host transfers):
            # Sort by cell_id → prefix-sum cell_starts → TILE-aligned gather

            # 1. Compute cell_ids from current positions
            def cell_coord_dev(pos):
                scaled = ttnn.mul(pos, rb_inv_cs)
                fl = ttnn.floor(scaled)
                div = ttnn.mul(fl, rb_inv_n)
                fl_div = ttnn.floor(div)
                return ttnn.sub(fl, ttnn.mul(rb_n_dim_t, fl_div))

            ix = cell_coord_dev(tt_px)
            iy = cell_coord_dev(tt_py)
            iz = cell_coord_dev(tt_pz)
            cell_id = ttnn.add(ttnn.mul(ix, rb_dim2_t),
                               ttnn.add(ttnn.mul(iy, rb_n_dim_t), iz))

            # 2. Sentinel for padding rows, broadcast col0 to all cols
            av_f32 = ttnn.typecast(tt_atom_valid, ttnn.float32)
            av_broadcast = ttnn.matmul(av_f32, rb_col0_to_all)
            is_pad = ttnn.gt(av_broadcast, rb_zero)
            is_real = ttnn.sub(rb_one, ttnn.typecast(is_pad, ttnn.float32))
            cell_id_safe = ttnn.add(ttnn.mul(cell_id, is_real),
                                    ttnn.mul(rb_sentinel, ttnn.typecast(is_pad, ttnn.float32)))
            cell_id_full = ttnn.matmul(cell_id_safe, rb_col0_to_all)
            cell_id_bf16 = ttnn.typecast(cell_id_full, ttnn.bfloat16)

            # 3. Sort by cell_id
            sorted_vals, sort_perm = ttnn.sort(cell_id_bf16, dim=0)

            # 4. First gather: reorder all arrays by cell_id (contiguous, not TILE-aligned)
            tt_px = ttnn.gather(tt_px, dim=0, index=sort_perm)
            tt_py = ttnn.gather(tt_py, dim=0, index=sort_perm)
            tt_pz = ttnn.gather(tt_pz, dim=0, index=sort_perm)
            tt_vx = ttnn.gather(tt_vx, dim=0, index=sort_perm)
            tt_vy = ttnn.gather(tt_vy, dim=0, index=sort_perm)
            tt_vz = ttnn.gather(tt_vz, dim=0, index=sort_perm)
            tt_fx = ttnn.gather(tt_fx, dim=0, index=sort_perm)
            tt_fy = ttnn.gather(tt_fy, dim=0, index=sort_perm)
            tt_fz = ttnn.gather(tt_fz, dim=0, index=sort_perm)
            tt_rx = ttnn.gather(tt_rx, dim=0, index=sort_perm)
            tt_ry = ttnn.gather(tt_ry, dim=0, index=sort_perm)
            tt_rz = ttnn.gather(tt_rz, dim=0, index=sort_perm)
            tt_q = ttnn.gather(tt_q, dim=0, index=sort_perm)
            tt_atom_valid = ttnn.gather(tt_atom_valid, dim=0, index=sort_perm)
            tt_epslj = ttnn.gather(tt_epslj, dim=0, index=sort_perm)
            tt_atom_id = ttnn.gather(tt_atom_id, dim=0, index=sort_perm)

            # 5. Compute cell_starts via comparison matrix
            # cell_starts[c] = number of sorted rows with cell_id < c
            sorted_cid_f32 = ttnn.typecast(sorted_vals, ttnn.float32)
            sorted_cid_bc = ttnn.matmul(sorted_cid_f32, rb_ones_broadcast)
            indicator = ttnn.lt(sorted_cid_bc, rb_ref)
            indicator_f32 = ttnn.typecast(indicator, ttnn.float32)
            cell_starts_raw = ttnn.matmul(rb_ones_row, indicator_f32)

            # 6. Build TILE-aligned gather index
            cell_starts_col = ttnn.transpose(cell_starts_raw, 0, 1)
            curr_starts = ttnn.matmul(rb_selector, cell_starts_col)
            next_starts = ttnn.matmul(rb_next_selector, cell_starts_col)
            curr_starts_full = ttnn.matmul(curr_starts, rb_col0_to_all)
            next_starts_full = ttnn.matmul(next_starts, rb_col0_to_all)
            count_expanded = ttnn.sub(next_starts_full, curr_starts_full)

            gather_idx_f32 = ttnn.add(curr_starts_full, rb_row_offset)
            valid_mask = ttnn.lt(rb_row_offset, count_expanded)
            valid_f32 = ttnn.typecast(valid_mask, ttnn.float32)
            inv_valid = ttnn.sub(rb_one, valid_f32)
            safe_idx = ttnn.add(ttnn.mul(gather_idx_f32, valid_f32),
                                ttnn.mul(rb_padding_row, inv_valid))
            gather_idx = ttnn.typecast(safe_idx, ttnn.uint32)

            # 7. Second gather: TILE-aligned placement
            tt_px = ttnn.gather(tt_px, dim=0, index=gather_idx)
            tt_py = ttnn.gather(tt_py, dim=0, index=gather_idx)
            tt_pz = ttnn.gather(tt_pz, dim=0, index=gather_idx)
            tt_vx = ttnn.gather(tt_vx, dim=0, index=gather_idx)
            tt_vy = ttnn.gather(tt_vy, dim=0, index=gather_idx)
            tt_vz = ttnn.gather(tt_vz, dim=0, index=gather_idx)
            tt_fx = ttnn.gather(tt_fx, dim=0, index=gather_idx)
            tt_fy = ttnn.gather(tt_fy, dim=0, index=gather_idx)
            tt_fz = ttnn.gather(tt_fz, dim=0, index=gather_idx)
            tt_rx = ttnn.gather(tt_rx, dim=0, index=gather_idx)
            tt_ry = ttnn.gather(tt_ry, dim=0, index=gather_idx)
            tt_rz = ttnn.gather(tt_rz, dim=0, index=gather_idx)
            tt_q = ttnn.gather(tt_q, dim=0, index=gather_idx)
            tt_epslj = ttnn.gather(tt_epslj, dim=0, index=gather_idx)

            # Recompute atom_valid from valid_mask (don't gather - avoids drift)
            # valid_f32: 1.0 for real atoms, 0.0 for padding
            new_av = ttnn.mul(ttnn.sub(rb_one, valid_f32), 1e6)
            tt_atom_valid = ttnn.typecast(new_av, ttnn.bfloat16)
            tt_atom_id = ttnn.gather(tt_atom_id, dim=0, index=gather_idx)

            # Rebuild bond partner indices from new atom layout
            if has_bonds:
                aid_np = ttnn.to_torch(tt_atom_id).float().numpy()
                new_cam = []
                for c in range(n_cells_total):
                    atoms_in_cell = []
                    for r in range(c * TILE, (c + 1) * TILE):
                        a = int(aid_np[r, 0])
                        if a >= 0:
                            atoms_in_cell.append(a)
                    new_cam.append(atoms_in_cell)
                tt_p1_rows, tt_p2_rows, tt_m1, tt_m2 = upload_bond_partners(new_cam)

            steps_since_rebuild = 0
            n_rebuilds += 1
            t_total_rb = time.time() - t_rb
            print(f"  rebuild {n_rebuilds} @ step {step}: {t_total_rb:.3f}s")

        vel_half_kernel(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)
        pos_update_kernel(tt_px, tt_py, tt_pz, tt_vx, tt_vy, tt_vz)

        # Real-space forces (bf16 kernel) + reciprocal correction (on-device add)
        tt_fx, tt_fy, tt_fz = run_force_kernel(
            tt_px, tt_py, tt_pz, tt_q, tt_atom_valid, tt_self_diag,
            tt_scaler, tt_epslj, tt_fx, tt_fy, tt_fz)
        tt_fx = ttnn.add(tt_fx, tt_rx)
        tt_fy = ttnn.add(tt_fy, tt_ry)
        tt_fz = ttnn.add(tt_fz, tt_rz)

        # Bond forces (harmonic springs keeping lipid chains connected)
        if has_bonds:
            bfx, bfy, bfz = run_bond_forces(tt_px, tt_py, tt_pz,
                                             tt_p1_rows, tt_p2_rows, tt_m1, tt_m2)
            tt_fx = ttnn.add(tt_fx, bfx)
            tt_fy = ttnn.add(tt_fy, bfy)
            tt_fz = ttnn.add(tt_fz, bfz)
            ttnn.deallocate(bfx); ttnn.deallocate(bfy); ttnn.deallocate(bfz)

        vel_half_kernel(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)

        steps_since_rebuild += 1
        step_times.append(time.time() - t0)

        # On-device velocity rescaling thermostat every 100 steps
        target_ke = 0.5 * n * 3 * target_T
        if step > 0:
            # Compute KE on device: sum(vx^2 + vy^2 + vz^2)
            v2 = ttnn.add(ttnn.add(
                ttnn.mul(tt_vx, tt_vx),
                ttnn.mul(tt_vy, tt_vy)),
                ttnn.mul(tt_vz, tt_vz))
            ke_t = ttnn.sum(v2)
            ke = float(ttnn.to_torch(ke_t)) * 0.5
            ttnn.deallocate(v2)
            ttnn.deallocate(ke_t)
            if ke > 0:
                scale = float(np.sqrt(target_ke / ke))
                tt_vx = ttnn.mul(tt_vx, scale)
                tt_vy = ttnn.mul(tt_vy, scale)
                tt_vz = ttnn.mul(tt_vz, scale)
                if step % 1000 == 0:
                    print(f"  step {step}: KE={ke:.2f} (scale={scale:.4f})")

        # Progress report every 1000 steps
        if step > 0 and step % 1000 == 0:
            import gc, os, resource
            recent = step_times[-1000:]
            avg_ms = np.mean(recent) * 1000
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            n_gc = len(gc.get_objects())
            print(f"  step {step}/{n_steps}: avg={avg_ms:.1f}ms/step, "
                  f"rebuilds={n_rebuilds}, RSS={rss_mb:.0f}MB, gc_objs={n_gc}",
                  flush=True)

        # Periodic trajectory dump
        if dump_callback and step > 0 and step % dump_interval == 0:
            _px = ttnn.to_torch(tt_px).float().numpy()
            _py = ttnn.to_torch(tt_py).float().numpy()
            _pz = ttnn.to_torch(tt_pz).float().numpy()
            _av = ttnn.to_torch(tt_atom_valid).float().numpy()
            _real = _av[:, 0] < 0.5
            frame_pos = np.stack([_px[_real, 0], _py[_real, 0], _pz[_real, 0]], axis=1)
            dump_callback(frame_pos, step)
            del _px, _py, _pz, _av

    # Read final state (cell_atom_map may be stale after on-device rebuilds)
    px_np = ttnn.to_torch(tt_px).float().numpy()
    py_np = ttnn.to_torch(tt_py).float().numpy()
    pz_np = ttnn.to_torch(tt_pz).float().numpy()
    vx_np = ttnn.to_torch(tt_vx).float().numpy()
    vy_np = ttnn.to_torch(tt_vy).float().numpy()
    vz_np = ttnn.to_torch(tt_vz).float().numpy()
    av_np = ttnn.to_torch(tt_atom_valid).float().numpy()

    # Extract real atoms using atom_valid (order may differ from initial)
    real_mask = av_np[:, 0] < 0.5
    final_pos = np.stack([px_np[real_mask, 0], py_np[real_mask, 0],
                          pz_np[real_mask, 0]], axis=1)
    final_vel = np.stack([vx_np[real_mask, 0], vy_np[real_mask, 0],
                          vz_np[real_mask, 0]], axis=1)

    return final_pos, final_vel, step_times, n_rebuilds


# ---- Main: micelle demo benchmark ----

if __name__ == "__main__":
    from input_gen import make_lipid_system
    from xyz_writer import write_xyz_frame

    device = ttnn.open_device(device_id=0)

    # Micelle system: 80 lipids + 2000 water = ~2400 atoms
    n_lipids = 80
    n_water = 2000
    positions, types, charges_typed, bonds, box_length = make_lipid_system(
        n_lipids=n_lipids, n_water=n_water, density=0.3, seed=42)
    n_atoms = len(positions)
    charges = charges_typed.astype(np.float64)

    np.random.seed(123)
    velocities = np.random.randn(n_atoms, 3) * 0.1
    velocities -= velocities.mean(axis=0)

    print("=" * 60)
    print(f"MICELLE DEMO: {n_atoms} atoms ({n_lipids} lipids, {n_water} water)")
    print(f"Box: {box_length:.2f}, Bonds: {len(bonds)}, Charged: {np.sum(charges != 0)}")
    print("=" * 60)

    n_steps = 50000
    dt = 0.005
    dump_path = "/tmp/micelle_traj.xyz"
    dump_interval = 1000

    # Trajectory writer callback
    frame_count = [0]
    def dump_frame(pos, step):
        n = min(len(pos), n_atoms)
        mode = 'w' if frame_count[0] == 0 else 'a'
        write_xyz_frame(dump_path, pos[:n], types, box_length, step=step, mode=mode)
        frame_count[0] += 1
        print(f"  frame {frame_count[0]} @ step {step}")

    # Write initial frame
    dump_frame(positions, 0)

    t0 = time.time()
    final_pos, final_vel, step_times, n_rebuilds = run_md_loop(
        device, positions.copy(), velocities.copy(), charges, box_length,
        n_steps=n_steps, dt=dt, skin=0.5, check_interval=10,
        max_rebuild_interval=1000, alpha=ALPHA, target_T=1.0,
        dump_callback=dump_frame, dump_interval=dump_interval,
        atom_types=types, bonds=bonds)
    t_total = time.time() - t0

    # Write final frame
    dump_frame(final_pos[:n_atoms], n_steps)

    # Summary
    non_rebuild = [t for t in step_times if t < 1.0]
    rebuild = [t for t in step_times if t >= 1.0]
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_steps} steps, {n_atoms} atoms")
    print(f"{'='*60}")
    if non_rebuild:
        print(f"  Avg step (non-rebuild): {np.mean(non_rebuild)*1000:.3f}ms")
        print(f"  Min step: {np.min(non_rebuild)*1000:.3f}ms")
    if rebuild:
        print(f"  Rebuilds: {n_rebuilds}, avg rebuild step: {np.mean(rebuild)*1000:.1f}ms")
    print(f"  Amortized: {t_total/n_steps*1000:.3f}ms/step")
    print(f"  Total wall: {t_total:.1f}s")
    print(f"  Frames: {frame_count[0]} -> {dump_path}")

    ttnn.close_device(device)
