"""Per-type LJ force kernel test.

Uses Lorentz-Berthelot combining with per-atom epsilon, fixed sigma=1.0.
This keeps the kernel at exactly 28 DFBs (same structure as md_cell_list,
swapping charge arrays for epsilon arrays, removing Coulomb).

eps_ij = sqrt(eps_i * eps_j) via broadcast+transpose, same as charge products.
sigma = 1.0 for all types (standard CG simplification).

Force: F_ij = 24 * eps_ij / r^2 * (2/r^12 - 1/r^6) * dr_ij
"""
import numpy as np
import torch
import ttnn
import ttl

TILE = 32
N_NBR = 27


def build_test_cell_data(positions, eps_per_atom, box_length, r_cut):
    """Build cell data with per-atom epsilon."""
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

    valid_atoms = sort_idx[valid]
    valid_cells = sorted_cell_id[valid]
    valid_local = local_idx[valid]
    rows = valid_cells * TILE + valid_local

    def make_arr():
        return np.zeros((n_cells_total * TILE, TILE), dtype=np.float32)

    own_px, own_py, own_pz, own_eps = make_arr(), make_arr(), make_arr(), make_arr()
    own_px[rows, 0] = positions[valid_atoms, 0]
    own_py[rows, 0] = positions[valid_atoms, 1]
    own_pz[rows, 0] = positions[valid_atoms, 2]
    own_eps[rows, 0] = eps_per_atom[valid_atoms]

    # Self-exclusion masks
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

    cell_atom_map = [sort_idx[cell_starts[c]:cell_starts[c+1]].tolist()
                     for c in range(n_cells_total)]

    return (own_px, own_py, own_pz, own_eps,
            masks, cell_atom_map, n_cells_total, n_cells_dim)


def make_lj_force_kernel(c_n_dim, c_dim2, c_box, c_inv_box):
    """Per-type LJ force kernel. 28 DFBs, same structure as md_cell_list.

    Replaces charge arrays with epsilon arrays. No Coulomb.
    eps_ij = sqrt(eps_i * eps_j) computed via broadcast+transpose (same as
    charge product q_i * q_j in the original, but with sqrt).

    Force: F_ij = 24 * eps_ij / r^2 * (2/r^12 - 1/r^6) * dr_ij
    (sigma = 1.0 for all types)
    """
    c_half = 0.5
    c_lj_scale = 24.0

    @ttl.kernel(grid="auto")
    def lj_force_kernel(own_px, own_py, own_pz, own_eps,
                        self_mask, scaler,
                        fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_cells = own_px.shape[0] // TILE
        cells_per_core = -(-n_cells // grid_cols)

        ox_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        oy_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        oz_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        oe_cb = ttl.make_dataflow_buffer_like(own_eps, shape=(1, 1), buffer_factor=2)
        ex_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ey_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        ez_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        ee_cb = ttl.make_dataflow_buffer_like(own_eps, shape=(1, 1), buffer_factor=2)
        sm_cb = ttl.make_dataflow_buffer_like(self_mask, shape=(1, 1), buffer_factor=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        ba_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        tr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        bb_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        r2_tmp = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        r2_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ee_pair_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
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
                    with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz, oe_cb.wait() as o_eps:
                        with sc_cb.wait() as sc:
                            for nbr_i in range(N_NBR):
                                with ex_cb.wait() as ex, ey_cb.wait() as ey, ez_cb.wait() as ez, ee_cb.wait() as e_eps, sm_cb.wait() as sm:
                                    # PBC x-displacements
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

                                    # PBC y-displacements
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

                                    # PBC z-displacements + self-exclusion mask
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(oz, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(ez))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                                        dz_raw = bav - bbv
                                        dz_pbc = dz_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dz_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                        with r2_cb.reserve() as r2o:
                                            r2o.store(r2p + dz_pbc * dz_pbc + sm)
                                        with dz_cb.reserve() as dzo:
                                            dzo.store(dz_pbc)

                                    # Pairwise eps: sqrt(eps_i * eps_j)
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(o_eps, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(e_eps))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, ee_pair_cb.reserve() as epo:
                                        epo.store(ttl.math.sqrt(bav * bbv))

                                    # LJ force: 24*eps/r^2 * (2/r^12 - 1/r^6)
                                    with r2_cb.wait() as r2, ee_pair_cb.wait() as eps_pair:
                                        r2_inv = ttl.math.recip(r2)
                                        r4_inv = r2_inv * r2_inv
                                        r6_inv = r4_inv * r2_inv
                                        r12_inv = r6_inv * r6_inv
                                        with fm_cb.reserve() as fmo:
                                            fmo.store(ttl.math.fill(r2, c_lj_scale) * eps_pair * r2_inv * (r12_inv + r12_inv - r6_inv))

                                    # Project onto displacements, reduce, accumulate
                                    with fm_cb.wait() as fm:
                                        with dx_cb.wait() as dxv:
                                            with ft_cb.reserve() as ft:
                                                ft.store(fm * dxv)
                                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[0]))
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
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[0]))
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
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[0]))
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
                    with oe_cb.reserve() as blk:
                        tx = ttl.copy(own_eps[cell_id, 0], blk); tx.wait()
                    with sc_cb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()
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
                        with ee_cb.reserve() as blk:
                            tx = ttl.copy(own_eps[nbr_cell, 0], blk); tx.wait()
                        with sm_cb.reserve() as blk:
                            tx = ttl.copy(self_mask[cell_id * N_NBR + nbr, 0], blk); tx.wait()

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

    return lj_force_kernel


def reference_lj_lb(positions, eps_per_atom, box_length):
    """O(N^2) reference LJ with Lorentz-Berthelot combining, sigma=1."""
    n = len(positions)
    dr = positions[:, None, :] - positions[None, :, :]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(r2, 1e10)

    eps_ij = np.sqrt(eps_per_atom[:, None] * eps_per_atom[None, :])

    r2_inv = 1.0 / r2
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2
    f_mag = 24.0 * eps_ij * r2_inv * (2.0 * r12_inv - r6_inv)
    forces = np.sum(f_mag[:, :, None] * dr, axis=1)
    return forces


def test_lj_forces():
    device = ttnn.open_device(device_id=0)

    np.random.seed(42)
    n_atoms = 128
    density = 0.3
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

    # Per-atom epsilon based on type
    types = np.zeros(n_atoms, dtype=int)
    types[:int(0.2*n_atoms)] = 0   # HEAD
    types[int(0.2*n_atoms):int(0.6*n_atoms)] = 1   # TAIL
    types[int(0.6*n_atoms):] = 2   # WATER

    eps_table = {0: 0.5, 1: 2.0, 2: 1.0}
    eps_per_atom = np.array([eps_table[t] for t in types], dtype=np.float32)

    ref_forces = reference_lj_lb(positions, eps_per_atom.astype(np.float64), box_length)
    print(f"System: {n_atoms} atoms, box={box_length:.2f}")
    print(f"Ref force range: [{np.min(np.abs(ref_forces)):.4f}, {np.max(np.abs(ref_forces)):.4f}]")

    r_cut = min(box_length / 2.0 - 0.1, 3.0)
    (own_px, own_py, own_pz, own_eps,
     masks, cell_atom_map, n_cells_total, n_cells_dim) = \
        build_test_cell_data(positions, eps_per_atom, box_length, r_cut)

    print(f"Cells: {n_cells_total} ({n_cells_dim}^3), r_cut={r_cut:.2f}")

    def to_bf16(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_px = to_bf16(own_px)
    tt_py = to_bf16(own_py)
    tt_pz = to_bf16(own_pz)
    tt_eps = to_bf16(own_eps)
    tt_masks = to_bf16(masks)
    tt_scaler = to_bf16(np.ones((TILE, TILE), dtype=np.float32))

    fx_np = np.zeros_like(own_px)
    fy_np = np.zeros_like(own_px)
    fz_np = np.zeros_like(own_px)
    tt_fx = to_bf16(fx_np)
    tt_fy = to_bf16(fy_np)
    tt_fz = to_bf16(fz_np)

    c_n_dim = int(n_cells_dim)
    c_dim2 = c_n_dim * c_n_dim

    kernel = make_lj_force_kernel(c_n_dim, c_dim2, float(box_length), 1.0/float(box_length))
    print("Running LJ force kernel...")
    kernel(tt_px, tt_py, tt_pz, tt_eps, tt_masks, tt_scaler, tt_fx, tt_fy, tt_fz)

    fx_result = ttnn.to_torch(tt_fx).float().numpy()
    fy_result = ttnn.to_torch(tt_fy).float().numpy()
    fz_result = ttnn.to_torch(tt_fz).float().numpy()

    tt_forces = np.zeros((n_atoms, 3))
    for cell_id, atoms in enumerate(cell_atom_map):
        for local_idx, atom_id in enumerate(atoms[:TILE]):
            tt_forces[atom_id, 0] = fx_result[cell_id * TILE + local_idx, 0]
            tt_forces[atom_id, 1] = fy_result[cell_id * TILE + local_idx, 0]
            tt_forces[atom_id, 2] = fz_result[cell_id * TILE + local_idx, 0]

    max_diff = np.max(np.abs(tt_forces - ref_forces))
    ref_scale = np.max(np.abs(ref_forces)) + 1e-10
    rel_err = max_diff / ref_scale
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Relative error: {rel_err:.6f}")

    for i in range(min(5, n_atoms)):
        print(f"  Atom {i}: ref=[{ref_forces[i,0]:.3f}, {ref_forces[i,1]:.3f}, {ref_forces[i,2]:.3f}] "
              f"tt=[{tt_forces[i,0]:.3f}, {tt_forces[i,1]:.3f}, {tt_forces[i,2]:.3f}]")

    if rel_err < 0.15:
        print("PASS: LJ forces match reference (bf16 tolerance)")
    else:
        print(f"WARN: relative error {rel_err:.4f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_lj_forces()
