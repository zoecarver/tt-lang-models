"""Streaming pair force kernel.

Instead of cell-list 32x32 pairwise matrices (12% utilization at ~11 atoms/cell),
packs per-atom neighbor lists into 32x32 tiles where:
  Row k = atom k (within group of 32)
  Column j = j-th neighbor of atom k
  Near 100% tile utilization.

Geometry is 1 broadcast per axis (no transpose), vs 2 broadcasts + 1 transpose.
Pre-computes eps_ij and qq on host to eliminate eps/charge broadcasts entirely.
"""
import numpy as np
import torch
import ttnn
import ttl

TILE = 32

ALPHA = 1.0
ERFC_A1 = 0.254829592
ERFC_A2 = -0.284496736
ERFC_A3 = 1.421413741
ERFC_A4 = -1.453152027
ERFC_A5 = 1.061405429
ERFC_P = 0.3275911


def build_pair_data(positions, eps_per_atom, charges, box_length, r_cut):
    """Build per-atom neighbor tiles packed into 32x32 matrices."""
    n = len(positions)

    n_cells_dim = max(3, int(box_length / r_cut))
    cell_size = box_length / n_cells_dim
    n_cells_total = n_cells_dim ** 3

    cidx = np.floor(positions / cell_size).astype(int) % n_cells_dim
    cell_id = cidx[:, 0] * n_cells_dim**2 + cidx[:, 1] * n_cells_dim + cidx[:, 2]

    cell_members = [[] for _ in range(n_cells_total)]
    for i in range(n):
        cell_members[cell_id[i]].append(i)

    offsets = [(dx, dy, dz)
               for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]

    neighbor_lists = []
    for i in range(n):
        ci = cell_id[i]
        cx = ci // (n_cells_dim**2)
        cy = (ci // n_cells_dim) % n_cells_dim
        cz = ci % n_cells_dim
        neighbors = []
        for dx, dy, dz in offsets:
            ncx = (cx + dx) % n_cells_dim
            ncy = (cy + dy) % n_cells_dim
            ncz = (cz + dz) % n_cells_dim
            nc = ncx * n_cells_dim**2 + ncy * n_cells_dim + ncz
            for j in cell_members[nc]:
                if j != i:
                    neighbors.append(j)
        neighbor_lists.append(neighbors)

    max_nbr = max(len(nl) for nl in neighbor_lists)
    n_nbr_tiles = -(-max_nbr // TILE)
    max_nbr_padded = n_nbr_tiles * TILE
    print(f"  Max neighbors/atom: {max_nbr}, padded to {max_nbr_padded} ({n_nbr_tiles} tiles)")

    n_groups = -(-n // TILE)
    n_padded = n_groups * TILE

    # Own positions: [n_groups * TILE, TILE], data in column 0
    own_x = np.zeros((n_padded, TILE), dtype=np.float32)
    own_y = np.zeros_like(own_x)
    own_z = np.zeros_like(own_x)
    own_x[:n, 0] = positions[:n, 0]
    own_y[:n, 0] = positions[:n, 1]
    own_z[:n, 0] = positions[:n, 2]

    # Neighbor data: [n_groups * n_nbr_tiles * TILE, TILE]
    # Tile at index (g * n_nbr_tiles + t): row k, col j = (t*32+j)-th neighbor of atom g*32+k
    total_nbr_rows = n_groups * n_nbr_tiles * TILE
    nbr_x = np.zeros((total_nbr_rows, TILE), dtype=np.float32)
    nbr_y = np.zeros_like(nbr_x)
    nbr_z = np.zeros_like(nbr_x)
    nbr_eps = np.zeros_like(nbr_x)
    nbr_qq = np.zeros_like(nbr_x)
    nbr_mask = np.full_like(nbr_x, 1e6)

    for i in range(n):
        g = i // TILE
        k = i % TILE
        for slot, j in enumerate(neighbor_lists[i]):
            t = slot // TILE
            col = slot % TILE
            row = (g * n_nbr_tiles + t) * TILE + k
            nbr_x[row, col] = positions[j, 0]
            nbr_y[row, col] = positions[j, 1]
            nbr_z[row, col] = positions[j, 2]
            nbr_eps[row, col] = np.sqrt(eps_per_atom[i] * eps_per_atom[j])
            nbr_qq[row, col] = charges[i] * charges[j]
            nbr_mask[row, col] = 0.0

    return (own_x, own_y, own_z,
            nbr_x, nbr_y, nbr_z, nbr_eps, nbr_qq, nbr_mask,
            n_groups, n_nbr_tiles, neighbor_lists)


def make_pair_force_kernel(c_box, c_inv_box, c_n_nbr_tiles):
    """Streaming pair force kernel. Much simpler than cell-list version."""
    c_half = 0.5
    c_lj_scale = 24.0
    c_alpha_sq = float(ALPHA * ALPHA)
    c_p_alpha = float(ERFC_P * ALPHA)
    c_two_a_sp = float(2.0 * ALPHA / np.sqrt(np.pi))
    c_a1 = float(ERFC_A1)
    c_a2 = float(-ERFC_A2)
    c_a3 = float(ERFC_A3)
    c_a4 = float(-ERFC_A4)
    c_a5 = float(ERFC_A5)

    @ttl.kernel(grid="auto")
    def pair_force_kernel(own_x, own_y, own_z,
                          nbr_x, nbr_y, nbr_z, nbr_eps, nbr_qq, nbr_mask,
                          scaler,
                          fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_groups = own_x.shape[0] // TILE
        groups_per_core = -(-n_groups // grid_cols)

        # Own positions (buffer_factor=1, loaded once per group)
        ox_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=1)
        oy_cb = ttl.make_dataflow_buffer_like(own_y, shape=(1, 1), buffer_factor=1)
        oz_cb = ttl.make_dataflow_buffer_like(own_z, shape=(1, 1), buffer_factor=1)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)

        # Neighbor data (streamed per neighbor tile)
        nx_cb = ttl.make_dataflow_buffer_like(nbr_x, shape=(1, 1), buffer_factor=2)
        ny_cb = ttl.make_dataflow_buffer_like(nbr_y, shape=(1, 1), buffer_factor=2)
        nz_cb = ttl.make_dataflow_buffer_like(nbr_z, shape=(1, 1), buffer_factor=2)
        ne_cb = ttl.make_dataflow_buffer_like(nbr_eps, shape=(1, 1), buffer_factor=2)
        nq_cb = ttl.make_dataflow_buffer_like(nbr_qq, shape=(1, 1), buffer_factor=2)
        nm_cb = ttl.make_dataflow_buffer_like(nbr_mask, shape=(1, 1), buffer_factor=2)

        # Broadcast own positions (self-cycling across neighbor loop)
        bx_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        by_cb = ttl.make_dataflow_buffer_like(own_y, shape=(1, 1), buffer_factor=2)
        bz_cb = ttl.make_dataflow_buffer_like(own_z, shape=(1, 1), buffer_factor=2)

        # Scratch
        s1_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        s2_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        s3_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)

        # Displacement (kept for force projection)
        dx_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        dy_cb = ttl.make_dataflow_buffer_like(own_y, shape=(1, 1), buffer_factor=2)
        dz_cb = ttl.make_dataflow_buffer_like(own_z, shape=(1, 1), buffer_factor=2)

        # Force magnitude + scratch
        fm_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        ft_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)

        # Force accumulators (self-cycling)
        ax_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        ay_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        az_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        # Output
        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)
        # Total: 4 + 6 + 3 + 3 + 3 + 2 + 3 + 3 = 27

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for lg in range(groups_per_core):
                gid = core_x * groups_per_core + lg
                if gid < n_groups:
                    with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz:
                        with sc_cb.wait() as sc:
                            # Broadcast own positions once
                            with bx_cb.reserve() as o:
                                o.store(ttl.math.broadcast(ox, dims=[1]))
                            with by_cb.reserve() as o:
                                o.store(ttl.math.broadcast(oy, dims=[1]))
                            with bz_cb.reserve() as o:
                                o.store(ttl.math.broadcast(oz, dims=[1]))

                            for nbr_t in range(c_n_nbr_tiles):
                                with nx_cb.wait() as nx, ny_cb.wait() as ny, nz_cb.wait() as nz:
                                    with ne_cb.wait() as ne, nq_cb.wait() as nq, nm_cb.wait() as nm:

                                        # === PBC DISPLACEMENTS ===

                                        # dx_raw = own_x_bcast - nbr_x
                                        with bx_cb.wait() as bxv:
                                            with s1_cb.reserve() as o:
                                                o.store(bxv - nx)
                                            with bx_cb.reserve() as o:
                                                o.store(bxv)

                                        # floor(dx_raw * inv_box + 0.5)
                                        with s1_cb.wait() as dx_raw:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.floor(dx_raw * ttl.math.fill(dx_raw, c_inv_box) + ttl.math.fill(dx_raw, c_half)))
                                            with s1_cb.reserve() as o:
                                                o.store(dx_raw)

                                        # dx = dx_raw - box * floor_val
                                        with s1_cb.wait() as dx_raw, s2_cb.wait() as fl:
                                            with dx_cb.reserve() as o:
                                                o.store(dx_raw - ttl.math.fill(dx_raw, c_box) * fl)

                                        # dy_raw
                                        with by_cb.wait() as byv:
                                            with s1_cb.reserve() as o:
                                                o.store(byv - ny)
                                            with by_cb.reserve() as o:
                                                o.store(byv)

                                        with s1_cb.wait() as dy_raw:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.floor(dy_raw * ttl.math.fill(dy_raw, c_inv_box) + ttl.math.fill(dy_raw, c_half)))
                                            with s1_cb.reserve() as o:
                                                o.store(dy_raw)

                                        with s1_cb.wait() as dy_raw, s2_cb.wait() as fl:
                                            with dy_cb.reserve() as o:
                                                o.store(dy_raw - ttl.math.fill(dy_raw, c_box) * fl)

                                        # dz_raw
                                        with bz_cb.wait() as bzv:
                                            with s1_cb.reserve() as o:
                                                o.store(bzv - nz)
                                            with bz_cb.reserve() as o:
                                                o.store(bzv)

                                        with s1_cb.wait() as dz_raw:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.floor(dz_raw * ttl.math.fill(dz_raw, c_inv_box) + ttl.math.fill(dz_raw, c_half)))
                                            with s1_cb.reserve() as o:
                                                o.store(dz_raw)

                                        with s1_cb.wait() as dz_raw, s2_cb.wait() as fl:
                                            with dz_cb.reserve() as o:
                                                o.store(dz_raw - ttl.math.fill(dz_raw, c_box) * fl)

                                        # === r2 = dx^2 + dy^2 + dz^2 + mask ===

                                        with dx_cb.wait() as dxv:
                                            with s1_cb.reserve() as o:
                                                o.store(dxv * dxv)
                                            with dx_cb.reserve() as o:
                                                o.store(dxv)

                                        with dy_cb.wait() as dyv, s1_cb.wait() as r2p:
                                            with s1_cb.reserve() as o:
                                                o.store(r2p + dyv * dyv)
                                            with dy_cb.reserve() as o:
                                                o.store(dyv)

                                        with dz_cb.wait() as dzv, s1_cb.wait() as r2p:
                                            with s1_cb.reserve() as o:
                                                o.store(r2p + dzv * dzv + nm)
                                            with dz_cb.reserve() as o:
                                                o.store(dzv)

                                        # === LJ FORCE ===
                                        # s1_cb has r2

                                        # r2_inv
                                        with s1_cb.wait() as r2:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.recip(r2))
                                            with s1_cb.reserve() as o:
                                                o.store(r2)

                                        # r6_inv = r2_inv^3
                                        with s2_cb.wait() as r2_inv:
                                            with s3_cb.reserve() as o:
                                                o.store(r2_inv * r2_inv * r2_inv)
                                            with s2_cb.reserve() as o:
                                                o.store(r2_inv)

                                        # f_lj = 24 * eps * r2_inv * (2*r12 - r6)
                                        with s3_cb.wait() as r6, s2_cb.wait() as r2_inv:
                                            with fm_cb.reserve() as o:
                                                o.store(ttl.math.fill(r6, c_lj_scale) * ne * r2_inv * (r6 * r6 + r6 * r6 - r6))

                                        # === COULOMB FORCE ===
                                        # s1_cb has r2, fm_cb has f_lj

                                        # r_inv = rsqrt(r2)
                                        with s1_cb.wait() as r2:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.rsqrt(r2))
                                            with s1_cb.reserve() as o:
                                                o.store(r2)

                                        # r = r2 * r_inv (for t computation)
                                        with s1_cb.wait() as r2, s2_cb.wait() as r_inv:
                                            with s3_cb.reserve() as o:
                                                o.store(r2 * r_inv)
                                            with s2_cb.reserve() as o:
                                                o.store(r_inv)
                                            with s1_cb.reserve() as o:
                                                o.store(r2)

                                        # t = 1/(1 + p*alpha*r)
                                        with s3_cb.wait() as r:
                                            with s3_cb.reserve() as o:
                                                o.store(ttl.math.recip(ttl.math.fill(r, 1.0) + ttl.math.fill(r, c_p_alpha) * r))

                                        # exp_neg = exp(-alpha^2 * r2)
                                        with s1_cb.wait() as r2:
                                            with s1_cb.reserve() as o:
                                                o.store(ttl.math.exp(ttl.math.neg(ttl.math.fill(r2, c_alpha_sq) * r2)))

                                        # Horner: poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
                                        # s3_cb=t, s1_cb=exp_neg, s2_cb=r_inv
                                        # Use ft_cb for t (self-cycling), s3_cb for h

                                        # h = a4 + t*a5
                                        with s3_cb.wait() as t:
                                            with ft_cb.reserve() as o:
                                                o.store(t)
                                            with s3_cb.reserve() as o:
                                                o.store(ttl.math.fill(t, c_a4) + t * ttl.math.fill(t, c_a5))

                                        # h = a3 + t*h
                                        with ft_cb.wait() as t, s3_cb.wait() as h:
                                            with s3_cb.reserve() as o:
                                                o.store(ttl.math.fill(t, c_a3) + t * h)
                                            with ft_cb.reserve() as o:
                                                o.store(t)

                                        # h = a2 + t*h
                                        with ft_cb.wait() as t, s3_cb.wait() as h:
                                            with s3_cb.reserve() as o:
                                                o.store(ttl.math.fill(t, c_a2) + t * h)
                                            with ft_cb.reserve() as o:
                                                o.store(t)

                                        # h = a1 + t*h
                                        with ft_cb.wait() as t, s3_cb.wait() as h:
                                            with s3_cb.reserve() as o:
                                                o.store(ttl.math.fill(t, c_a1) + t * h)
                                            with ft_cb.reserve() as o:
                                                o.store(t)

                                        # poly = t * h
                                        with ft_cb.wait() as t, s3_cb.wait() as h:
                                            with s3_cb.reserve() as o:
                                                o.store(t * h)

                                        # erfc = poly * exp_neg
                                        with s3_cb.wait() as poly, s1_cb.wait() as exp_neg:
                                            with s3_cb.reserve() as o:
                                                o.store(poly * exp_neg)
                                            with s1_cb.reserve() as o:
                                                o.store(exp_neg)

                                        # coul_term1 = erfc * r_inv^3
                                        with s3_cb.wait() as erfc_v, s2_cb.wait() as r_inv:
                                            with ft_cb.reserve() as o:
                                                o.store(erfc_v * r_inv * r_inv * r_inv)
                                            with s2_cb.reserve() as o:
                                                o.store(r_inv)

                                        # coul_term2 = two_a_sp * exp_neg * r_inv^2
                                        with s1_cb.wait() as exp_neg, s2_cb.wait() as r_inv:
                                            with s1_cb.reserve() as o:
                                                o.store(ttl.math.fill(exp_neg, c_two_a_sp) * exp_neg * r_inv * r_inv)

                                        # coul_raw = term1 + term2
                                        with ft_cb.wait() as t1, s1_cb.wait() as t2:
                                            with ft_cb.reserve() as o:
                                                o.store(t1 + t2)

                                        # f_coul = qq * coul_raw
                                        with ft_cb.wait() as raw:
                                            with ft_cb.reserve() as o:
                                                o.store(nq * raw)

                                        # f_total = f_lj + f_coul
                                        with fm_cb.wait() as f_lj, ft_cb.wait() as f_coul:
                                            with fm_cb.reserve() as o:
                                                o.store(f_lj + f_coul)

                                        # === FORCE PROJECTION + ACCUMULATION ===
                                        # fm_cb has f_total (32x32)
                                        # dx_cb, dy_cb, dz_cb have displacements

                                        # fx_pair = f_total * dx
                                        with fm_cb.wait() as fm, dx_cb.wait() as dxv:
                                            with s1_cb.reserve() as o:
                                                o.store(fm * dxv)
                                            with fm_cb.reserve() as o:
                                                o.store(fm)

                                        # Reduce fx across columns and accumulate
                                        with s1_cb.wait() as fx_full:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.reduce_sum(fx_full, sc, dims=[1]))
                                        if nbr_t == 0:
                                            with s2_cb.wait() as fx_red, ax_cb.reserve() as ax:
                                                ax.store(fx_red)
                                        else:
                                            with s2_cb.wait() as fx_red, ax_cb.wait() as prev:
                                                with ax_cb.reserve() as ax:
                                                    ax.store(prev + fx_red)

                                        # fy_pair = f_total * dy
                                        with fm_cb.wait() as fm, dy_cb.wait() as dyv:
                                            with s1_cb.reserve() as o:
                                                o.store(fm * dyv)
                                            with fm_cb.reserve() as o:
                                                o.store(fm)

                                        with s1_cb.wait() as fy_full:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.reduce_sum(fy_full, sc, dims=[1]))
                                        if nbr_t == 0:
                                            with s2_cb.wait() as fy_red, ay_cb.reserve() as ay:
                                                ay.store(fy_red)
                                        else:
                                            with s2_cb.wait() as fy_red, ay_cb.wait() as prev:
                                                with ay_cb.reserve() as ay:
                                                    ay.store(prev + fy_red)

                                        # fz_pair = f_total * dz
                                        with fm_cb.wait() as fm, dz_cb.wait() as dzv:
                                            with s1_cb.reserve() as o:
                                                o.store(fm * dzv)

                                        with s1_cb.wait() as fz_full:
                                            with s2_cb.reserve() as o:
                                                o.store(ttl.math.reduce_sum(fz_full, sc, dims=[1]))
                                        if nbr_t == 0:
                                            with s2_cb.wait() as fz_red, az_cb.reserve() as az:
                                                az.store(fz_red)
                                        else:
                                            with s2_cb.wait() as fz_red, az_cb.wait() as prev:
                                                with az_cb.reserve() as az:
                                                    az.store(prev + fz_red)

                            # Drain stale broadcast self-cycles
                            with bx_cb.wait() as _:
                                pass
                            with by_cb.wait() as _:
                                pass
                            with bz_cb.wait() as _:
                                pass

                            # Write accumulated forces
                            with ax_cb.wait() as fx, fxo_cb.reserve() as o:
                                o.store(fx)
                            with ay_cb.wait() as fy, fyo_cb.reserve() as o:
                                o.store(fy)
                            with az_cb.wait() as fz, fzo_cb.reserve() as o:
                                o.store(fz)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for lg in range(groups_per_core):
                gid = core_x * groups_per_core + lg
                if gid < n_groups:
                    with ox_cb.reserve() as blk:
                        tx = ttl.copy(own_x[gid, 0], blk); tx.wait()
                    with oy_cb.reserve() as blk:
                        tx = ttl.copy(own_y[gid, 0], blk); tx.wait()
                    with oz_cb.reserve() as blk:
                        tx = ttl.copy(own_z[gid, 0], blk); tx.wait()
                    with sc_cb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()
                    for t in range(c_n_nbr_tiles):
                        tile_idx = gid * c_n_nbr_tiles + t
                        with nx_cb.reserve() as blk:
                            tx = ttl.copy(nbr_x[tile_idx, 0], blk); tx.wait()
                        with ny_cb.reserve() as blk:
                            tx = ttl.copy(nbr_y[tile_idx, 0], blk); tx.wait()
                        with nz_cb.reserve() as blk:
                            tx = ttl.copy(nbr_z[tile_idx, 0], blk); tx.wait()
                        with ne_cb.reserve() as blk:
                            tx = ttl.copy(nbr_eps[tile_idx, 0], blk); tx.wait()
                        with nq_cb.reserve() as blk:
                            tx = ttl.copy(nbr_qq[tile_idx, 0], blk); tx.wait()
                        with nm_cb.reserve() as blk:
                            tx = ttl.copy(nbr_mask[tile_idx, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for lg in range(groups_per_core):
                gid = core_x * groups_per_core + lg
                if gid < n_groups:
                    with fxo_cb.wait() as blk:
                        tx = ttl.copy(blk, fx_out[gid, 0]); tx.wait()
                    with fyo_cb.wait() as blk:
                        tx = ttl.copy(blk, fy_out[gid, 0]); tx.wait()
                    with fzo_cb.wait() as blk:
                        tx = ttl.copy(blk, fz_out[gid, 0]); tx.wait()

    return pair_force_kernel


def erfc_approx(x):
    t = 1.0 / (1.0 + ERFC_P * x)
    poly = t * (ERFC_A1 + t * (ERFC_A2 + t * (ERFC_A3 + t * (ERFC_A4 + t * ERFC_A5))))
    return poly * np.exp(-x * x)


def reference_forces(positions, eps_per_atom, charges, box_length, alpha=ALPHA):
    """O(N^2) reference LJ + erfc Coulomb."""
    dr = positions[:, None, :] - positions[None, :, :]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(r2, 1e10)
    r = np.sqrt(r2)
    r2_inv = 1.0 / r2

    eps_ij = np.sqrt(eps_per_atom[:, None] * eps_per_atom[None, :])
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2
    f_lj = 24.0 * eps_ij * r2_inv * (2.0 * r12_inv - r6_inv)

    qq = charges[:, None] * charges[None, :]
    erfc_val = erfc_approx(alpha * r)
    exp_neg = np.exp(-alpha**2 * r2)
    f_coul = qq * (erfc_val * r2_inv / r + 2.0 * alpha / np.sqrt(np.pi) * exp_neg / r2)

    f_total = f_lj + f_coul
    return np.sum(f_total[:, :, None] * dr, axis=1)


def test_pair_forces():
    try:
        default_size = ttnn.device.get_max_worker_l1_unreserved_size()
        device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 80000)
    except AttributeError:
        device = ttnn.open_device(device_id=0)

    np.random.seed(42)
    n_atoms = 10000
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

    types = np.zeros(n_atoms, dtype=int)
    types[:int(0.2*n_atoms)] = 0
    types[int(0.2*n_atoms):int(0.6*n_atoms)] = 1
    types[int(0.6*n_atoms):] = 2
    eps_table = {0: 0.5, 1: 2.0, 2: 1.0}
    eps_per_atom = np.array([eps_table[t] for t in types], dtype=np.float32)

    charges = np.zeros(n_atoms, dtype=np.float32)
    for i in range(n_atoms):
        if types[i] == 0:
            charges[i] = 0.5 if i % 2 == 0 else -0.5

    r_cut = min(box_length / 2.0 - 0.1, 3.0)
    print(f"System: {n_atoms} atoms, box={box_length:.2f}, r_cut={r_cut:.2f}")
    print(f"Charged atoms: {np.sum(charges != 0)}")

    if n_atoms < 500:
        ref_forces = reference_forces(positions, eps_per_atom.astype(np.float64),
                                      charges.astype(np.float64), box_length)
    else:
        ref_forces = None
        print("Skipping O(N^2) reference (too slow)")

    (own_x, own_y, own_z,
     nbr_x, nbr_y, nbr_z, nbr_eps, nbr_qq, nbr_mask,
     n_groups, n_nbr_tiles, _) = \
        build_pair_data(positions, eps_per_atom, charges, box_length, r_cut)
    print(f"Groups: {n_groups}, Neighbor tiles/group: {n_nbr_tiles}")

    def to_f32(arr, mem=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=mem)

    # Small tensors in L1, large neighbor arrays in DRAM
    L1 = ttnn.L1_MEMORY_CONFIG
    tt_own_x = to_f32(own_x, L1)
    tt_own_y = to_f32(own_y, L1)
    tt_own_z = to_f32(own_z, L1)
    tt_nbr_x = to_f32(nbr_x, L1)
    tt_nbr_y = to_f32(nbr_y, L1)
    tt_nbr_z = to_f32(nbr_z, L1)
    tt_nbr_eps = to_f32(nbr_eps, L1)
    tt_nbr_qq = to_f32(nbr_qq, L1)
    tt_nbr_mask = to_f32(nbr_mask, L1)
    tt_scaler = to_f32(np.ones((TILE, TILE), dtype=np.float32), L1)

    n_padded = n_groups * TILE
    tt_fx = to_f32(np.zeros((n_padded, TILE), dtype=np.float32), L1)
    tt_fy = to_f32(np.zeros((n_padded, TILE), dtype=np.float32), L1)
    tt_fz = to_f32(np.zeros((n_padded, TILE), dtype=np.float32), L1)

    kernel = make_pair_force_kernel(float(box_length), 1.0/float(box_length), n_nbr_tiles)
    print("Tensors allocated. Running pair force kernel...", flush=True)
    import time as _time
    _t0 = _time.time()
    kernel(tt_own_x, tt_own_y, tt_own_z,
           tt_nbr_x, tt_nbr_y, tt_nbr_z, tt_nbr_eps, tt_nbr_qq, tt_nbr_mask,
           tt_scaler,
           tt_fx, tt_fy, tt_fz)
    if hasattr(ttnn, 'synchronize_device'):
        ttnn.synchronize_device(device)
    print(f"Kernel done in {_time.time()-_t0:.2f}s", flush=True)

    fx_r = ttnn.to_torch(tt_fx).float().numpy()
    fy_r = ttnn.to_torch(tt_fy).float().numpy()
    fz_r = ttnn.to_torch(tt_fz).float().numpy()

    # Extract forces (atoms are in row order, column 0)
    tt_forces = np.zeros((n_atoms, 3))
    tt_forces[:, 0] = fx_r[:n_atoms, 0]
    tt_forces[:, 1] = fy_r[:n_atoms, 0]
    tt_forces[:, 2] = fz_r[:n_atoms, 0]

    if ref_forces is not None:
        max_diff = np.max(np.abs(tt_forces - ref_forces))
        ref_scale = np.max(np.abs(ref_forces)) + 1e-10
        rel_err = max_diff / ref_scale
        print(f"Max abs diff: {max_diff:.6f}, rel err: {rel_err:.6f}")
        for i in range(min(5, n_atoms)):
            print(f"  Atom {i} (type={types[i]}, q={charges[i]:.1f}): "
                  f"ref=[{ref_forces[i,0]:.3f}, {ref_forces[i,1]:.3f}, {ref_forces[i,2]:.3f}] "
                  f"tt=[{tt_forces[i,0]:.3f}, {tt_forces[i,1]:.3f}, {tt_forces[i,2]:.3f}]")
        if rel_err < 0.1:
            print("PASS")
        else:
            print(f"FAIL: rel_err={rel_err:.4f}")
    else:
        print(f"Force range: [{np.min(tt_forces):.3f}, {np.max(tt_forces):.3f}]")
        for i in range(min(5, n_atoms)):
            print(f"  Atom {i}: tt=[{tt_forces[i,0]:.3f}, {tt_forces[i,1]:.3f}, {tt_forces[i,2]:.3f}]")

    # Timing
    import time
    if hasattr(ttnn, 'synchronize_device'):
        ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(10):
        kernel(tt_own_x, tt_own_y, tt_own_z,
               tt_nbr_x, tt_nbr_y, tt_nbr_z, tt_nbr_eps, tt_nbr_qq, tt_nbr_mask,
               tt_scaler,
               tt_fx, tt_fy, tt_fz)
    if hasattr(ttnn, 'synchronize_device'):
        ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"\nTiming: {(t1-t0)/10*1000:.3f}ms/call")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_pair_forces()
