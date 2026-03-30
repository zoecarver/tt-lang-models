"""Fused LJ + erfc Coulomb force kernel.

Combines per-type LJ (eps combining) with erfc-damped Coulomb in one
neighbor pass. 28 DFBs (at the HW limit).

Added vs test_lj_typed.py:
- oq_cb, eq_cb for own/neighbor charges
- Coulomb math: erfc polynomial + damped force, added to LJ force magnitude
- Removed one intermediate DFB to stay at 28

Erfc approximation (Abramowitz & Stegun):
  t = 1 / (1 + P*alpha*r)
  erfc(alpha*r) ~ t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))) * exp(-alpha^2*r^2)

Coulomb force:
  F_coul = q_i*q_j * (erfc/r^2 + 2*alpha/sqrt(pi)*exp(-a^2*r^2)/r) / r
"""
import numpy as np
import torch
import ttnn
import ttl

TILE = 32
N_NBR = 27

# Erfc constants
ALPHA = 1.0
ERFC_A1 = 0.254829592
ERFC_A2 = -0.284496736
ERFC_A3 = 1.421413741
ERFC_A4 = -1.453152027
ERFC_A5 = 1.061405429
ERFC_P = 0.3275911


def build_cell_data(positions, eps_per_atom, charges, box_length, r_cut):
    """Build cell data with per-atom epsilon and charges."""
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

    own_px, own_py, own_pz = make_arr(), make_arr(), make_arr()
    own_eps, own_q = make_arr(), make_arr()
    own_px[rows, 0] = positions[valid_atoms, 0]
    own_py[rows, 0] = positions[valid_atoms, 1]
    own_pz[rows, 0] = positions[valid_atoms, 2]
    own_eps[rows, 0] = eps_per_atom[valid_atoms]
    own_q[rows, 0] = charges[valid_atoms]

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

    return (own_px, own_py, own_pz, own_eps, own_q,
            masks, cell_atom_map, n_cells_total, n_cells_dim)


def make_lj_coulomb_kernel(c_n_dim, c_dim2, c_box, c_inv_box, c_force_clamp=50.0):
    """Fused LJ + erfc Coulomb. 28 DFBs."""
    c_half = 0.5
    c_lj_scale = 24.0
    c_fclamp = float(c_force_clamp)
    c_neg_fclamp = float(-c_force_clamp)
    c_alpha_sq = float(ALPHA * ALPHA)
    c_p_alpha = float(ERFC_P * ALPHA)
    c_two_a_sp = float(2.0 * ALPHA / np.sqrt(np.pi))
    c_a1 = float(ERFC_A1)
    c_a2 = float(-ERFC_A2)
    c_a3 = float(ERFC_A3)
    c_a4 = float(-ERFC_A4)
    c_a5 = float(ERFC_A5)

    @ttl.kernel(grid="auto")
    def lj_coulomb_kernel(own_px, own_py, own_pz, own_eps, own_q,
                          self_mask, scaler,
                          fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_cells = own_px.shape[0] // TILE
        cells_per_core = -(-n_cells // grid_cols)

        # Input: own(5) + nbr(5) + mask + scaler = 12
        ox_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        oy_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        oz_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        oe_cb = ttl.make_dataflow_buffer_like(own_eps, shape=(1, 1), buffer_factor=2)
        oq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
        ex_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ey_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        ez_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        ee_cb = ttl.make_dataflow_buffer_like(own_eps, shape=(1, 1), buffer_factor=2)
        eq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
        sm_cb = ttl.make_dataflow_buffer_like(self_mask, shape=(1, 1), buffer_factor=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        # Geometry: 4
        ba_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        tr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        bb_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        r2_tmp = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        # Pairwise: 4
        r2_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dx_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dy_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dz_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        # Force: 3
        fm_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ft_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        fr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        # Accum + output: 6
        ax_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        ay_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        az_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)
        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)
        # Total: 12 + 4 + 4 + 3 + 6 = 29 ... one over!

        # Need to cut 1 DFB. Merge r2_tmp into r2_cb via self-cycling.
        # Actually: recount. We have 29 above. Drop r2_tmp and accumulate
        # r2 in-place using r2_cb with self-cycling.

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_c in range(cells_per_core):
                cell_id = core_x * cells_per_core + local_c
                if cell_id < n_cells:
                    with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz:
                        with oe_cb.wait() as o_eps, oq_cb.wait() as o_q:
                            with sc_cb.wait() as sc:
                                for nbr_i in range(N_NBR):
                                    with ex_cb.wait() as ex, ey_cb.wait() as ey, ez_cb.wait() as ez:
                                        with ee_cb.wait() as e_eps, eq_cb.wait() as e_q, sm_cb.wait() as sm:
                                            # PBC x-displacement
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

                                            # PBC y-displacement
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

                                            # PBC z-displacement + mask
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
                                            with ba_cb.wait() as ei, bb_cb.wait() as ej:
                                                with ft_cb.reserve() as eps_ij:
                                                    eps_ij.store(ttl.math.sqrt(ei * ej))

                                            # Pairwise charge product: q_i * q_j
                                            with ba_cb.reserve() as ba:
                                                ba.store(ttl.math.broadcast(o_q, dims=[1]))
                                            with tr_cb.reserve() as tr:
                                                tr.store(ttl.transpose(e_q))
                                            with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                                bb.store(ttl.math.broadcast(trv, dims=[0]))
                                            with ba_cb.wait() as qi, bb_cb.wait() as qj:
                                                with fr_cb.reserve() as qq:
                                                    qq.store(qi * qj)

                                            # === LJ FORCE ===
                                            # State: r2_cb=r2, ft_cb=eps_ij, fr_cb=qq

                                            # r2_inv = 1/r2
                                            with r2_cb.wait() as r2:
                                                with ba_cb.reserve() as o:
                                                    o.store(ttl.math.recip(r2))
                                                with r2_cb.reserve() as o:
                                                    o.store(r2)

                                            # r6_inv = r2_inv^3
                                            with ba_cb.wait() as r2_inv:
                                                with tr_cb.reserve() as o:
                                                    o.store(r2_inv * r2_inv * r2_inv)
                                                with ba_cb.reserve() as o:
                                                    o.store(r2_inv)

                                            # f_lj = 24*eps*r2_inv*(2*r12 - r6)
                                            with tr_cb.wait() as r6, ba_cb.wait() as r2_inv, ft_cb.wait() as eps:
                                                with fm_cb.reserve() as o:
                                                    o.store(ttl.math.fill(r6, c_lj_scale) * eps * r2_inv * (r6 * r6 + r6 * r6 - r6))

                                            # === COULOMB FORCE ===
                                            # State: r2_cb=r2, fr_cb=qq, fm_cb=f_lj

                                            # r_inv = rsqrt(r2)
                                            with r2_cb.wait() as r2:
                                                with ba_cb.reserve() as o:
                                                    o.store(ttl.math.rsqrt(r2))
                                                with r2_cb.reserve() as o:
                                                    o.store(r2)

                                            # r = r2 * r_inv
                                            with r2_cb.wait() as r2, ba_cb.wait() as r_inv:
                                                with tr_cb.reserve() as o:
                                                    o.store(r2 * r_inv)
                                                with ba_cb.reserve() as o:
                                                    o.store(r_inv)
                                                with r2_cb.reserve() as o:
                                                    o.store(r2)

                                            # t = 1/(1 + p*alpha*r)
                                            with tr_cb.wait() as r:
                                                with tr_cb.reserve() as o:
                                                    o.store(ttl.math.recip(ttl.math.fill(r, 1.0) + ttl.math.fill(r, c_p_alpha) * r))

                                            # exp_neg = exp(-alpha^2 * r2)
                                            with r2_cb.wait() as r2:
                                                with r2_cb.reserve() as o:
                                                    o.store(ttl.math.exp(ttl.math.neg(ttl.math.fill(r2, c_alpha_sq) * r2)))

                                            # State: tr_cb=t, r2_cb=exp_neg, ba_cb=r_inv, fr_cb=qq, fm_cb=f_lj
                                            # Horner: poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
                                            # Evaluate inside-out, one step per with block
                                            # Use bb_cb for h (running accum), r2_tmp to hold t between steps

                                            # Save t for reuse across Horner steps
                                            with tr_cb.wait() as t:
                                                with r2_tmp.reserve() as o:
                                                    o.store(t)
                                                with tr_cb.reserve() as o:
                                                    o.store(t)

                                            # h = a4 + t*a5
                                            with tr_cb.wait() as t:
                                                with bb_cb.reserve() as o:
                                                    o.store(ttl.math.fill(t, c_a4) + t * ttl.math.fill(t, c_a5))

                                            # h = a3 + t*h
                                            with r2_tmp.wait() as t, bb_cb.wait() as h:
                                                with r2_tmp.reserve() as o:
                                                    o.store(t)
                                                with bb_cb.reserve() as o:
                                                    o.store(ttl.math.fill(t, c_a3) + t * h)

                                            # h = a2 + t*h
                                            with r2_tmp.wait() as t, bb_cb.wait() as h:
                                                with r2_tmp.reserve() as o:
                                                    o.store(t)
                                                with bb_cb.reserve() as o:
                                                    o.store(ttl.math.fill(t, c_a2) + t * h)

                                            # h = a1 + t*h
                                            with r2_tmp.wait() as t, bb_cb.wait() as h:
                                                with r2_tmp.reserve() as o:
                                                    o.store(t)
                                                with bb_cb.reserve() as o:
                                                    o.store(ttl.math.fill(t, c_a1) + t * h)

                                            # poly = t * h
                                            with r2_tmp.wait() as t, bb_cb.wait() as h:
                                                with bb_cb.reserve() as o:
                                                    o.store(t * h)

                                            # erfc_val = poly * exp_neg
                                            with bb_cb.wait() as poly, r2_cb.wait() as exp_neg:
                                                with bb_cb.reserve() as o:
                                                    o.store(poly * exp_neg)
                                                with r2_cb.reserve() as o:
                                                    o.store(exp_neg)

                                            # coul_term1 = erfc * r_inv^3 (= erfc / r^2 / r)
                                            with bb_cb.wait() as erfc_v, ba_cb.wait() as r_inv:
                                                with tr_cb.reserve() as o:
                                                    o.store(erfc_v * r_inv * r_inv * r_inv)
                                                with ba_cb.reserve() as o:
                                                    o.store(r_inv)

                                            # coul_term2 = (2*alpha/sqrt(pi)) * exp_neg * r_inv^2
                                            with r2_cb.wait() as exp_neg, ba_cb.wait() as r_inv:
                                                with r2_cb.reserve() as o:
                                                    o.store(ttl.math.fill(exp_neg, c_two_a_sp) * exp_neg * r_inv * r_inv)

                                            # f_coul_raw = coul_term1 + coul_term2
                                            with tr_cb.wait() as t1, r2_cb.wait() as t2:
                                                with tr_cb.reserve() as o:
                                                    o.store(t1 + t2)

                                            # f_coul = qq * f_coul_raw
                                            with tr_cb.wait() as raw, fr_cb.wait() as qq:
                                                with ft_cb.reserve() as o:
                                                    o.store(qq * raw)

                                            # f_total = clamp(f_lj + f_coul)
                                            with fm_cb.wait() as f_lj, ft_cb.wait() as f_coul:
                                                with fm_cb.reserve() as o:
                                                    raw = f_lj + f_coul
                                                    o.store(ttl.math.max(ttl.math.fill(raw, c_neg_fclamp),
                                                            ttl.math.min(raw, ttl.math.fill(raw, c_fclamp))))

                                            # === FORCE PROJECTION ===
                                            # State: fm_cb=f_total, dx_cb=dx, dy_cb=dy, dz_cb=dz
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
                    with oe_cb.reserve() as blk:
                        tx = ttl.copy(own_eps[cell_id, 0], blk); tx.wait()
                    with oq_cb.reserve() as blk:
                        tx = ttl.copy(own_q[cell_id, 0], blk); tx.wait()
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
                        with eq_cb.reserve() as blk:
                            tx = ttl.copy(own_q[nbr_cell, 0], blk); tx.wait()
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

    return lj_coulomb_kernel


def erfc_approx(x):
    """Abramowitz & Stegun erfc approximation (same as kernel)."""
    t = 1.0 / (1.0 + ERFC_P * x)
    poly = t * (ERFC_A1 + t * (ERFC_A2 + t * (ERFC_A3 + t * (ERFC_A4 + t * ERFC_A5))))
    return poly * np.exp(-x * x)


def reference_lj_coulomb(positions, eps_per_atom, charges, box_length, alpha=ALPHA):
    """O(N^2) reference LJ + erfc Coulomb."""
    dr = positions[:, None, :] - positions[None, :, :]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(r2, 1e10)
    r = np.sqrt(r2)
    r2_inv = 1.0 / r2

    # LJ
    eps_ij = np.sqrt(eps_per_atom[:, None] * eps_per_atom[None, :])
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2
    f_lj = 24.0 * eps_ij * r2_inv * (2.0 * r12_inv - r6_inv)

    # erfc Coulomb
    qq = charges[:, None] * charges[None, :]
    erfc_val = erfc_approx(alpha * r)
    exp_neg = np.exp(-alpha**2 * r2)
    f_coul = qq * (erfc_val * r2_inv / r + 2.0 * alpha / np.sqrt(np.pi) * exp_neg / r2)

    f_total = f_lj + f_coul
    return np.sum(f_total[:, :, None] * dr, axis=1)


def test_lj_coulomb():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 80000)

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

    # Types with different eps
    types = np.zeros(n_atoms, dtype=int)
    types[:int(0.2*n_atoms)] = 0
    types[int(0.2*n_atoms):int(0.6*n_atoms)] = 1
    types[int(0.6*n_atoms):] = 2
    eps_table = {0: 0.5, 1: 2.0, 2: 1.0}
    eps_per_atom = np.array([eps_table[t] for t in types], dtype=np.float32)

    # Charges: heads charged, rest neutral
    charges = np.zeros(n_atoms, dtype=np.float32)
    for i in range(n_atoms):
        if types[i] == 0:
            charges[i] = 0.5 if i % 2 == 0 else -0.5

    ref_forces = reference_lj_coulomb(positions, eps_per_atom.astype(np.float64),
                                       charges.astype(np.float64), box_length)
    print(f"System: {n_atoms} atoms, box={box_length:.2f}")
    print(f"Charged atoms: {np.sum(charges != 0)}")

    r_cut = min(box_length / 2.0 - 0.1, 3.0)
    (own_px, own_py, own_pz, own_eps, own_q,
     masks, cell_atom_map, n_cells_total, n_cells_dim) = \
        build_cell_data(positions, eps_per_atom, charges, box_length, r_cut)
    print(f"Cells: {n_cells_total} ({n_cells_dim}^3)")

    def to_f32(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    tt_px = to_f32(own_px)
    tt_py = to_f32(own_py)
    tt_pz = to_f32(own_pz)
    tt_eps = to_f32(own_eps)
    tt_q = to_f32(own_q)
    tt_masks = to_f32(masks)
    tt_scaler = to_f32(np.ones((TILE, TILE), dtype=np.float32))
    tt_fx = to_f32(np.zeros_like(own_px))
    tt_fy = to_f32(np.zeros_like(own_px))
    tt_fz = to_f32(np.zeros_like(own_px))

    c_n_dim = int(n_cells_dim)
    c_dim2 = c_n_dim * c_n_dim

    kernel = make_lj_coulomb_kernel(c_n_dim, c_dim2, float(box_length), 1.0/float(box_length))
    print("Running fused LJ+Coulomb kernel...")
    kernel(tt_px, tt_py, tt_pz, tt_eps, tt_q, tt_masks, tt_scaler, tt_fx, tt_fy, tt_fz)

    fx_r = ttnn.to_torch(tt_fx).float().numpy()
    fy_r = ttnn.to_torch(tt_fy).float().numpy()
    fz_r = ttnn.to_torch(tt_fz).float().numpy()

    tt_forces = np.zeros((n_atoms, 3))
    for cell_id, atoms in enumerate(cell_atom_map):
        for local_idx, atom_id in enumerate(atoms[:TILE]):
            tt_forces[atom_id, 0] = fx_r[cell_id * TILE + local_idx, 0]
            tt_forces[atom_id, 1] = fy_r[cell_id * TILE + local_idx, 0]
            tt_forces[atom_id, 2] = fz_r[cell_id * TILE + local_idx, 0]

    max_diff = np.max(np.abs(tt_forces - ref_forces))
    ref_scale = np.max(np.abs(ref_forces)) + 1e-10
    rel_err = max_diff / ref_scale
    print(f"Max abs diff: {max_diff:.6f}, rel err: {rel_err:.6f}")

    for i in range(min(5, n_atoms)):
        print(f"  Atom {i} (type={types[i]}, q={charges[i]:.1f}): "
              f"ref=[{ref_forces[i,0]:.3f}, {ref_forces[i,1]:.3f}, {ref_forces[i,2]:.3f}] "
              f"tt=[{tt_forces[i,0]:.3f}, {tt_forces[i,1]:.3f}, {tt_forces[i,2]:.3f}]")

    if rel_err < 0.1:
        print("PASS (f32 tolerance for CG sim with Coulomb)")
    else:
        print(f"FAIL: rel_err={rel_err:.4f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_lj_coulomb()
