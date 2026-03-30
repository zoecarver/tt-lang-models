"""Standalone test for bond forces kernel - incremental build-up."""
import torch
import numpy as np
import ttnn
import ttl

TILE = 32
BOND_K = 100.0
BOND_R0 = 1.0


def to_bf16(arr, device):
    return ttnn.from_torch(
        torch.tensor(arr, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ---- Test 1: Single bond, no PBC ----
def test_single_bond_no_pbc(device):
    """Just F = -k*(r-r0)/r * dr, no PBC, one bond partner."""
    c_k = float(BOND_K)
    c_r0 = float(BOND_R0)
    c_eps = 1e-8

    @ttl.kernel(grid="auto")
    def kern(own_x, own_y, own_z, p_x, p_y, p_z, mask,
             fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_tiles = own_x.shape[0] // TILE
        tiles_per_core = -(-n_tiles // grid_cols)

        ox_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        oy_cb = ttl.make_dataflow_buffer_like(own_y, shape=(1, 1), buffer_factor=2)
        oz_cb = ttl.make_dataflow_buffer_like(own_z, shape=(1, 1), buffer_factor=2)
        px_cb = ttl.make_dataflow_buffer_like(p_x, shape=(1, 1), buffer_factor=2)
        py_cb = ttl.make_dataflow_buffer_like(p_y, shape=(1, 1), buffer_factor=2)
        pz_cb = ttl.make_dataflow_buffer_like(p_z, shape=(1, 1), buffer_factor=2)
        mk_cb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        rsq_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < n_tiles:
                    with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz:
                        with px_cb.wait() as px, py_cb.wait() as py, pz_cb.wait() as pz, mk_cb.wait() as m:
                            dx = ox - px
                            dy = oy - py
                            dz = oz - pz
                            with rsq_cb.reserve() as tmp:
                                tmp.store(dx * dx + dy * dy + dz * dz + ttl.math.fill(ox, c_eps))
                            with rsq_cb.wait() as r_sq:
                                r_inv = ttl.math.rsqrt(r_sq)
                                r = r_sq * r_inv
                                f = ttl.math.neg(ttl.math.fill(r_sq, c_k)) * (r - ttl.math.fill(r_sq, c_r0)) * r_inv * m
                                with fxo_cb.reserve() as o:
                                    o.store(f * dx)
                                with fyo_cb.reserve() as o:
                                    o.store(f * dy)
                                with fzo_cb.reserve() as o:
                                    o.store(f * dz)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < n_tiles:
                    with ox_cb.reserve() as blk:
                        tx = ttl.copy(own_x[tile_idx, 0], blk); tx.wait()
                    with oy_cb.reserve() as blk:
                        tx = ttl.copy(own_y[tile_idx, 0], blk); tx.wait()
                    with oz_cb.reserve() as blk:
                        tx = ttl.copy(own_z[tile_idx, 0], blk); tx.wait()
                    with px_cb.reserve() as blk:
                        tx = ttl.copy(p_x[tile_idx, 0], blk); tx.wait()
                    with py_cb.reserve() as blk:
                        tx = ttl.copy(p_y[tile_idx, 0], blk); tx.wait()
                    with pz_cb.reserve() as blk:
                        tx = ttl.copy(p_z[tile_idx, 0], blk); tx.wait()
                    with mk_cb.reserve() as blk:
                        tx = ttl.copy(mask[tile_idx, 0], blk); tx.wait()

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

    n_tiles = 2
    n_rows = n_tiles * TILE
    np.random.seed(42)

    ox = np.zeros((n_rows, TILE), dtype=np.float32)
    oy = np.zeros_like(ox); oz = np.zeros_like(ox)
    px = np.zeros_like(ox); py = np.zeros_like(ox); pz = np.zeros_like(ox)
    m = np.zeros_like(ox)

    # Place partners ~1.5 apart (near r0=1.0, moderate forces)
    ox[:, 0] = 5.0
    oy[:, 0] = 5.0
    oz[:, 0] = 5.0
    px[:, 0] = 5.0 + np.random.uniform(0.5, 2.0, n_rows)
    py[:, 0] = 5.0 + np.random.uniform(-0.5, 0.5, n_rows)
    pz[:, 0] = 5.0 + np.random.uniform(-0.5, 0.5, n_rows)
    m[:n_rows//2, 0] = 1.0  # half have bonds

    # Reference
    eps = 1e-8
    ddx = ox - px; ddy = oy - py; ddz = oz - pz
    r_sq = ddx*ddx + ddy*ddy + ddz*ddz + eps
    r_inv = 1.0 / np.sqrt(r_sq)
    r = r_sq * r_inv
    f = -BOND_K * (r - BOND_R0) * r_inv * m
    ref_fx = f * ddx; ref_fy = f * ddy; ref_fz = f * ddz

    zeros = np.zeros_like(ox)
    tt = lambda a: to_bf16(a, device)
    kern(tt(ox), tt(oy), tt(oz), tt(px), tt(py), tt(pz), tt(m),
         tt(zeros.copy()), tt(zeros.copy()), tt(zeros.copy()))

    # Read back - need to call kernel with output tensors we can read
    tt_fx = tt(zeros.copy()); tt_fy = tt(zeros.copy()); tt_fz = tt(zeros.copy())
    kern(tt(ox), tt(oy), tt(oz), tt(px), tt(py), tt(pz), tt(m),
         tt_fx, tt_fy, tt_fz)
    fx = ttnn.to_torch(tt_fx).float().numpy()
    fy = ttnn.to_torch(tt_fy).float().numpy()
    fz = ttnn.to_torch(tt_fz).float().numpy()

    max_ref = max(np.max(np.abs(ref_fx[:, 0])), np.max(np.abs(ref_fy[:, 0])), np.max(np.abs(ref_fz[:, 0])))
    err_x = np.max(np.abs(fx[:, 0] - ref_fx[:, 0]))
    err_y = np.max(np.abs(fy[:, 0] - ref_fy[:, 0]))
    err_z = np.max(np.abs(fz[:, 0] - ref_fz[:, 0]))
    max_err = max(err_x, err_y, err_z)
    print(f"  Test 1 (single bond, no PBC): max_ref={max_ref:.2f}, max_err={max_err:.4f}, rel={max_err/max_ref:.4f}")
    assert max_err / max_ref < 0.05, f"FAIL: relative error {max_err/max_ref:.4f}"
    print("  PASS")


# ---- Test 2: Single bond WITH PBC ----
def test_single_bond_pbc(device):
    """F = -k*(r-r0)/r * dr with PBC minimum image."""
    box_length = 20.0
    c_k = float(BOND_K)
    c_r0 = float(BOND_R0)
    c_box = float(box_length)
    c_inv_box = 1.0 / float(box_length)
    c_half = 0.5
    c_eps = 1e-8

    @ttl.kernel(grid="auto")
    def kern(own_x, own_y, own_z, p_x, p_y, p_z, mask,
             fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_tiles = own_x.shape[0] // TILE
        tiles_per_core = -(-n_tiles // grid_cols)

        ox_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        oy_cb = ttl.make_dataflow_buffer_like(own_y, shape=(1, 1), buffer_factor=2)
        oz_cb = ttl.make_dataflow_buffer_like(own_z, shape=(1, 1), buffer_factor=2)
        px_cb = ttl.make_dataflow_buffer_like(p_x, shape=(1, 1), buffer_factor=2)
        py_cb = ttl.make_dataflow_buffer_like(p_y, shape=(1, 1), buffer_factor=2)
        pz_cb = ttl.make_dataflow_buffer_like(p_z, shape=(1, 1), buffer_factor=2)
        mk_cb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
        rsq_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < n_tiles:
                    with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz:
                        with px_cb.wait() as px, py_cb.wait() as py, pz_cb.wait() as pz, mk_cb.wait() as m:
                            dx = ox - px
                            dx = dx - ttl.math.fill(ox, c_box) * ttl.math.floor(dx * ttl.math.fill(ox, c_inv_box) + ttl.math.fill(ox, c_half))
                            dy = oy - py
                            dy = dy - ttl.math.fill(oy, c_box) * ttl.math.floor(dy * ttl.math.fill(oy, c_inv_box) + ttl.math.fill(oy, c_half))
                            dz = oz - pz
                            dz = dz - ttl.math.fill(oz, c_box) * ttl.math.floor(dz * ttl.math.fill(oz, c_inv_box) + ttl.math.fill(oz, c_half))
                            with rsq_cb.reserve() as tmp:
                                tmp.store(dx * dx + dy * dy + dz * dz + ttl.math.fill(ox, c_eps))
                            with rsq_cb.wait() as r_sq:
                                r_inv = ttl.math.rsqrt(r_sq)
                                r = r_sq * r_inv
                                f = ttl.math.neg(ttl.math.fill(r_sq, c_k)) * (r - ttl.math.fill(r_sq, c_r0)) * r_inv * m
                                with fxo_cb.reserve() as o:
                                    o.store(f * dx)
                                with fyo_cb.reserve() as o:
                                    o.store(f * dy)
                                with fzo_cb.reserve() as o:
                                    o.store(f * dz)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < n_tiles:
                    with ox_cb.reserve() as blk:
                        tx = ttl.copy(own_x[tile_idx, 0], blk); tx.wait()
                    with oy_cb.reserve() as blk:
                        tx = ttl.copy(own_y[tile_idx, 0], blk); tx.wait()
                    with oz_cb.reserve() as blk:
                        tx = ttl.copy(own_z[tile_idx, 0], blk); tx.wait()
                    with px_cb.reserve() as blk:
                        tx = ttl.copy(p_x[tile_idx, 0], blk); tx.wait()
                    with py_cb.reserve() as blk:
                        tx = ttl.copy(p_y[tile_idx, 0], blk); tx.wait()
                    with pz_cb.reserve() as blk:
                        tx = ttl.copy(p_z[tile_idx, 0], blk); tx.wait()
                    with mk_cb.reserve() as blk:
                        tx = ttl.copy(mask[tile_idx, 0], blk); tx.wait()

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

    n_tiles = 2
    n_rows = n_tiles * TILE
    np.random.seed(42)

    ox = np.zeros((n_rows, TILE), dtype=np.float32)
    oy = np.zeros_like(ox); oz = np.zeros_like(ox)
    px = np.zeros_like(ox); py = np.zeros_like(ox); pz = np.zeros_like(ox)
    m = np.zeros_like(ox)

    ox[:, 0] = 5.0
    oy[:, 0] = 5.0
    oz[:, 0] = 5.0
    px[:, 0] = 5.0 + np.random.uniform(0.5, 2.0, n_rows)
    py[:, 0] = 5.0 + np.random.uniform(-0.5, 0.5, n_rows)
    pz[:, 0] = 5.0 + np.random.uniform(-0.5, 0.5, n_rows)
    m[:n_rows//2, 0] = 1.0

    # Reference with PBC
    def pbc(d):
        return d - box_length * np.floor(d / box_length + 0.5)
    eps = 1e-8
    ddx = pbc(ox - px); ddy = pbc(oy - py); ddz = pbc(oz - pz)
    r_sq = ddx*ddx + ddy*ddy + ddz*ddz + eps
    r_inv = 1.0 / np.sqrt(r_sq)
    r = r_sq * r_inv
    f = -BOND_K * (r - BOND_R0) * r_inv * m
    ref_fx = f * ddx; ref_fy = f * ddy; ref_fz = f * ddz

    zeros = np.zeros_like(ox)
    tt = lambda a: to_bf16(a, device)
    tt_fx = tt(zeros.copy()); tt_fy = tt(zeros.copy()); tt_fz = tt(zeros.copy())
    kern(tt(ox), tt(oy), tt(oz), tt(px), tt(py), tt(pz), tt(m),
         tt_fx, tt_fy, tt_fz)
    fx = ttnn.to_torch(tt_fx).float().numpy()
    fy = ttnn.to_torch(tt_fy).float().numpy()
    fz = ttnn.to_torch(tt_fz).float().numpy()

    max_ref = max(np.max(np.abs(ref_fx[:, 0])), np.max(np.abs(ref_fy[:, 0])), np.max(np.abs(ref_fz[:, 0])))
    err_x = np.max(np.abs(fx[:, 0] - ref_fx[:, 0]))
    err_y = np.max(np.abs(fy[:, 0] - ref_fy[:, 0]))
    err_z = np.max(np.abs(fz[:, 0] - ref_fz[:, 0]))
    max_err = max(err_x, err_y, err_z)
    print(f"  Test 2 (single bond, PBC): max_ref={max_ref:.2f}, max_err={max_err:.4f}, rel={max_err/max_ref:.4f}")
    assert max_err / max_ref < 0.05, f"FAIL: relative error {max_err/max_ref:.4f}"
    print("  PASS")


# ---- Test 3: Full two-bond kernel (the real thing) ----
def test_two_bond_full(device):
    """Both bond partners, PBC, controlled data."""
    box_length = 20.0
    c_k = float(BOND_K)
    c_r0 = float(BOND_R0)
    c_box = float(box_length)
    c_inv_box = 1.0 / float(box_length)
    c_half = 0.5
    c_eps = 1e-8

    @ttl.kernel(grid="auto")
    def kern(own_x, own_y, own_z,
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
                    # Bond 1
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

                    # Bond 2 + combine
                    with ox_cb.wait() as ox2, oy_cb.wait() as oy2, oz_cb.wait() as oz2:
                        with p2x_cb.wait() as p2x, p2y_cb.wait() as p2y, p2z_cb.wait() as p2z, m2_cb.wait() as mask2:
                            dx2 = ox2 - p2x
                            dx2 = dx2 - ttl.math.fill(ox2, c_box) * ttl.math.floor(dx2 * ttl.math.fill(ox2, c_inv_box) + ttl.math.fill(ox2, c_half))
                            dy2 = oy2 - p2y
                            dy2 = dy2 - ttl.math.fill(oy2, c_box) * ttl.math.floor(dy2 * ttl.math.fill(oy2, c_inv_box) + ttl.math.fill(oy2, c_half))
                            dz2 = oz2 - p2z
                            dz2 = dz2 - ttl.math.fill(ox2, c_box) * ttl.math.floor(dz2 * ttl.math.fill(ox2, c_inv_box) + ttl.math.fill(ox2, c_half))
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

    n_tiles = 2
    n_rows = n_tiles * TILE
    np.random.seed(42)

    ox = np.zeros((n_rows, TILE), dtype=np.float32)
    oy = np.zeros_like(ox); oz = np.zeros_like(ox)
    p1x = np.zeros_like(ox); p1y = np.zeros_like(ox); p1z = np.zeros_like(ox)
    p2x = np.zeros_like(ox); p2y = np.zeros_like(ox); p2z = np.zeros_like(ox)
    m1 = np.zeros_like(ox); m2 = np.zeros_like(ox)

    # Controlled: partners ~1.0-2.0 apart
    ox[:, 0] = 10.0
    oy[:, 0] = 10.0
    oz[:, 0] = 10.0
    p1x[:, 0] = 10.0 + np.random.uniform(0.5, 2.0, n_rows)
    p1y[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    p1z[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    p2x[:, 0] = 10.0 + np.random.uniform(-2.0, -0.5, n_rows)
    p2y[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    p2z[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    m1[:, 0] = 1.0  # all have bond 1
    m2[:n_rows//2, 0] = 1.0  # half have bond 2

    # Reference
    def pbc(d):
        return d - box_length * np.floor(d / box_length + 0.5)
    eps = 1e-8
    def bond_ref(oox, ooy, ooz, ppx, ppy, ppz, mm):
        ddx = pbc(oox - ppx); ddy = pbc(ooy - ppy); ddz = pbc(ooz - ppz)
        rsq = ddx*ddx + ddy*ddy + ddz*ddz + eps
        ri = 1.0/np.sqrt(rsq); r = rsq * ri
        ff = -BOND_K * (r - BOND_R0) * ri * mm
        return ff*ddx, ff*ddy, ff*ddz
    fx1, fy1, fz1 = bond_ref(ox, oy, oz, p1x, p1y, p1z, m1)
    fx2, fy2, fz2 = bond_ref(ox, oy, oz, p2x, p2y, p2z, m2)
    ref_fx = fx1+fx2; ref_fy = fy1+fy2; ref_fz = fz1+fz2

    zeros = np.zeros_like(ox)
    tt = lambda a: to_bf16(a, device)
    tt_fx = tt(zeros.copy()); tt_fy = tt(zeros.copy()); tt_fz = tt(zeros.copy())
    kern(tt(ox), tt(oy), tt(oz),
         tt(p1x), tt(p1y), tt(p1z), tt(m1),
         tt(p2x), tt(p2y), tt(p2z), tt(m2),
         tt_fx, tt_fy, tt_fz)
    fx = ttnn.to_torch(tt_fx).float().numpy()
    fy = ttnn.to_torch(tt_fy).float().numpy()
    fz = ttnn.to_torch(tt_fz).float().numpy()

    max_ref = max(np.max(np.abs(ref_fx[:, 0])), np.max(np.abs(ref_fy[:, 0])), np.max(np.abs(ref_fz[:, 0])))
    err_x = np.max(np.abs(fx[:, 0] - ref_fx[:, 0]))
    err_y = np.max(np.abs(fy[:, 0] - ref_fy[:, 0]))
    err_z = np.max(np.abs(fz[:, 0] - ref_fz[:, 0]))
    max_err = max(err_x, err_y, err_z)
    print(f"  Test 3 (two bonds, PBC): max_ref={max_ref:.2f}, max_err={max_err:.4f}, rel={max_err/max_ref:.4f}")
    assert max_err / max_ref < 0.10, f"FAIL: relative error {max_err/max_ref:.4f}"
    print("  PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    test_single_bond_no_pbc(device)
    test_single_bond_pbc(device)
    test_two_bond_full(device)
    ttnn.close_device(device)
