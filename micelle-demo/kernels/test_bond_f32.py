"""f32 bond forces kernel + test.

Exports make_bond_kernel(c_box, c_inv_box) for use in test_md_step.py.
Two-bond harmonic spring with PBC minimum image.

f32 has only 4 DST registers, so we traffic all intermediates through DFBs
and self-cycle values that are needed across multiple steps.
"""
import torch
import numpy as np
import ttnn
import ttl

TILE = 32
BOND_K = 50000.0
BOND_R0 = 1.0


def to_f32(arr, device):
    return ttnn.from_torch(
        torch.tensor(arr, dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


def make_bond_kernel(c_box, c_inv_box):
    """Factory: two-bond harmonic spring with PBC.

    Each atom can have up to 2 bond partners. Partner positions are
    pre-gathered into tile-aligned arrays. Mask=1 where bond exists, 0 otherwise.
    """
    c_k = float(BOND_K)
    c_r0 = float(BOND_R0)
    c_half = 0.5
    c_eps = 1e-8

    @ttl.kernel(grid="auto")
    def bond_kernel(own_x, own_y, own_z,
                    p1_x, p1_y, p1_z, m1,
                    p2_x, p2_y, p2_z, m2,
                    fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_tiles = own_x.shape[0] // TILE
        tiles_per_core = -(-n_tiles // grid_cols)

        # Input DFBs: own(3) + partner1(3+mask) + partner2(3+mask) = 10
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
        # Intermediate DFBs for f32 DST spilling: dx/dy/dz + rsq = 4
        dx_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        dy_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        dz_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        rsq_cb = ttl.make_dataflow_buffer_like(own_x, shape=(1, 1), buffer_factor=2)
        # Bond 1 partial results + final output = 6
        b1x_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        b1y_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        b1z_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)
        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)
        # Total: 10 + 4 + 6 = 20 DFBs

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < n_tiles:
                    # === Bond 1 ===
                    # PBC displacements, one axis at a time to limit DST
                    with ox_cb.wait() as ox, p1x_cb.wait() as p1x:
                        d = ox - p1x
                        d = d - ttl.math.fill(d, c_box) * ttl.math.floor(d * ttl.math.fill(d, c_inv_box) + ttl.math.fill(d, c_half))
                        with dx_cb.reserve() as t:
                            t.store(d)
                    with oy_cb.wait() as oy, p1y_cb.wait() as p1y:
                        d = oy - p1y
                        d = d - ttl.math.fill(d, c_box) * ttl.math.floor(d * ttl.math.fill(d, c_inv_box) + ttl.math.fill(d, c_half))
                        with dy_cb.reserve() as t:
                            t.store(d)
                    with oz_cb.wait() as oz, p1z_cb.wait() as p1z:
                        d = oz - p1z
                        d = d - ttl.math.fill(d, c_box) * ttl.math.floor(d * ttl.math.fill(d, c_inv_box) + ttl.math.fill(d, c_half))
                        with dz_cb.reserve() as t:
                            t.store(d)

                    # r_sq accumulation: one axis at a time, self-cycle displacements
                    with dx_cb.wait() as dxv:
                        with rsq_cb.reserve() as t:
                            t.store(dxv * dxv)
                        with dx_cb.reserve() as t:
                            t.store(dxv)
                    with dy_cb.wait() as dyv, rsq_cb.wait() as acc:
                        with rsq_cb.reserve() as t:
                            t.store(acc + dyv * dyv)
                        with dy_cb.reserve() as t:
                            t.store(dyv)
                    with dz_cb.wait() as dzv, rsq_cb.wait() as acc:
                        with rsq_cb.reserve() as t:
                            t.store(acc + dzv * dzv + ttl.math.fill(acc, c_eps))
                        with dz_cb.reserve() as t:
                            t.store(dzv)

                    # Force magnitude → rsq_cb
                    with rsq_cb.wait() as r_sq, m1_cb.wait() as mask:
                        r_inv = ttl.math.rsqrt(r_sq)
                        r = r_sq * r_inv
                        with rsq_cb.reserve() as t:
                            t.store(ttl.math.neg(ttl.math.fill(r_sq, c_k)) * (r - ttl.math.fill(r_sq, c_r0)) * r_inv * mask)

                    # Project f onto displacements, self-cycle f for each axis
                    with rsq_cb.wait() as fv, dx_cb.wait() as dxv:
                        with b1x_cb.reserve() as t:
                            t.store(fv * dxv)
                        with rsq_cb.reserve() as t:
                            t.store(fv)
                    with rsq_cb.wait() as fv, dy_cb.wait() as dyv:
                        with b1y_cb.reserve() as t:
                            t.store(fv * dyv)
                        with rsq_cb.reserve() as t:
                            t.store(fv)
                    with rsq_cb.wait() as fv, dz_cb.wait() as dzv:
                        with b1z_cb.reserve() as t:
                            t.store(fv * dzv)

                    # === Bond 2: same pattern ===
                    with ox_cb.wait() as ox2, p2x_cb.wait() as p2x:
                        d = ox2 - p2x
                        d = d - ttl.math.fill(d, c_box) * ttl.math.floor(d * ttl.math.fill(d, c_inv_box) + ttl.math.fill(d, c_half))
                        with dx_cb.reserve() as t:
                            t.store(d)
                    with oy_cb.wait() as oy2, p2y_cb.wait() as p2y:
                        d = oy2 - p2y
                        d = d - ttl.math.fill(d, c_box) * ttl.math.floor(d * ttl.math.fill(d, c_inv_box) + ttl.math.fill(d, c_half))
                        with dy_cb.reserve() as t:
                            t.store(d)
                    with oz_cb.wait() as oz2, p2z_cb.wait() as p2z:
                        d = oz2 - p2z
                        d = d - ttl.math.fill(d, c_box) * ttl.math.floor(d * ttl.math.fill(d, c_inv_box) + ttl.math.fill(d, c_half))
                        with dz_cb.reserve() as t:
                            t.store(d)

                    with dx_cb.wait() as dxv:
                        with rsq_cb.reserve() as t:
                            t.store(dxv * dxv)
                        with dx_cb.reserve() as t:
                            t.store(dxv)
                    with dy_cb.wait() as dyv, rsq_cb.wait() as acc:
                        with rsq_cb.reserve() as t:
                            t.store(acc + dyv * dyv)
                        with dy_cb.reserve() as t:
                            t.store(dyv)
                    with dz_cb.wait() as dzv, rsq_cb.wait() as acc:
                        with rsq_cb.reserve() as t:
                            t.store(acc + dzv * dzv + ttl.math.fill(acc, c_eps))
                        with dz_cb.reserve() as t:
                            t.store(dzv)

                    with rsq_cb.wait() as r_sq, m2_cb.wait() as mask:
                        r_inv = ttl.math.rsqrt(r_sq)
                        r = r_sq * r_inv
                        with rsq_cb.reserve() as t:
                            t.store(ttl.math.neg(ttl.math.fill(r_sq, c_k)) * (r - ttl.math.fill(r_sq, c_r0)) * r_inv * mask)

                    # Bond 2 force projection + combine with bond 1
                    with rsq_cb.wait() as fv, dx_cb.wait() as dxv, b1x_cb.wait() as fx1:
                        with fxo_cb.reserve() as t:
                            t.store(fx1 + fv * dxv)
                        with rsq_cb.reserve() as t:
                            t.store(fv)
                    with rsq_cb.wait() as fv, dy_cb.wait() as dyv, b1y_cb.wait() as fy1:
                        with fyo_cb.reserve() as t:
                            t.store(fy1 + fv * dyv)
                        with rsq_cb.reserve() as t:
                            t.store(fv)
                    with rsq_cb.wait() as fv, dz_cb.wait() as dzv, b1z_cb.wait() as fz1:
                        with fzo_cb.reserve() as t:
                            t.store(fz1 + fv * dzv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < n_tiles:
                    # Bond 1 inputs
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
                    # Bond 2 inputs (own positions re-sent)
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

    return bond_kernel


def test_bond_f32(device):
    """Two-bond kernel, f32, PBC."""
    box_length = 20.0
    bond_kern = make_bond_kernel(float(box_length), 1.0 / float(box_length))

    n_tiles = 2
    n_rows = n_tiles * TILE
    np.random.seed(42)

    ox = np.zeros((n_rows, TILE), dtype=np.float32)
    oy = np.zeros_like(ox); oz = np.zeros_like(ox)
    p1x = np.zeros_like(ox); p1y = np.zeros_like(ox); p1z = np.zeros_like(ox)
    p2x = np.zeros_like(ox); p2y = np.zeros_like(ox); p2z = np.zeros_like(ox)
    m1 = np.zeros_like(ox); m2 = np.zeros_like(ox)

    ox[:, 0] = 10.0; oy[:, 0] = 10.0; oz[:, 0] = 10.0
    p1x[:, 0] = 10.0 + np.random.uniform(0.5, 2.0, n_rows)
    p1y[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    p1z[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    p2x[:, 0] = 10.0 + np.random.uniform(-2.0, -0.5, n_rows)
    p2y[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    p2z[:, 0] = 10.0 + np.random.uniform(-0.5, 0.5, n_rows)
    m1[:, 0] = 1.0
    m2[:n_rows//2, 0] = 1.0

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
    tt = lambda a: to_f32(a, device)
    tt_fx = tt(zeros.copy()); tt_fy = tt(zeros.copy()); tt_fz = tt(zeros.copy())
    bond_kern(tt(ox), tt(oy), tt(oz),
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
    rel = max_err / max_ref if max_ref > 0 else 0
    print(f"Bond f32 test: max_ref={max_ref:.2f}, max_err={max_err:.6f}, rel={rel:.6f}")
    assert rel < 0.02, f"FAIL: relative error {rel:.6f}"
    print("PASS")


if __name__ == "__main__":
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(device_id=0,
                              worker_l1_size=default_size - 80000)
    test_bond_f32(device)
    ttnn.close_device(device)
