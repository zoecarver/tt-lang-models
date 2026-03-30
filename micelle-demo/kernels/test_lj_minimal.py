"""Step 1: Minimal LJ kernel - single cell, basic pairwise force.

Just tests the core LJ math: broadcast+transpose to get pairwise distances,
compute LJ force magnitude, project and reduce. No neighbor loop, no epsilon
combining. Just the proven md_cell_list geometry+force pattern on one cell.
"""
import numpy as np
import torch
import ttnn
import ttl

TILE = 32


def make_minimal_lj_kernel(c_box, c_inv_box):
    c_half = 0.5
    c_lj_scale = 24.0

    @ttl.kernel(grid=(1, 1))
    def lj_one_cell(own_px, own_py, own_pz, self_mask, scaler,
                    fx_out, fy_out, fz_out):
        ox_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        oy_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        oz_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        sm_cb = ttl.make_dataflow_buffer_like(self_mask, shape=(1, 1), buffer_factor=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        ba_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        tr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        bb_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        r2_tmp = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        r2_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dx_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dy_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dz_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        fm_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ft_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        fr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz:
                with sm_cb.wait() as sm, sc_cb.wait() as sc:
                    # PBC x-displacement: broadcast own_x as rows, own_x as cols
                    with ba_cb.reserve() as ba:
                        ba.store(ttl.math.broadcast(ox, dims=[1]))
                    with tr_cb.reserve() as tr:
                        tr.store(ttl.transpose(ox))
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
                        tr.store(ttl.transpose(oy))
                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                        dy_raw = bav - bbv
                        dy_pbc = dy_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dy_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                        with r2_tmp.reserve() as r2o:
                            r2o.store(r2p + dy_pbc * dy_pbc)
                        with dy_cb.reserve() as dyo:
                            dyo.store(dy_pbc)

                    # PBC z-displacement + self-exclusion mask
                    with ba_cb.reserve() as ba:
                        ba.store(ttl.math.broadcast(oz, dims=[1]))
                    with tr_cb.reserve() as tr:
                        tr.store(ttl.transpose(oz))
                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                        dz_raw = bav - bbv
                        dz_pbc = dz_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dz_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                        with r2_cb.reserve() as r2o:
                            r2o.store(r2p + dz_pbc * dz_pbc + sm)
                        with dz_cb.reserve() as dzo:
                            dzo.store(dz_pbc)

                    # LJ force: 24/r^2 * (2/r^12 - 1/r^6)  (eps=1, sigma=1)
                    with r2_cb.wait() as r2:
                        r2_inv = ttl.math.recip(r2)
                        r4_inv = r2_inv * r2_inv
                        r6_inv = r4_inv * r2_inv
                        r12_inv = r6_inv * r6_inv
                        with fm_cb.reserve() as fmo:
                            fmo.store(ttl.math.fill(r2, c_lj_scale) * r2_inv * (r12_inv + r12_inv - r6_inv))

                    # Project force onto x, reduce over neighbors (dim 1), store
                    with fm_cb.wait() as fm:
                        with dx_cb.wait() as dxv:
                            with ft_cb.reserve() as ft:
                                ft.store(fm * dxv)
                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                            with fr_cb.wait() as frv, fxo_cb.reserve() as fxo:
                                fxo.store(frv)
                        with dy_cb.wait() as dyv:
                            with ft_cb.reserve() as ft:
                                ft.store(fm * dyv)
                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                            with fr_cb.wait() as frv, fyo_cb.reserve() as fyo:
                                fyo.store(frv)
                        with dz_cb.wait() as dzv:
                            with ft_cb.reserve() as ft:
                                ft.store(fm * dzv)
                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                            with fr_cb.wait() as frv, fzo_cb.reserve() as fzo:
                                fzo.store(frv)

        @ttl.datamovement()
        def dm_read():
            with ox_cb.reserve() as blk:
                tx = ttl.copy(own_px[0, 0], blk); tx.wait()
            with oy_cb.reserve() as blk:
                tx = ttl.copy(own_py[0, 0], blk); tx.wait()
            with oz_cb.reserve() as blk:
                tx = ttl.copy(own_pz[0, 0], blk); tx.wait()
            with sm_cb.reserve() as blk:
                tx = ttl.copy(self_mask[0, 0], blk); tx.wait()
            with sc_cb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with fxo_cb.wait() as blk:
                tx = ttl.copy(blk, fx_out[0, 0]); tx.wait()
            with fyo_cb.wait() as blk:
                tx = ttl.copy(blk, fy_out[0, 0]); tx.wait()
            with fzo_cb.wait() as blk:
                tx = ttl.copy(blk, fz_out[0, 0]); tx.wait()

    return lj_one_cell


def test_minimal():
    device = ttnn.open_device(device_id=0)

    np.random.seed(42)
    n_atoms = 4
    box_length = 10.0

    positions = np.array([
        [1.0, 1.0, 1.0],
        [2.5, 1.0, 1.0],
        [1.0, 2.5, 1.0],
        [1.0, 1.0, 2.5],
    ])

    # Pack into single tile (all atoms in one cell)
    own_px = np.zeros((TILE, TILE), dtype=np.float32)
    own_py = np.zeros_like(own_px)
    own_pz = np.zeros_like(own_px)
    for i in range(n_atoms):
        own_px[i, 0] = positions[i, 0]
        own_py[i, 0] = positions[i, 1]
        own_pz[i, 0] = positions[i, 2]

    # Self-mask: 1e6 for diagonal (self-pairs) and empty slots
    mask = np.full((TILE, TILE), 1e6, dtype=np.float32)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                mask[i, j] = 0.0

    scaler = np.ones((TILE, TILE), dtype=np.float32)
    fx = np.zeros((TILE, TILE), dtype=np.float32)
    fy = np.zeros_like(fx)
    fz = np.zeros_like(fx)

    def to_bf16(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_px = to_bf16(own_px)
    tt_py = to_bf16(own_py)
    tt_pz = to_bf16(own_pz)
    tt_mask = to_bf16(mask)
    tt_scaler = to_bf16(scaler)
    tt_fx = to_bf16(fx)
    tt_fy = to_bf16(fy)
    tt_fz = to_bf16(fz)

    kernel = make_minimal_lj_kernel(float(box_length), 1.0/box_length)
    print("Running minimal LJ kernel (single cell, 4 atoms)...")
    kernel(tt_px, tt_py, tt_pz, tt_mask, tt_scaler, tt_fx, tt_fy, tt_fz)

    fx_r = ttnn.to_torch(tt_fx).float().numpy()
    fy_r = ttnn.to_torch(tt_fy).float().numpy()
    fz_r = ttnn.to_torch(tt_fz).float().numpy()

    # Reference: O(N^2) LJ with sigma=1, eps=1
    dr = positions[:, None, :] - positions[None, :, :]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(r2, 1e10)
    r2_inv = 1.0 / r2
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2
    f_mag = 24.0 * r2_inv * (2.0 * r12_inv - r6_inv)
    ref_forces = np.sum(f_mag[:, :, None] * dr, axis=1)

    print("Results (first 4 atoms):")
    for i in range(n_atoms):
        tt_f = [fx_r[i, 0], fy_r[i, 0], fz_r[i, 0]]
        ref_f = ref_forces[i]
        print(f"  Atom {i}: ref=[{ref_f[0]:.4f}, {ref_f[1]:.4f}, {ref_f[2]:.4f}] "
              f"tt=[{tt_f[0]:.4f}, {tt_f[1]:.4f}, {tt_f[2]:.4f}]")

    max_diff = 0
    for i in range(n_atoms):
        for d, arr in enumerate([fx_r, fy_r, fz_r]):
            diff = abs(arr[i, 0] - ref_forces[i, d])
            max_diff = max(max_diff, diff)

    ref_scale = np.max(np.abs(ref_forces))
    rel_err = max_diff / (ref_scale + 1e-10)
    print(f"Max abs diff: {max_diff:.6f}, rel err: {rel_err:.6f}")
    if rel_err < 0.15:
        print("PASS")
    else:
        print("FAIL")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_minimal()
