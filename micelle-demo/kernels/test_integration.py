"""Velocity Verlet integration kernels.

vel_half: v += 0.5 * dt * f  (f32, streaming, grid="auto")
pos_update: x = (x + dt*v) % box  (f32, streaming, grid="auto")

Ported from md_cell_list.py. These operate on cell-layout arrays
where data is in column 0 of each tile. Process x/y/z sequentially
within each tile (3 iterations per cell).
"""
import numpy as np
import torch
import ttnn
import ttl

TILE = 32


def make_vel_half_kernel(c_dt_half):
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

    return vel_half_kernel


def make_pos_update_kernel(c_dt, c_box, c_inv_box):
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

    return pos_update_kernel


def test_integration():
    device = ttnn.open_device(device_id=0)

    dt = 0.005
    box_length = 10.0

    # Simple test: 2 atoms in one cell-layout tile
    n_tiles = 1
    pos_x = np.zeros((TILE, TILE), dtype=np.float32)
    pos_y = np.zeros_like(pos_x)
    pos_z = np.zeros_like(pos_x)
    vel_x = np.zeros_like(pos_x)
    vel_y = np.zeros_like(pos_x)
    vel_z = np.zeros_like(pos_x)
    fx = np.zeros_like(pos_x)
    fy = np.zeros_like(pos_x)
    fz = np.zeros_like(pos_x)

    # Atom 0: pos=(1,2,3), vel=(0.1,0.2,0.3), force=(10,20,30)
    pos_x[0, 0] = 1.0; pos_y[0, 0] = 2.0; pos_z[0, 0] = 3.0
    vel_x[0, 0] = 0.1; vel_y[0, 0] = 0.2; vel_z[0, 0] = 0.3
    fx[0, 0] = 10.0; fy[0, 0] = 20.0; fz[0, 0] = 30.0

    # Atom 1: pos near boundary to test PBC wrap
    pos_x[1, 0] = 9.99; pos_y[1, 0] = 5.0; pos_z[1, 0] = 5.0
    vel_x[1, 0] = 1.0; vel_y[1, 0] = 0.0; vel_z[1, 0] = 0.0
    fx[1, 0] = 0.0; fy[1, 0] = 0.0; fz[1, 0] = 0.0

    def to_f32(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    tt_px = to_f32(pos_x); tt_py = to_f32(pos_y); tt_pz = to_f32(pos_z)
    tt_vx = to_f32(vel_x); tt_vy = to_f32(vel_y); tt_vz = to_f32(vel_z)
    tt_fx = to_f32(fx); tt_fy = to_f32(fy); tt_fz = to_f32(fz)

    # Test vel_half: v += 0.5*dt*f
    vel_half = make_vel_half_kernel(0.5 * dt)
    print("Testing vel_half...")
    vel_half(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)

    vx_r = ttnn.to_torch(tt_vx).float().numpy()
    vy_r = ttnn.to_torch(tt_vy).float().numpy()
    vz_r = ttnn.to_torch(tt_vz).float().numpy()

    # Expected: v0 = (0.1 + 0.5*0.005*10, 0.2 + 0.5*0.005*20, 0.3 + 0.5*0.005*30)
    #         = (0.125, 0.25, 0.375)
    exp_vx0 = 0.1 + 0.5 * dt * 10.0
    exp_vy0 = 0.2 + 0.5 * dt * 20.0
    exp_vz0 = 0.3 + 0.5 * dt * 30.0
    print(f"  Atom 0 vel: expected=({exp_vx0:.4f}, {exp_vy0:.4f}, {exp_vz0:.4f}) "
          f"got=({vx_r[0,0]:.4f}, {vy_r[0,0]:.4f}, {vz_r[0,0]:.4f})")
    assert abs(vx_r[0, 0] - exp_vx0) < 0.01, f"vx mismatch: {vx_r[0,0]} != {exp_vx0}"
    assert abs(vy_r[0, 0] - exp_vy0) < 0.01, f"vy mismatch: {vy_r[0,0]} != {exp_vy0}"
    assert abs(vz_r[0, 0] - exp_vz0) < 0.01, f"vz mismatch: {vz_r[0,0]} != {exp_vz0}"
    # Atom 1: force=0, so vel unchanged at (1.0, 0, 0)
    assert abs(vx_r[1, 0] - 1.0) < 0.01, f"v1x should be 1.0: {vx_r[1,0]}"
    print("  vel_half: PASS")

    # Test pos_update: x = (x + dt*v) % box
    pos_update = make_pos_update_kernel(dt, box_length, 1.0/box_length)
    print("Testing pos_update...")
    pos_update(tt_px, tt_py, tt_pz, tt_vx, tt_vy, tt_vz)

    px_r = ttnn.to_torch(tt_px).float().numpy()
    py_r = ttnn.to_torch(tt_py).float().numpy()
    pz_r = ttnn.to_torch(tt_pz).float().numpy()

    # Atom 0: pos = (1 + 0.005*0.125, 2 + 0.005*0.25, 3 + 0.005*0.375)
    exp_px0 = 1.0 + dt * exp_vx0
    exp_py0 = 2.0 + dt * exp_vy0
    exp_pz0 = 3.0 + dt * exp_vz0
    print(f"  Atom 0 pos: expected=({exp_px0:.6f}, {exp_py0:.6f}, {exp_pz0:.6f}) "
          f"got=({px_r[0,0]:.6f}, {py_r[0,0]:.6f}, {pz_r[0,0]:.6f})")
    assert abs(px_r[0, 0] - exp_px0) < 0.01
    assert abs(py_r[0, 0] - exp_py0) < 0.01
    assert abs(pz_r[0, 0] - exp_pz0) < 0.01

    # Atom 1: pos = (9.99 + 0.005*1.0) % 10.0 = 9.995 % 10 = 9.995
    exp_px1 = (9.99 + dt * 1.0) % box_length
    print(f"  Atom 1 pos_x: expected={exp_px1:.6f} got={px_r[1,0]:.6f}")
    assert abs(px_r[1, 0] - exp_px1) < 0.01
    print("  pos_update: PASS")

    # Test PBC wrap: move atom 1 past boundary
    # Set vel_x[1] very high so it wraps
    vx_np = np.zeros((TILE, TILE), dtype=np.float32)
    vx_np[1, 0] = 200.0  # will move 1.0 past boundary
    tt_vx2 = to_f32(vx_np)
    vy_np = np.zeros((TILE, TILE), dtype=np.float32)
    vz_np = np.zeros_like(vy_np)
    tt_vy2 = to_f32(vy_np)
    tt_vz2 = to_f32(vz_np)

    # Reset pos to 9.5
    px_np = np.zeros((TILE, TILE), dtype=np.float32)
    px_np[1, 0] = 9.5
    tt_px2 = to_f32(px_np)
    py_np = np.zeros((TILE, TILE), dtype=np.float32)
    pz_np = np.zeros_like(py_np)
    tt_py2 = to_f32(py_np)
    tt_pz2 = to_f32(pz_np)

    pos_update(tt_px2, tt_py2, tt_pz2, tt_vx2, tt_vy2, tt_vz2)
    px2_r = ttnn.to_torch(tt_px2).float().numpy()
    # 9.5 + 0.005*200 = 10.5, %10 = 0.5
    exp_wrap = (9.5 + dt * 200.0) % box_length
    print(f"  PBC wrap: expected={exp_wrap:.4f} got={px2_r[1,0]:.4f}")
    assert abs(px2_r[1, 0] - exp_wrap) < 0.05, f"PBC wrap failed: {px2_r[1,0]} != {exp_wrap}"
    print("  PBC wrap: PASS")

    print("\nAll integration tests PASS")
    ttnn.close_device(device)


if __name__ == "__main__":
    test_integration()
