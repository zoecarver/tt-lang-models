"""On-device MD step loop with ttnn.gather-based position refresh.

All data stays in L1. Per-step pipeline:
  1. vel_half(vel, forces)
  2. pos_update(pos, vel)
  3. pos_full = ttnn.matmul(pos, ones)   # broadcast col 0 to all columns
  4. nbr_xyz = ttnn.gather(pos_full, 0, nbr_idx)  # refresh neighbor positions
  5. force_kernel(pos, nbr, ...) → forces
  6. vel_half(vel, forces)

Neighbor index tensor is static between rebuilds (host-side, infrequent).
Positions are gathered fresh every step on device. Zero host round-trips.
"""
import time
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


def build_neighbor_index(positions, eps_per_atom, charges, box_length, r_cut):
    """Build neighbor index tensor + static pair data (eps, qq, mask).

    Returns index tensor (uint32) for ttnn.gather, plus eps/qq/mask arrays
    that are static between rebuilds.
    """
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
    print(f"  Max neighbors/atom: {max_nbr}, padded to {n_nbr_tiles * TILE} ({n_nbr_tiles} tiles)")

    n_groups = -(-n // TILE)
    n_padded = n_groups * TILE
    total_nbr_rows = n_groups * n_nbr_tiles * TILE

    # Neighbor index: atom ID for each (row, col) in neighbor tiles
    # Padded slots point to atom 0 (safe: mask will zero out their forces)
    nbr_idx = np.zeros((total_nbr_rows, TILE), dtype=np.uint32)
    nbr_eps = np.zeros((total_nbr_rows, TILE), dtype=np.float32)
    nbr_qq = np.zeros((total_nbr_rows, TILE), dtype=np.float32)
    nbr_mask = np.full((total_nbr_rows, TILE), 1e6, dtype=np.float32)

    for i in range(n):
        g = i // TILE
        k = i % TILE
        for slot, j in enumerate(neighbor_lists[i]):
            t = slot // TILE
            col = slot % TILE
            row = (g * n_nbr_tiles + t) * TILE + k
            nbr_idx[row, col] = j
            nbr_eps[row, col] = np.sqrt(eps_per_atom[i] * eps_per_atom[j])
            nbr_qq[row, col] = charges[i] * charges[j]
            nbr_mask[row, col] = 0.0

    return nbr_idx, nbr_eps, nbr_qq, nbr_mask, n_groups, n_nbr_tiles, neighbor_lists


# Import force kernel factory from test_pair_forces
from test_pair_forces import make_pair_force_kernel

# Import integration kernels
from test_integration import make_vel_half_kernel, make_pos_update_kernel


def test_md_ondevice():
    try:
        default_size = ttnn.device.get_max_worker_l1_unreserved_size()
        device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 80000)
    except AttributeError:
        device = ttnn.open_device(device_id=0)

    np.random.seed(42)
    n_atoms = 1024
    density = 0.3
    box_length = (n_atoms / density) ** (1.0 / 3.0)
    dt = 0.0005

    # Simple lattice system
    n_side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
    spacing = box_length / n_side
    positions = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(positions) < n_atoms:
                    positions.append([(ix+0.5)*spacing, (iy+0.5)*spacing, (iz+0.5)*spacing])
    positions = np.array(positions[:n_atoms], dtype=np.float64)
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

    # Build neighbor index (host-side, done once or on rebuild)
    nbr_idx_np, nbr_eps_np, nbr_qq_np, nbr_mask_np, n_groups, n_nbr_tiles, _ = \
        build_neighbor_index(positions, eps_per_atom, charges, box_length, r_cut)
    n_padded = n_groups * TILE
    print(f"Groups: {n_groups}, Neighbor tiles/group: {n_nbr_tiles}")

    # Initial velocities
    velocities = np.random.randn(n_atoms, 3) * 0.01
    velocities -= velocities.mean(axis=0)

    # Pack into column-0 tile arrays
    pos_x_np = np.zeros((n_padded, TILE), dtype=np.float32)
    pos_y_np = np.zeros_like(pos_x_np)
    pos_z_np = np.zeros_like(pos_x_np)
    pos_x_np[:n_atoms, 0] = positions[:, 0]
    pos_y_np[:n_atoms, 0] = positions[:, 1]
    pos_z_np[:n_atoms, 0] = positions[:, 2]

    vel_x_np = np.zeros_like(pos_x_np)
    vel_y_np = np.zeros_like(pos_x_np)
    vel_z_np = np.zeros_like(pos_x_np)
    vel_x_np[:n_atoms, 0] = velocities[:, 0]
    vel_y_np[:n_atoms, 0] = velocities[:, 1]
    vel_z_np[:n_atoms, 0] = velocities[:, 2]

    # Upload everything to L1
    L1 = ttnn.L1_MEMORY_CONFIG

    def to_f32(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=L1)

    tt_px = to_f32(pos_x_np)
    tt_py = to_f32(pos_y_np)
    tt_pz = to_f32(pos_z_np)
    tt_vx = to_f32(vel_x_np)
    tt_vy = to_f32(vel_y_np)
    tt_vz = to_f32(vel_z_np)
    tt_fx = to_f32(np.zeros_like(pos_x_np))
    tt_fy = to_f32(np.zeros_like(pos_x_np))
    tt_fz = to_f32(np.zeros_like(pos_x_np))

    # Neighbor index (uint32) and static pair data
    tt_nbr_idx = ttnn.from_torch(
        torch.tensor(nbr_idx_np.astype(np.int64), dtype=torch.int64),
        dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=L1)
    tt_nbr_eps = to_f32(nbr_eps_np)
    tt_nbr_qq = to_f32(nbr_qq_np)
    tt_nbr_mask = to_f32(nbr_mask_np)

    # Scaler (32x32 ones) and broadcast matrix (32x32 ones for matmul)
    tt_scaler = to_f32(np.ones((TILE, TILE), dtype=np.float32))
    tt_ones = to_f32(np.ones((TILE, TILE), dtype=np.float32))

    # Build kernels
    c_box = float(box_length)
    c_inv_box = 1.0 / c_box
    force_kernel = make_pair_force_kernel(c_box, c_inv_box, n_nbr_tiles)
    vel_half = make_vel_half_kernel(0.5 * dt)
    pos_update = make_pos_update_kernel(dt, c_box, c_inv_box)

    def gather_neighbor_positions():
        """Refresh neighbor positions from current atom positions via gather."""
        # Broadcast column 0 to all columns: pos_full = pos @ ones
        px_full = ttnn.matmul(tt_px, tt_ones)
        py_full = ttnn.matmul(tt_py, tt_ones)
        pz_full = ttnn.matmul(tt_pz, tt_ones)
        # Gather neighbor positions using index tensor
        nbr_x = ttnn.gather(px_full, dim=0, index=tt_nbr_idx)
        nbr_y = ttnn.gather(py_full, dim=0, index=tt_nbr_idx)
        nbr_z = ttnn.gather(pz_full, dim=0, index=tt_nbr_idx)
        # Free temporaries
        px_full.deallocate()
        py_full.deallocate()
        pz_full.deallocate()
        return nbr_x, nbr_y, nbr_z

    def run_one_step():
        """Full velocity Verlet step, all on device."""
        vel_half(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)
        pos_update(tt_px, tt_py, tt_pz, tt_vx, tt_vy, tt_vz)
        nbr_x, nbr_y, nbr_z = gather_neighbor_positions()
        force_kernel(tt_px, tt_py, tt_pz,
                     nbr_x, nbr_y, nbr_z, tt_nbr_eps, tt_nbr_qq, tt_nbr_mask,
                     tt_scaler,
                     tt_fx, tt_fy, tt_fz)
        nbr_x.deallocate()
        nbr_y.deallocate()
        nbr_z.deallocate()
        vel_half(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)

    # Initial forces
    print("Computing initial forces...")
    nbr_x, nbr_y, nbr_z = gather_neighbor_positions()
    force_kernel(tt_px, tt_py, tt_pz,
                 nbr_x, nbr_y, nbr_z, tt_nbr_eps, tt_nbr_qq, tt_nbr_mask,
                 tt_scaler,
                 tt_fx, tt_fy, tt_fz)
    nbr_x.deallocate()
    nbr_y.deallocate()
    nbr_z.deallocate()

    # Warmup step
    print("Warmup step...")
    run_one_step()
    ttnn.synchronize_device(device)

    # Timing
    n_steps = 50
    print(f"\nRunning {n_steps} steps (all on device)...")
    ttnn.synchronize_device(device)
    t0 = time.time()
    for step in range(n_steps):
        run_one_step()
    ttnn.synchronize_device(device)
    t1 = time.time()

    total_ms = (t1 - t0) * 1000
    ms_per_step = total_ms / n_steps
    print(f"Total: {total_ms:.1f}ms, {ms_per_step:.3f}ms/step")

    # Read back positions to verify
    px_np = ttnn.to_torch(tt_px).float().numpy()
    py_np = ttnn.to_torch(tt_py).float().numpy()
    pz_np = ttnn.to_torch(tt_pz).float().numpy()
    vx_np = ttnn.to_torch(tt_vx).float().numpy()
    vy_np = ttnn.to_torch(tt_vy).float().numpy()
    vz_np = ttnn.to_torch(tt_vz).float().numpy()

    pos_final = np.stack([px_np[:n_atoms, 0], py_np[:n_atoms, 0], pz_np[:n_atoms, 0]], axis=1)
    vel_final = np.stack([vx_np[:n_atoms, 0], vy_np[:n_atoms, 0], vz_np[:n_atoms, 0]], axis=1)

    ke = 0.5 * np.sum(vel_final ** 2)
    print(f"Final KE: {ke:.6f}")
    print(f"Position range: [{np.min(pos_final):.2f}, {np.max(pos_final):.2f}]")
    print(f"Velocity range: [{np.min(vel_final):.4f}, {np.max(vel_final):.4f}]")

    # Check positions are within box
    assert np.all(pos_final >= 0) and np.all(pos_final < box_length), \
        f"Positions escaped box! [{np.min(pos_final):.2f}, {np.max(pos_final):.2f}]"
    print("Positions within box: OK")

    # Check KE is reasonable (not exploding)
    ke_initial = 0.5 * np.sum(velocities ** 2)
    print(f"KE ratio (final/initial): {ke / ke_initial:.4f}")
    if ke / ke_initial > 100:
        print("WARNING: KE exploded! Check dt or force accuracy.")
    else:
        print("Energy looks stable.")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_md_ondevice()
