"""Test on-device cell-list rebuild using ttnn ops.

Goal: replace the host-side read-back/sort/re-upload cycle with
on-device ttnn.sort + ttnn.gather to enable tracing through rebuilds.

Pipeline:
1. Extract atom positions from cell-layout column 0 (matmul trick or direct)
2. Compute cell_id = floor(pos/cell_size) for x,y,z -> combine
3. ttnn.sort(cell_ids) -> sorted_ids, perm
4. ttnn.gather(data, perm) -> reordered data arrays
5. Rebuild masks (still needs cell counts - tiny host readback)
"""
import time
import numpy as np
import torch
import ttnn

TILE = 32
N_NBR = 27


def test_on_device_rebuild():
    device = ttnn.open_device(device_id=0)

    # Setup: 2400 atoms, same as micelle demo
    np.random.seed(42)
    n_atoms = 2400
    density = 0.3
    box_length = (n_atoms / density) ** (1.0 / 3.0)
    r_cut = 3.0
    skin = 0.5
    cell_r = r_cut + skin
    n_cells_dim = max(3, int(box_length / cell_r))
    cell_size = box_length / n_cells_dim
    n_cells_total = n_cells_dim ** 3
    n_padded = n_cells_total * TILE  # rows in cell layout

    print(f"System: {n_atoms} atoms, box={box_length:.2f}")
    print(f"Cells: {n_cells_total} ({n_cells_dim}^3), cell_size={cell_size:.2f}")

    # Random positions
    positions = np.random.uniform(0, box_length, (n_atoms, 3))

    # --- Test ttnn.sort on cell IDs ---
    # Compute cell_ids on host first to understand the data
    cidx = np.floor(positions / cell_size).astype(int) % n_cells_dim
    cell_ids = cidx[:, 0] * n_cells_dim**2 + cidx[:, 1] * n_cells_dim + cidx[:, 2]
    print(f"Cell ID range: [{cell_ids.min()}, {cell_ids.max()}]")
    print(f"Max atoms/cell: {np.bincount(cell_ids).max()}")

    # Pad to tile-aligned size
    n_tiles = -(-n_atoms // TILE)
    n_pad = n_tiles * TILE
    cell_ids_padded = np.full(n_pad, n_cells_total + 1, dtype=np.float32)  # large sentinel
    cell_ids_padded[:n_atoms] = cell_ids.astype(np.float32)

    # For sort: need (n_pad, TILE) shape? Or can we use 1D?
    # ttnn.sort works on last dim by default. Let's try a 2D tensor.
    # Shape it as (n_tiles, TILE) so sort operates within each row.
    # But we want a global sort...

    # Actually ttnn.sort sorts along dim=-1 (columns within each row).
    # For a global sort of N atoms, we need them in one row: (1, N) or reshape.
    # But N=2400 padded to 2432 = 76 tiles. So shape (1, 2432) = (1, 76*32).
    # In tile layout that's (TILE, 2432) effectively.

    # Let's try: shape (TILE, n_pad) with cell_ids in row 0
    sort_input = np.zeros((TILE, n_pad), dtype=np.float32)
    sort_input[0, :] = cell_ids_padded

    print(f"\nSort input shape: {sort_input.shape}")

    # Convert to bf16 since ttnn.sort only supports bf16/uint16
    tt_sort_in = ttnn.from_torch(
        torch.tensor(sort_input, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    print("Testing ttnn.sort...")
    t0 = time.time()
    sorted_vals, sorted_indices = ttnn.sort(tt_sort_in, dim=-1)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Sort time (first call, incl compile): {t1-t0:.3f}s")

    # Second call (compiled)
    t0 = time.time()
    sorted_vals, sorted_indices = ttnn.sort(tt_sort_in, dim=-1)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Sort time (compiled): {(t1-t0)*1000:.3f}ms")

    # Read back and verify
    sv = ttnn.to_torch(sorted_vals).float().numpy()
    si = ttnn.to_torch(sorted_indices).numpy()
    print(f"  Sorted values (row 0, first 20): {sv[0, :20]}")
    print(f"  Sorted indices (row 0, first 20): {si[0, :20]}")

    # Verify sort is correct
    host_sorted_idx = np.argsort(cell_ids_padded)
    host_sorted_vals = cell_ids_padded[host_sorted_idx]
    print(f"  Host sorted (first 20): {host_sorted_vals[:20]}")

    # --- Test ttnn.gather to reorder positions ---
    # Pack positions into (TILE, n_pad) with pos_x in row 0, pos_y in row 1, pos_z in row 2
    pos_packed = np.zeros((TILE, n_pad), dtype=np.float32)
    pos_packed[0, :n_atoms] = positions[:, 0]
    pos_packed[1, :n_atoms] = positions[:, 1]
    pos_packed[2, :n_atoms] = positions[:, 2]

    tt_pos = ttnn.from_torch(
        torch.tensor(pos_packed, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    print("\nTesting ttnn.gather with sort indices...")
    t0 = time.time()
    reordered_pos = ttnn.gather(tt_pos, dim=-1, index=sorted_indices)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Gather time (first call): {(t1-t0)*1000:.3f}ms")

    t0 = time.time()
    reordered_pos = ttnn.gather(tt_pos, dim=-1, index=sorted_indices)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Gather time (compiled): {(t1-t0)*1000:.3f}ms")

    rp = ttnn.to_torch(reordered_pos).float().numpy()
    print(f"  Reordered pos_x (first 10): {rp[0, :10]}")

    # --- Test computing cell_ids on device ---
    print("\nTesting on-device cell_id computation...")
    # pos_x in (TILE, n_pad) row 0
    tt_px_flat = ttnn.from_torch(
        torch.tensor(pos_packed[0:1, :].reshape(1, n_pad).repeat(TILE, axis=0),
                     dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    inv_cell = ttnn.from_torch(
        torch.full((TILE, n_pad), 1.0/cell_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    t0 = time.time()
    # cell_ix = floor(pos_x / cell_size)
    scaled = ttnn.mul(tt_px_flat, inv_cell)
    # Note: ttnn might not have floor directly. Check alternatives.
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Elementwise mul time: {(t1-t0)*1000:.3f}ms")

    # --- Full pipeline timing ---
    print("\n--- Full on-device rebuild pipeline estimate ---")
    print("  1. Extract atom positions from cell layout: matmul or gather (~1ms)")
    print("  2. Compute cell_ids (elementwise floor/mul/add): ~1ms")
    print("  3. ttnn.sort(cell_ids): measured above")
    print("  4. ttnn.gather(pos/vel/forces, perm): measured above per array")
    print("  5. Rebuild masks: host-side from cell counts (~5ms)")
    print("  6. Upload masks: ~2ms")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_on_device_rebuild()
