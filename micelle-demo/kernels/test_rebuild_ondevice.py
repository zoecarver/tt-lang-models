"""Test fully on-device cell rebuild using ttnn ops.

Tests: floor, elementwise cell_id computation, sort for TILE alignment,
gather, and static masks.
"""
import time
import numpy as np
import torch
import ttnn

TILE = 32
N_NBR = 27


def test_ttnn_ops():
    """Check which ttnn ops exist and measure their cost."""
    device = ttnn.open_device(device_id=0)

    n_atoms = 2400
    box_length = 20.0
    cell_size = 4.0
    n_dim = 5
    dim2 = n_dim * n_dim
    n_cells = n_dim ** 3
    n_rows = n_cells * TILE

    np.random.seed(42)
    positions = np.random.uniform(0, box_length, (n_atoms, 3))

    # Create cell-layout position tensor (same as MD kernel)
    # Assign atoms to cells, pack into (n_rows, TILE)
    cidx = np.floor(positions / cell_size).astype(int) % n_dim
    cell_ids_ref = cidx[:, 0] * dim2 + cidx[:, 1] * n_dim + cidx[:, 2]
    sort_idx = np.argsort(cell_ids_ref, kind='stable')
    counts = np.bincount(cell_ids_ref, minlength=n_cells)
    starts = np.zeros(n_cells + 1, dtype=int)
    np.cumsum(counts, out=starts[1:])

    px_np = np.zeros((n_rows, TILE), dtype=np.float32)
    py_np = np.zeros_like(px_np)
    pz_np = np.zeros_like(px_np)
    for c in range(n_cells):
        atoms = sort_idx[starts[c]:starts[c+1]]
        for li, a in enumerate(atoms[:TILE]):
            px_np[c * TILE + li, 0] = positions[a, 0]
            py_np[c * TILE + li, 0] = positions[a, 1]
            pz_np[c * TILE + li, 0] = positions[a, 2]

    L1 = ttnn.L1_MEMORY_CONFIG
    def to_f32(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=L1)

    tt_px = to_f32(px_np)
    tt_py = to_f32(py_np)
    tt_pz = to_f32(pz_np)

    # --- Test 1: ttnn.floor ---
    print("Test 1: ttnn.floor")
    inv_cs = to_f32(np.full((n_rows, TILE), 1.0 / cell_size, dtype=np.float32))
    try:
        scaled = ttnn.mul(tt_px, inv_cs)
        floored = ttnn.floor(scaled)
        print(f"  ttnn.floor EXISTS")
        r = ttnn.to_torch(floored).float().numpy()
        print(f"  floor(px/cell_size) first 5 rows col0: {r[:5, 0]}")
    except Exception as e:
        print(f"  ttnn.floor MISSING: {e}")
        # Try alternative: typecast to uint32 and back
        try:
            scaled = ttnn.mul(tt_px, inv_cs)
            # Truncation via typecast
            as_int = ttnn.typecast(scaled, ttnn.uint32)
            floored = ttnn.typecast(as_int, ttnn.float32)
            print(f"  Workaround (typecast to uint32): works")
            r = ttnn.to_torch(floored).float().numpy()
            print(f"  floor via typecast first 5: {r[:5, 0]}")
        except Exception as e2:
            print(f"  Workaround also failed: {e2}")

    # --- Test 2: modulo via floor ---
    print("\nTest 2: modulo (x % n = x - n * floor(x/n))")
    try:
        n_dim_t = to_f32(np.full((n_rows, TILE), float(n_dim), dtype=np.float32))
        inv_n = to_f32(np.full((n_rows, TILE), 1.0 / n_dim, dtype=np.float32))
        scaled_x = ttnn.mul(tt_px, inv_cs)
        floored_x = ttnn.floor(scaled_x)
        # cell_ix = floored_x % n_dim
        div_n = ttnn.mul(floored_x, inv_n)
        floor_div = ttnn.floor(div_n)
        mod_result = ttnn.sub(floored_x, ttnn.mul(n_dim_t, floor_div))
        r = ttnn.to_torch(mod_result).float().numpy()
        print(f"  cell_ix first 5: {r[:5, 0]}")
        print(f"  Expected: {np.floor(positions[sort_idx[:5], 0] / cell_size).astype(int) % n_dim}")
    except Exception as e:
        print(f"  Failed: {e}")

    # --- Test 3: Full cell_id computation ---
    print("\nTest 3: Full cell_id = ix*dim2 + iy*dim + iz")
    try:
        inv_cs_t = to_f32(np.full((n_rows, TILE), 1.0 / cell_size, dtype=np.float32))
        n_dim_t = to_f32(np.full((n_rows, TILE), float(n_dim), dtype=np.float32))
        inv_n_t = to_f32(np.full((n_rows, TILE), 1.0 / n_dim, dtype=np.float32))
        dim2_t = to_f32(np.full((n_rows, TILE), float(dim2), dtype=np.float32))

        def cell_coord(tt_pos):
            scaled = ttnn.mul(tt_pos, inv_cs_t)
            fl = ttnn.floor(scaled)
            div = ttnn.mul(fl, inv_n_t)
            fl_div = ttnn.floor(div)
            return ttnn.sub(fl, ttnn.mul(n_dim_t, fl_div))

        ix = cell_coord(tt_px)
        iy = cell_coord(tt_py)
        iz = cell_coord(tt_pz)
        cell_id = ttnn.add(ttnn.mul(ix, dim2_t), ttnn.add(ttnn.mul(iy, n_dim_t), iz))

        r = ttnn.to_torch(cell_id).float().numpy()
        print(f"  cell_id first 10 rows col0: {r[:10, 0]}")
        # In the initial layout, row i belongs to cell i//TILE
        expected = np.arange(n_cells).repeat(TILE)
        print(f"  Expected (initial):          {expected[:10]}")
        match = np.sum(r[:n_rows, 0] == expected) / n_rows * 100
        print(f"  Match: {match:.1f}%")
    except Exception as e:
        print(f"  Failed: {e}")

    # --- Test 4: Sort by cell_id along dim=0 ---
    print("\nTest 4: ttnn.sort along dim=0")
    try:
        # Broadcast cell_id column 0 to all columns via matmul with ones
        ones = to_f32(np.ones((TILE, TILE), dtype=np.float32))
        cell_id_full = ttnn.matmul(cell_id, ones)

        # Need bf16 for sort
        cell_id_bf16 = ttnn.typecast(cell_id_full, ttnn.bfloat16)

        t0 = time.time()
        sorted_vals, sort_perm = ttnn.sort(cell_id_bf16, dim=0)
        ttnn.synchronize_device(device)
        t1 = time.time()
        print(f"  Sort dim=0 time (first): {t1-t0:.3f}s")

        t0 = time.time()
        sorted_vals, sort_perm = ttnn.sort(cell_id_bf16, dim=0)
        ttnn.synchronize_device(device)
        t1 = time.time()
        print(f"  Sort dim=0 time (compiled): {(t1-t0)*1000:.3f}ms")

        sv = ttnn.to_torch(sorted_vals).float().numpy()
        print(f"  Sorted col0 first 10: {sv[:10, 0]}")

    except Exception as e:
        print(f"  Failed: {e}")

    # --- Test 5: Gather with sort permutation ---
    print("\nTest 5: Gather positions with sort perm")
    try:
        t0 = time.time()
        new_px = ttnn.gather(tt_px, dim=0, index=sort_perm)
        ttnn.synchronize_device(device)
        t1 = time.time()
        print(f"  Gather time (first): {(t1-t0)*1000:.3f}ms")

        t0 = time.time()
        new_px = ttnn.gather(tt_px, dim=0, index=sort_perm)
        ttnn.synchronize_device(device)
        t1 = time.time()
        print(f"  Gather time (compiled): {(t1-t0)*1000:.3f}ms")
    except Exception as e:
        print(f"  Failed: {e}")

    # --- Test 6: Static masks ---
    print("\nTest 6: Static masks (self-exclusion only)")
    # Build static mask: 1e6 on diagonal for self-cell (nbr=13), 0 elsewhere
    masks = np.zeros((n_cells * N_NBR * TILE, TILE), dtype=np.float32)
    diag_mask = np.zeros((TILE, TILE), dtype=np.float32)
    np.fill_diagonal(diag_mask, 1e6)
    for c in range(n_cells):
        # Neighbor 13 is self (offset 0,0,0)
        masks[(c * N_NBR + 13) * TILE:(c * N_NBR + 13 + 1) * TILE, :] = diag_mask
    print(f"  Static mask shape: {masks.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(masks)} (expected: {n_cells * TILE})")
    print(f"  Memory: {masks.nbytes / 1e6:.1f}MB")

    ttnn.close_device(device)
    print("\nDone!")


if __name__ == "__main__":
    test_ttnn_ops()
