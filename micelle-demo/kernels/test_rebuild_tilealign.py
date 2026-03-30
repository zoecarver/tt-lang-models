"""Test TILE-aligned on-device rebuild using prefix-sum approach.

Pipeline:
1. Sort by cell_id → contiguous but not TILE-aligned
2. Comparison matrix: indicator[r,c] = 1 if sorted_cid[r] < c
3. Sum indicator along dim=0 → cell_starts[c]
4. Build TILE-aligned gather index: gather_idx[r] = cell_starts[r//TILE] + r%TILE
5. Second gather → exact TILE alignment
"""
import time
import numpy as np
import torch
import ttnn

TILE = 32


def test_tile_aligned_rebuild():
    device = ttnn.open_device(device_id=0)

    n_atoms = 2400
    box_length = 20.0
    cell_size = 4.0
    n_dim = 5
    dim2 = n_dim * n_dim
    n_cells = n_dim ** 3  # 125
    n_rows = n_cells * TILE  # 4000

    # Pad n_cells to tile alignment for the comparison matrix
    n_cells_padded = ((n_cells + TILE - 1) // TILE) * TILE  # 128

    np.random.seed(42)
    positions = np.random.uniform(0, box_length, (n_atoms, 3))

    # --- Host reference: build cell layout ---
    cidx = np.floor(positions / cell_size).astype(int) % n_dim
    cell_ids_ref = cidx[:, 0] * dim2 + cidx[:, 1] * n_dim + cidx[:, 2]
    sort_idx = np.argsort(cell_ids_ref, kind='stable')
    counts = np.bincount(cell_ids_ref, minlength=n_cells)
    starts = np.zeros(n_cells + 1, dtype=int)
    np.cumsum(counts, out=starts[1:])

    # Build reference TILE-aligned layout (use f32 to match device precision)
    positions_f32 = positions.astype(np.float32)
    px_ref = np.zeros((n_rows, TILE), dtype=np.float32)
    py_ref = np.zeros((n_rows, TILE), dtype=np.float32)
    pz_ref = np.zeros((n_rows, TILE), dtype=np.float32)
    atom_valid_ref = np.full((n_rows, TILE), 1e6, dtype=np.float32)
    for c in range(n_cells):
        atoms = sort_idx[starts[c]:starts[c + 1]]
        for li, a in enumerate(atoms[:TILE]):
            px_ref[c * TILE + li, 0] = positions_f32[a, 0]
            py_ref[c * TILE + li, 0] = positions_f32[a, 1]
            pz_ref[c * TILE + li, 0] = positions_f32[a, 2]
            atom_valid_ref[c * TILE + li, 0] = 0.0

    # --- Build initial scrambled layout (simulating mid-simulation state) ---
    px_np = np.zeros((n_rows, TILE), dtype=np.float32)
    py_np = np.zeros((n_rows, TILE), dtype=np.float32)
    pz_np = np.zeros((n_rows, TILE), dtype=np.float32)
    av_np = np.full((n_rows, TILE), 1e6, dtype=np.float32)

    # Put atoms in scrambled cell positions
    perm = np.random.permutation(n_atoms)
    for i, a in enumerate(perm):
        cell_slot = i // TILE
        row_in_cell = i % TILE
        r = cell_slot * TILE + row_in_cell
        if r < n_rows:
            px_np[r, 0] = positions_f32[a, 0]
            py_np[r, 0] = positions_f32[a, 1]
            pz_np[r, 0] = positions_f32[a, 2]
            av_np[r, 0] = 0.0

    L1 = ttnn.L1_MEMORY_CONFIG

    def to_f32(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=L1)

    tt_px = to_f32(px_np)
    tt_py = to_f32(py_np)
    tt_pz = to_f32(pz_np)
    tt_av = to_f32(av_np)

    # --- Key matrix: extract col 0 and broadcast to all columns ---
    # matmul(X, col0_to_all) where col0_to_all[k,c] = 1 if k==0 else 0
    # gives output[r,c] = X[r,0] for all c
    col0_to_all_np = np.zeros((TILE, TILE), dtype=np.float32)
    col0_to_all_np[0, :] = 1.0
    tt_col0_to_all = to_f32(col0_to_all_np)

    # --- Static tensors for rebuild ---
    inv_cs = to_f32(np.full((n_rows, TILE), 1.0 / cell_size, dtype=np.float32))
    n_dim_t = to_f32(np.full((n_rows, TILE), float(n_dim), dtype=np.float32))
    inv_n = to_f32(np.full((n_rows, TILE), 1.0 / n_dim, dtype=np.float32))
    dim2_t = to_f32(np.full((n_rows, TILE), float(dim2), dtype=np.float32))
    sentinel_t = to_f32(np.full((n_rows, TILE), float(n_cells), dtype=np.float32))
    one_t = to_f32(np.full((n_rows, TILE), 1.0, dtype=np.float32))
    zero_t = to_f32(np.full((n_rows, TILE), 0.0, dtype=np.float32))
    ones_tile = to_f32(np.ones((TILE, TILE), dtype=np.float32))

    # Reference tensor for comparison: ref[r, c] = c
    ref_np = np.zeros((n_rows, n_cells_padded), dtype=np.float32)
    for c in range(n_cells_padded):
        ref_np[:, c] = float(c)
    tt_ref = to_f32(ref_np)

    # Ones broadcast: (TILE, n_cells_padded) for broadcasting col0 to comparison width
    ones_broadcast_np = np.zeros((TILE, n_cells_padded), dtype=np.float32)
    ones_broadcast_np[0, :] = 1.0  # Extract col 0, broadcast to n_cells_padded cols
    tt_ones_broadcast = to_f32(ones_broadcast_np)

    # Ones row for summing: (TILE, n_rows) with row 0 = 1
    ones_row_np = np.zeros((TILE, n_rows), dtype=np.float32)
    ones_row_np[0, :] = 1.0
    tt_ones_row = to_f32(ones_row_np)

    # Selector: (n_rows, n_cells_padded) where selector[r, r//TILE] = 1
    selector_np = np.zeros((n_rows, n_cells_padded), dtype=np.float32)
    for r in range(n_rows):
        c = r // TILE
        if c < n_cells_padded:
            selector_np[r, c] = 1.0
    tt_selector = to_f32(selector_np)

    # Next-cell selector: (n_rows, n_cells_padded) where next_selector[r, r//TILE + 1] = 1
    next_selector_np = np.zeros((n_rows, n_cells_padded), dtype=np.float32)
    for r in range(n_rows):
        c = r // TILE + 1
        if c < n_cells_padded:
            next_selector_np[r, c] = 1.0
    tt_next_selector = to_f32(next_selector_np)

    # Row offset: (n_rows, TILE) where all cols = r % TILE
    row_offset_np = np.zeros((n_rows, TILE), dtype=np.float32)
    for r in range(n_rows):
        row_offset_np[r, :] = float(r % TILE)
    tt_row_offset = to_f32(row_offset_np)

    # Padding row index (last row is guaranteed padding after sort)
    padding_row_t = to_f32(np.full((n_rows, TILE), float(n_rows - 1), dtype=np.float32))

    # --- Step 1: Compute cell_ids from positions ---
    print("Step 1: Compute cell_ids")
    def cell_coord_dev(tt_pos):
        scaled = ttnn.mul(tt_pos, inv_cs)
        fl = ttnn.floor(scaled)
        div = ttnn.mul(fl, inv_n)
        fl_div = ttnn.floor(div)
        return ttnn.sub(fl, ttnn.mul(n_dim_t, fl_div))

    ix = cell_coord_dev(tt_px)
    iy = cell_coord_dev(tt_py)
    iz = cell_coord_dev(tt_pz)
    cell_id = ttnn.add(ttnn.mul(ix, dim2_t), ttnn.add(ttnn.mul(iy, n_dim_t), iz))

    # Extract col 0 of atom_valid and broadcast to all cols
    av_broadcast = ttnn.matmul(tt_av, tt_col0_to_all)  # [r,c] = av[r,0]
    is_padding = ttnn.gt(av_broadcast, zero_t)  # 1.0 where padding
    is_real = ttnn.sub(one_t, is_padding)
    cell_id_safe = ttnn.add(ttnn.mul(cell_id, is_real), ttnn.mul(sentinel_t, is_padding))

    # Extract col 0 and broadcast for sort key
    cell_id_full = ttnn.matmul(cell_id_safe, tt_col0_to_all)

    cid_np = ttnn.to_torch(cell_id_full).float().numpy()
    print(f"  Cell IDs first 5 col0: {cid_np[:5, 0]}")
    print(f"  Cell IDs unique real: {len(np.unique(cid_np[cid_np[:, 0] < n_cells, 0]))}")
    print(f"  Sentinel count: {np.sum(cid_np[:, 0] >= n_cells)}")

    # --- Step 2: Sort by cell_id ---
    print("\nStep 2: Sort by cell_id")
    cell_id_bf16 = ttnn.typecast(cell_id_full, ttnn.bfloat16)

    t0 = time.time()
    sorted_vals, sort_perm = ttnn.sort(cell_id_bf16, dim=0)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Sort time: {(t1-t0)*1000:.1f}ms")

    # First gather: reorder all arrays by cell_id
    sorted_px = ttnn.gather(tt_px, dim=0, index=sort_perm)
    sorted_py = ttnn.gather(tt_py, dim=0, index=sort_perm)
    sorted_pz = ttnn.gather(tt_pz, dim=0, index=sort_perm)
    sorted_av = ttnn.gather(tt_av, dim=0, index=sort_perm)

    sv_np = ttnn.to_torch(sorted_vals).float().numpy()
    print(f"  Sorted cell_ids first 10: {sv_np[:10, 0]}")
    print(f"  Sorted cell_ids at row 2400: {sv_np[2400, 0]} (should be sentinel={n_cells})")

    # --- Step 3: Compute cell_starts via comparison matrix ---
    print("\nStep 3: Cell starts via comparison matrix")
    # sorted_cid col 0 has correct values. Broadcast to (n_rows, n_cells_padded).
    sorted_cid_f32 = ttnn.typecast(sorted_vals, ttnn.float32)
    # Use col0 extractor to get clean (n_rows, n_cells_padded) broadcast
    sorted_cid_broadcast = ttnn.matmul(sorted_cid_f32, tt_ones_broadcast)

    # indicator[r, c] = 1 if sorted_cid[r] < c, else 0
    t0 = time.time()
    indicator = ttnn.lt(sorted_cid_broadcast, tt_ref)
    indicator_f32 = ttnn.typecast(indicator, ttnn.float32)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Comparison time: {(t1-t0)*1000:.1f}ms")

    # Sum along dim=0: cell_starts[c] = sum_r indicator[r, c]
    # (TILE, n_rows) @ (n_rows, n_cells_padded) → (TILE, n_cells_padded)
    # Row 0 of result has cell_starts.
    t0 = time.time()
    cell_starts_raw = ttnn.matmul(tt_ones_row, indicator_f32)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Sum matmul time: {(t1-t0)*1000:.1f}ms")

    cs_np = ttnn.to_torch(cell_starts_raw).float().numpy()
    print(f"  cell_starts first 10: {cs_np[0, :10]}")

    # Verify against host
    sorted_cid_host = sv_np[:, 0]
    host_starts = np.array([np.sum(sorted_cid_host < c) for c in range(10)])
    print(f"  Host ref      first 10: {host_starts}")

    # --- Step 4: Build TILE-aligned gather index with overflow capping ---
    print("\nStep 4: Build TILE-aligned gather index")
    # Transpose cell_starts: (TILE, n_cells_padded) → (n_cells_padded, TILE)
    cell_starts_col = ttnn.transpose(cell_starts_raw, 0, 1)

    # Expand cell_starts[c] to all rows in cell c's TILE block
    t0 = time.time()
    curr_starts = ttnn.matmul(tt_selector, cell_starts_col)
    next_starts = ttnn.matmul(tt_next_selector, cell_starts_col)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Expand matmul time: {(t1-t0)*1000:.1f}ms")

    # Broadcast col 0 to all cols
    curr_starts_full = ttnn.matmul(curr_starts, tt_col0_to_all)
    next_starts_full = ttnn.matmul(next_starts, tt_col0_to_all)

    # Count per cell: count[c] = next_starts[c] - curr_starts[c]
    count_expanded = ttnn.sub(next_starts_full, curr_starts_full)

    # Uncapped gather index
    gather_idx_f32 = ttnn.add(curr_starts_full, tt_row_offset)

    # Overflow: row_offset >= count means this slot exceeds the cell's atom count
    # Use lt: valid = lt(row_offset, count) → 1 where k < count, 0 where k >= count
    valid_mask = ttnn.lt(tt_row_offset, count_expanded)
    valid_f32 = ttnn.typecast(valid_mask, ttnn.float32)

    # Where valid, use gather_idx; where overflow, use padding row (last row)
    # safe_idx = gather_idx * valid + padding_row * (1 - valid)
    inv_valid = ttnn.sub(one_t, valid_f32)
    safe_idx = ttnn.add(ttnn.mul(gather_idx_f32, valid_f32),
                        ttnn.mul(padding_row_t, inv_valid))

    gather_idx_u32 = ttnn.typecast(safe_idx, ttnn.uint32)

    gi_np = ttnn.to_torch(gather_idx_u32).numpy()
    print(f"  Gather index first 5: {gi_np[:5, 0]}")
    print(f"  Gather index rows 32-36: {gi_np[32:36, 0]}")
    print(f"  Gather index rows 64-68: {gi_np[64:68, 0]}")

    # --- Step 5: Second gather for TILE alignment ---
    print("\nStep 5: TILE-aligned gather")
    t0 = time.time()
    aligned_px = ttnn.gather(sorted_px, dim=0, index=gather_idx_u32)
    aligned_py = ttnn.gather(sorted_py, dim=0, index=gather_idx_u32)
    aligned_pz = ttnn.gather(sorted_pz, dim=0, index=gather_idx_u32)
    aligned_av = ttnn.gather(sorted_av, dim=0, index=gather_idx_u32)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  Gather time: {(t1-t0)*1000:.1f}ms")

    # --- Verification ---
    apx_np = ttnn.to_torch(aligned_px).float().numpy()
    apy_np = ttnn.to_torch(aligned_py).float().numpy()
    apz_np = ttnn.to_torch(aligned_pz).float().numpy()
    aav_np = ttnn.to_torch(aligned_av).float().numpy()

    print("\nVerification:")
    n_correct_cells = 0
    mismatches = []
    for c in range(n_cells):
        ref_atoms = []
        for li in range(TILE):
            if atom_valid_ref[c * TILE + li, 0] < 0.5:
                ref_atoms.append((px_ref[c * TILE + li, 0],
                                  py_ref[c * TILE + li, 0],
                                  pz_ref[c * TILE + li, 0]))

        got_atoms = []
        for li in range(TILE):
            if aav_np[c * TILE + li, 0] < 0.5:
                got_atoms.append((apx_np[c * TILE + li, 0],
                                  apy_np[c * TILE + li, 0],
                                  apz_np[c * TILE + li, 0]))

        # Match atoms with tolerance (device f32 is not IEEE)
        matched = len(ref_atoms) == len(got_atoms)
        if matched and ref_atoms:
            ref_arr = np.array(ref_atoms)
            got_arr = np.array(got_atoms)
            # Sort by x then y then z for stable ordering
            ref_arr = ref_arr[np.lexsort(ref_arr.T)]
            got_arr = got_arr[np.lexsort(got_arr.T)]
            if not np.allclose(ref_arr, got_arr, atol=0.05, rtol=1e-2):
                matched = False
        if matched:
            n_correct_cells += 1
        else:
            mismatches.append((c, len(ref_atoms), len(got_atoms)))

    print(f"  Correct cells: {n_correct_cells}/{n_cells} ({n_correct_cells/n_cells*100:.0f}%)")
    if mismatches:
        print(f"  First 5 mismatches: {mismatches[:5]}")

    n_real_ref = np.sum(atom_valid_ref[:, 0] < 0.5)
    n_real_got = np.sum(aav_np[:, 0] < 0.5)
    print(f"  Real atoms: ref={n_real_ref}, got={n_real_got}")

    # --- Full pipeline timing (no readback) ---
    print("\n--- Full rebuild pipeline timing ---")
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(5):
        # 1. Cell IDs
        ix2 = cell_coord_dev(tt_px)
        iy2 = cell_coord_dev(tt_py)
        iz2 = cell_coord_dev(tt_pz)
        cid2 = ttnn.add(ttnn.mul(ix2, dim2_t), ttnn.add(ttnn.mul(iy2, n_dim_t), iz2))
        av_bc2 = ttnn.matmul(tt_av, tt_col0_to_all)
        is_pad2 = ttnn.gt(av_bc2, zero_t)
        is_real2 = ttnn.sub(one_t, is_pad2)
        cid_safe2 = ttnn.add(ttnn.mul(cid2, is_real2), ttnn.mul(sentinel_t, is_pad2))
        cid_full2 = ttnn.matmul(cid_safe2, tt_col0_to_all)
        cid_bf16_2 = ttnn.typecast(cid_full2, ttnn.bfloat16)

        # 2. Sort
        sv2, sp2 = ttnn.sort(cid_bf16_2, dim=0)

        # 3. First gather
        spx2 = ttnn.gather(tt_px, dim=0, index=sp2)
        spy2 = ttnn.gather(tt_py, dim=0, index=sp2)
        spz2 = ttnn.gather(tt_pz, dim=0, index=sp2)
        sav2 = ttnn.gather(tt_av, dim=0, index=sp2)

        # 4. Cell starts + overflow capping
        scid_f32 = ttnn.typecast(sv2, ttnn.float32)
        scid_bc = ttnn.matmul(scid_f32, tt_ones_broadcast)
        ind = ttnn.lt(scid_bc, tt_ref)
        ind_f32 = ttnn.typecast(ind, ttnn.float32)
        cs = ttnn.matmul(tt_ones_row, ind_f32)
        cs_col = ttnn.transpose(cs, 0, 1)
        cs_curr = ttnn.matmul(tt_selector, cs_col)
        cs_next = ttnn.matmul(tt_next_selector, cs_col)
        cs_curr_f = ttnn.matmul(cs_curr, tt_col0_to_all)
        cs_next_f = ttnn.matmul(cs_next, tt_col0_to_all)
        cnt = ttnn.sub(cs_next_f, cs_curr_f)
        gi = ttnn.add(cs_curr_f, tt_row_offset)
        vmask = ttnn.lt(tt_row_offset, cnt)
        vf = ttnn.typecast(vmask, ttnn.float32)
        ivf = ttnn.sub(one_t, vf)
        gi_safe = ttnn.add(ttnn.mul(gi, vf), ttnn.mul(padding_row_t, ivf))
        gi_u32 = ttnn.typecast(gi_safe, ttnn.uint32)

        # 5. Second gather
        tt_px = ttnn.gather(spx2, dim=0, index=gi_u32)
        tt_py = ttnn.gather(spy2, dim=0, index=gi_u32)
        tt_pz = ttnn.gather(spz2, dim=0, index=gi_u32)
        tt_av = ttnn.gather(sav2, dim=0, index=gi_u32)

    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  5 rebuilds: {(t1-t0)*1000:.1f}ms ({(t1-t0)*200:.1f}ms/rebuild)")

    ttnn.close_device(device)
    print("\nDone!")


if __name__ == "__main__":
    test_tile_aligned_rebuild()
