"""Cell-list construction and data packing utilities.

Assigns atoms to spatial cells and packs into tile-aligned arrays
for efficient TT kernel consumption. Masks are computed on-the-fly
in the kernel to save L1 memory.
"""
import numpy as np
try:
    from .params import TILE, N_NBR, N_TYPES
except ImportError:
    from params import TILE, N_NBR, N_TYPES


def build_cell_data(positions, types, charges, box_length, r_cut):
    """Assign atoms to cells, pack into tile arrays.

    Unlike md_cell_list.py, we do NOT pre-compute masks here to save
    ~38MB of L1. Instead, we store atom counts per cell and let the
    kernel mask on-the-fly.

    Args:
        positions: (N, 3) atom positions
        types: (N,) int, bead type IDs
        charges: (N,) float, atom charges
        box_length: float
        r_cut: float, cell size (r_cut + skin)

    Returns:
        own_px, own_py, own_pz: (n_cells*TILE, TILE) position arrays
        own_q: (n_cells*TILE, TILE) charge array
        own_type: (n_cells*TILE, TILE) type ID array (as float for bf16)
        own_count: (n_cells*TILE, TILE) atom count per cell (broadcast to col 0)
        masks: (n_cells*N_NBR*TILE, TILE) self-exclusion masks
        cell_atom_map: list of lists, cell_id -> global atom indices
        n_cells_total: int
        n_cells_dim: int
    """
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

    cell_atom_map = [sort_idx[cell_starts[c]:cell_starts[c+1]].tolist()
                     for c in range(n_cells_total)]

    valid_atoms = sort_idx[valid]
    valid_cells = sorted_cell_id[valid]
    valid_local = local_idx[valid]
    rows = valid_cells * TILE + valid_local

    own_px = np.zeros((n_cells_total * TILE, TILE), dtype=np.float32)
    own_py = np.zeros_like(own_px)
    own_pz = np.zeros_like(own_px)
    own_q = np.zeros_like(own_px)
    own_type = np.zeros_like(own_px)
    own_px[rows, 0] = positions[valid_atoms, 0]
    own_py[rows, 0] = positions[valid_atoms, 1]
    own_pz[rows, 0] = positions[valid_atoms, 2]
    own_q[rows, 0] = charges[valid_atoms]
    own_type[rows, 0] = types[valid_atoms].astype(np.float32)

    # Self-exclusion masks (same as md_cell_list)
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

    return (own_px, own_py, own_pz, own_q, own_type,
            masks, cell_atom_map, n_cells_total, n_cells_dim)


def _build_cell_index_arrays(cell_atom_map):
    """Build flat row/atom index arrays for vectorized cell-layout operations."""
    rows_list, atoms_list = [], []
    for cell_id, atoms in enumerate(cell_atom_map):
        na = min(len(atoms), TILE)
        if na > 0:
            rows_list.append(cell_id * TILE + np.arange(na))
            atoms_list.append(np.array(atoms[:na]))
    if rows_list:
        return np.concatenate(rows_list), np.concatenate(atoms_list)
    return np.array([], dtype=int), np.array([], dtype=int)


def pack_cell_layout(data_3col, cell_atom_map, n_cells_total):
    """Pack per-atom (N,3) data into cell-layout tile arrays."""
    ax = np.zeros((n_cells_total * TILE, TILE), dtype=np.float32)
    ay = np.zeros_like(ax)
    az = np.zeros_like(ax)
    rows, atom_ids = _build_cell_index_arrays(cell_atom_map)
    if len(rows) > 0:
        ax[rows, 0] = data_3col[atom_ids, 0]
        ay[rows, 0] = data_3col[atom_ids, 1]
        az[rows, 0] = data_3col[atom_ids, 2]
    return ax, ay, az


def extract_cell_data(dx_np, dy_np, dz_np, cell_atom_map, n_atoms):
    """Unpack cell-layout tile arrays back to per-atom (N,3) order."""
    result = np.zeros((n_atoms, 3))
    rows, atom_ids = _build_cell_index_arrays(cell_atom_map)
    if len(rows) > 0:
        result[atom_ids, 0] = dx_np[rows, 0]
        result[atom_ids, 1] = dy_np[rows, 0]
        result[atom_ids, 2] = dz_np[rows, 0]
    return result
