"""End-to-end MD simulation with XYZ trajectory output.

Runs velocity Verlet integration with fused LJ+Coulomb forces and
harmonic bond forces on Tenstorrent hardware. Host-side cell rebuild
every N steps to handle atom migration.
"""
import time
import numpy as np
import torch
import ttnn
import ttl

TILE = 32
N_NBR = 27

from test_lj_coulomb import make_lj_coulomb_kernel, build_cell_data
from test_bond_f32 import make_bond_kernel
from test_integration import make_vel_half_kernel, make_pos_update_kernel
from params import BOND_K, BOND_R0
from input_gen import make_lipid_system
from cell_list import pack_cell_layout, extract_cell_data
from xyz_writer import write_xyz_frame


def unwrap_chains(positions, bonds, box_length, lipid_types=None, types=None):
    """Unwrap bonded chains relative to the lipid center of mass.

    1. Compute lipid COM (using minimum-image to handle PBC)
    2. Wrap all atoms so they're within half a box of the lipid COM
    3. Unwrap each bonded chain so atoms are contiguous
    This keeps the main cluster centered and prevents teleporting.
    """
    n = len(positions)
    out = positions.copy()

    # Find lipid atom indices for COM calculation
    if types is not None and lipid_types is not None:
        lipid_mask = np.isin(types, lipid_types)
    else:
        # Fall back: atoms involved in bonds are lipids
        bonded = set()
        for b in bonds:
            bonded.add(int(b[0])); bonded.add(int(b[1]))
        lipid_mask = np.array([i in bonded for i in range(n)])

    if np.any(lipid_mask):
        # Compute lipid COM using minimum-image convention
        # (use first lipid atom as reference to avoid PBC averaging artifacts)
        lip_pos = out[lipid_mask]
        ref = lip_pos[0]
        delta = lip_pos - ref
        delta -= box_length * np.floor(delta / box_length + 0.5)
        com = ref + delta.mean(axis=0)

        # Wrap ALL atoms relative to lipid COM (within half-box of COM)
        dr = out - com
        dr -= box_length * np.floor(dr / box_length + 0.5)
        out = com + dr

        # Shift so lipid COM is at box center (camera target)
        box_center = np.array([box_length / 2.0] * 3)
        out += box_center - com

    # Unwrap bonded chains so each molecule is contiguous
    adj = [[] for _ in range(n)]
    for b in bonds:
        i, j = int(b[0]), int(b[1])
        adj[i].append(j)
        adj[j].append(i)

    visited = np.zeros(n, dtype=bool)
    for root in range(n):
        if visited[root] or len(adj[root]) == 0:
            continue
        visited[root] = True
        queue = [root]
        qi = 0
        while qi < len(queue):
            parent = queue[qi]; qi += 1
            for child in adj[parent]:
                if visited[child]:
                    continue
                visited[child] = True
                dr = out[child] - out[parent]
                dr -= box_length * np.floor(dr / box_length + 0.5)
                out[child] = out[parent] + dr
                queue.append(child)

    return out


def build_bond_index_map(bonds, cell_atom_map, n_cells_total):
    """Build bond partner row-index arrays (rebuilt at each cell rebuild).

    Returns:
        p1_rows: array of partner-1 cell-layout rows (indexed by own row), -1 if none
        p2_rows: array of partner-2 cell-layout rows, -1 if none
        atom_to_row: dict mapping atom_id → cell_layout_row
    """
    atom_to_row = {}
    for cell_id, atoms in enumerate(cell_atom_map):
        for local_idx, atom_id in enumerate(atoms[:TILE]):
            atom_to_row[atom_id] = cell_id * TILE + local_idx

    partner_map = {}
    for b in bonds:
        i, j = int(b[0]), int(b[1])
        partner_map.setdefault(i, []).append(j)
        partner_map.setdefault(j, []).append(i)

    n_rows = n_cells_total * TILE
    p1_rows = np.full(n_rows, -1, dtype=int)
    p2_rows = np.full(n_rows, -1, dtype=int)

    for atom_id, row in atom_to_row.items():
        partners = partner_map.get(atom_id, [])
        if len(partners) >= 1:
            pr = atom_to_row.get(partners[0], -1)
            if pr >= 0:
                p1_rows[row] = pr
        if len(partners) >= 2:
            pr = atom_to_row.get(partners[1], -1)
            if pr >= 0:
                p2_rows[row] = pr

    return p1_rows, p2_rows, atom_to_row


def gather_bond_partners(own_px, own_py, own_pz, p1_rows, p2_rows, n_cells_total):
    """Gather partner positions from current position arrays using index map.

    Called every step to keep partner positions in sync with evolving positions.
    """
    n_rows = n_cells_total * TILE
    p1_x = np.zeros((n_rows, TILE), dtype=np.float32)
    p1_y = np.zeros_like(p1_x); p1_z = np.zeros_like(p1_x)
    m1 = np.zeros_like(p1_x)
    p2_x = np.zeros_like(p1_x)
    p2_y = np.zeros_like(p1_x); p2_z = np.zeros_like(p1_x)
    m2 = np.zeros_like(p1_x)

    has_p1 = p1_rows >= 0
    p1_x[has_p1, 0] = own_px[p1_rows[has_p1], 0]
    p1_y[has_p1, 0] = own_py[p1_rows[has_p1], 0]
    p1_z[has_p1, 0] = own_pz[p1_rows[has_p1], 0]
    m1[has_p1, 0] = 1.0

    has_p2 = p2_rows >= 0
    p2_x[has_p2, 0] = own_px[p2_rows[has_p2], 0]
    p2_y[has_p2, 0] = own_py[p2_rows[has_p2], 0]
    p2_z[has_p2, 0] = own_pz[p2_rows[has_p2], 0]
    m2[has_p2, 0] = 1.0

    return p1_x, p1_y, p1_z, m1, p2_x, p2_y, p2_z, m2


def run_md(device, n_steps=100, n_lipids=20, n_water=200, dt=0.0005,
           dump_every=50, rebuild_every=25,
           dump_path="/tmp/micelle_traj.xyz", kT=1.0):
    """Run MD simulation with trajectory output."""

    positions, types, charges, bonds, box_length = make_lipid_system(
        n_lipids=n_lipids, n_water=n_water, density=0.5, seed=42)
    n_atoms = len(positions)
    has_bonds = len(bonds) > 0
    print(f"System: {n_atoms} atoms, {n_lipids} lipids, {n_water} water, box={box_length:.2f}")
    print(f"Bonds: {len(bonds)}, Charged: {np.sum(charges != 0)}")

    # HEAD=0.5, TAIL=8.0 (strong hydrophobic), WATER=0.1 (weak with tails)
    # Pair eps = sqrt(eps_i * eps_j), so TAIL-TAIL=8.0, TAIL-WATER=sqrt(8*0.1)=0.89
    # HEAD-WATER=sqrt(0.5*0.1)=0.22, WATER-WATER=0.1
    eps_table = {0: 0.5, 1: 8.0, 2: 0.1, 3: 1.0, 4: 1.0, 5: 1.2}
    eps_per_atom = np.array([eps_table.get(t, 1.0) for t in types], dtype=np.float32)

    np.random.seed(123)
    velocities = np.random.randn(n_atoms, 3) * 0.01
    velocities -= velocities.mean(axis=0)

    # Cell size: smaller = more cells = less overflow risk during clustering
    r_cut = min(box_length / 2.0 - 0.1, 1.5)

    # Initial cell layout
    (own_px, own_py, own_pz, own_eps, own_q,
     masks, cell_atom_map, n_cells_total, n_cells_dim) = \
        build_cell_data(positions, eps_per_atom, charges.astype(np.float32), box_length, r_cut)
    print(f"Cells: {n_cells_total} ({n_cells_dim}^3)")

    vel_x, vel_y, vel_z = pack_cell_layout(velocities, cell_atom_map, n_cells_total)

    def to_f32(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Upload initial state
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
    tt_vx = to_f32(vel_x)
    tt_vy = to_f32(vel_y)
    tt_vz = to_f32(vel_z)

    # Bond index map (rebuilt at each cell rebuild) + initial partner positions
    if has_bonds:
        bp1_rows, bp2_rows, atom_to_row = build_bond_index_map(bonds, cell_atom_map, n_cells_total)
        p1_x, p1_y, p1_z, m1, p2_x, p2_y, p2_z, m2 = \
            gather_bond_partners(own_px, own_py, own_pz, bp1_rows, bp2_rows, n_cells_total)
        tt_p1x = to_f32(p1_x); tt_p1y = to_f32(p1_y); tt_p1z = to_f32(p1_z)
        tt_m1 = to_f32(m1)
        tt_p2x = to_f32(p2_x); tt_p2y = to_f32(p2_y); tt_p2z = to_f32(p2_z)
        tt_m2 = to_f32(m2)

    c_n_dim = int(n_cells_dim)
    c_dim2 = c_n_dim * c_n_dim
    c_box = float(box_length)
    c_inv_box = 1.0 / c_box

    # Build kernels
    force_kernel = make_lj_coulomb_kernel(c_n_dim, c_dim2, c_box, c_inv_box)
    vel_half = make_vel_half_kernel(0.5 * dt)
    pos_update = make_pos_update_kernel(dt, c_box, c_inv_box)
    bond_kernel = make_bond_kernel(c_box, c_inv_box) if has_bonds else None

    def read_cell_positions():
        """Read position arrays in cell-layout format (for bond partner refresh)."""
        px = ttnn.to_torch(tt_px).float().numpy()
        py = ttnn.to_torch(tt_py).float().numpy()
        pz = ttnn.to_torch(tt_pz).float().numpy()
        return px, py, pz

    def read_positions():
        px, py, pz = read_cell_positions()
        return extract_cell_data(px, py, pz, cell_atom_map, n_atoms)

    def read_velocities():
        vx = ttnn.to_torch(tt_vx).float().numpy()
        vy = ttnn.to_torch(tt_vy).float().numpy()
        vz = ttnn.to_torch(tt_vz).float().numpy()
        return extract_cell_data(vx, vy, vz, cell_atom_map, n_atoms)

    # Dump initial frame (unwrap chains for visualization)
    write_xyz_frame(dump_path,
                    unwrap_chains(positions, bonds, box_length,
                                  lipid_types=[0, 1], types=types),
                    types, box_length, step=0, mode='w', charges=charges)
    print(f"Wrote initial frame to {dump_path}")

    # Initial forces
    print("Computing initial forces...")
    force_kernel(tt_px, tt_py, tt_pz, tt_eps, tt_q, tt_masks, tt_scaler,
                 tt_fx, tt_fy, tt_fz)
    if has_bonds:
        tt_bfx = to_f32(np.zeros_like(own_px))
        tt_bfy = to_f32(np.zeros_like(own_px))
        tt_bfz = to_f32(np.zeros_like(own_px))
        bond_kernel(tt_px, tt_py, tt_pz,
                    tt_p1x, tt_p1y, tt_p1z, tt_m1,
                    tt_p2x, tt_p2y, tt_p2z, tt_m2,
                    tt_bfx, tt_bfy, tt_bfz)
        tt_fx = ttnn.add(tt_fx, tt_bfx)
        tt_fy = ttnn.add(tt_fy, tt_bfy)
        tt_fz = ttnn.add(tt_fz, tt_bfz)

        # Sanity check: bond forces should be non-zero for bonded atoms
        bfx_np = ttnn.to_torch(tt_bfx).float().numpy()
        bfy_np = ttnn.to_torch(tt_bfy).float().numpy()
        bfz_np = ttnn.to_torch(tt_bfz).float().numpy()
        bf_mag = np.sqrt(bfx_np[:, 0]**2 + bfy_np[:, 0]**2 + bfz_np[:, 0]**2)
        n_nonzero_bf = np.sum(bf_mag > 0.01)
        n_bonded = np.sum(bp1_rows >= 0) + np.sum(bp2_rows >= 0)
        print(f"Bond force sanity: {n_nonzero_bf} rows with |f_bond|>0.01, "
              f"{n_bonded} bond connections, max |f_bond|={np.max(bf_mag):.4f}")

        # Check partner distances
        cpx = own_px[:, 0]
        cpy = own_py[:, 0]
        cpz = own_pz[:, 0]
        has_p1 = bp1_rows >= 0
        if np.any(has_p1):
            dx = cpx[has_p1] - cpx[bp1_rows[has_p1]]
            dy = cpy[has_p1] - cpy[bp1_rows[has_p1]]
            dz = cpz[has_p1] - cpz[bp1_rows[has_p1]]
            dx = dx - box_length * np.floor(dx / box_length + 0.5)
            dy = dy - box_length * np.floor(dy / box_length + 0.5)
            dz = dz - box_length * np.floor(dz / box_length + 0.5)
            dists = np.sqrt(dx**2 + dy**2 + dz**2)
            print(f"Partner1 distances: min={np.min(dists):.4f}, max={np.max(dists):.4f}, "
                  f"mean={np.mean(dists):.4f}, n={len(dists)}")

        # Check total forces (LJ + bond)
        fx_np = ttnn.to_torch(tt_fx).float().numpy()
        lj_fx = fx_np[:, 0] - bfx_np[:, 0]
        bonded_rows = np.where(has_p1)[0]
        if len(bonded_rows) > 0:
            print(f"Bonded atom forces: |LJ|_max={np.max(np.abs(lj_fx[bonded_rows])):.2f}, "
                  f"|bond|_max={np.max(bf_mag[bonded_rows]):.2f}, "
                  f"|total|_max={np.max(np.abs(fx_np[bonded_rows, 0])):.2f}")

    # Warmup step
    print("Warmup step (compiling kernels)...")
    vel_half(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)
    pos_update(tt_px, tt_py, tt_pz, tt_vx, tt_vy, tt_vz)
    force_kernel(tt_px, tt_py, tt_pz, tt_eps, tt_q, tt_masks, tt_scaler,
                 tt_fx, tt_fy, tt_fz)
    if has_bonds:
        bond_kernel(tt_px, tt_py, tt_pz,
                    tt_p1x, tt_p1y, tt_p1z, tt_m1,
                    tt_p2x, tt_p2y, tt_p2z, tt_m2,
                    tt_bfx, tt_bfy, tt_bfz)
        tt_fx = ttnn.add(tt_fx, tt_bfx)
        tt_fy = ttnn.add(tt_fy, tt_bfy)
        tt_fz = ttnn.add(tt_fz, tt_bfz)
    vel_half(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)
    ttnn.synchronize_device(device)
    print("Warmup done.")

    # MD loop
    total_steps_done = 1
    n_rebuilds = 0
    step_times = []
    print(f"Running {n_steps} steps (rebuild every {rebuild_every}, dump every {dump_every})...")

    for step in range(2, n_steps + 1):
        t0 = time.time()

        # Cell rebuild
        if step > 2 and (step - 1) % rebuild_every == 0:
            t_rb = time.time()
            pos_atom = read_positions()
            vel_atom = read_velocities()

            # Diagnostic: check for out-of-box or NaN positions before rebuild
            n_oob = np.sum((pos_atom < 0) | (pos_atom > box_length))
            n_nan_pos = np.sum(~np.isfinite(pos_atom))
            if n_oob > 0 or n_nan_pos > 0:
                print(f"  *** Pre-rebuild: {n_nan_pos} NaN, {n_oob} out-of-box positions ***")
            pos_atom = pos_atom % box_length

            # Rebuild cell layout
            (own_px, own_py, own_pz, own_eps, own_q,
             masks, cell_atom_map, n_cells_total, n_cells_dim) = \
                build_cell_data(pos_atom, eps_per_atom, charges.astype(np.float32),
                                box_length, r_cut)
            max_occ = max(len(a) for a in cell_atom_map)
            if max_occ > TILE:
                print(f"  *** Cell overflow: max {max_occ} > TILE ***")
            vel_x, vel_y, vel_z = pack_cell_layout(vel_atom, cell_atom_map, n_cells_total)

            # Re-upload
            tt_px = to_f32(own_px)
            tt_py = to_f32(own_py)
            tt_pz = to_f32(own_pz)
            tt_eps = to_f32(own_eps)
            tt_q = to_f32(own_q)
            tt_masks = to_f32(masks)
            tt_vx = to_f32(vel_x)
            tt_vy = to_f32(vel_y)
            tt_vz = to_f32(vel_z)
            tt_fx = to_f32(np.zeros_like(own_px))
            tt_fy = to_f32(np.zeros_like(own_px))
            tt_fz = to_f32(np.zeros_like(own_px))

            if has_bonds:
                bp1_rows, bp2_rows, atom_to_row = build_bond_index_map(bonds, cell_atom_map, n_cells_total)
                p1_x, p1_y, p1_z, m1, p2_x, p2_y, p2_z, m2 = \
                    gather_bond_partners(own_px, own_py, own_pz, bp1_rows, bp2_rows, n_cells_total)
                tt_p1x = to_f32(p1_x); tt_p1y = to_f32(p1_y); tt_p1z = to_f32(p1_z)
                tt_m1 = to_f32(m1)
                tt_p2x = to_f32(p2_x); tt_p2y = to_f32(p2_y); tt_p2z = to_f32(p2_z)
                tt_m2 = to_f32(m2)
                tt_bfx = to_f32(np.zeros_like(own_px))
                tt_bfy = to_f32(np.zeros_like(own_px))
                tt_bfz = to_f32(np.zeros_like(own_px))

            n_rebuilds += 1
            print(f"  rebuild {n_rebuilds} @ step {step}: {time.time()-t_rb:.3f}s")

        # Velocity Verlet
        vel_half(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)
        pos_update(tt_px, tt_py, tt_pz, tt_vx, tt_vy, tt_vz)

        # Refresh bond partner positions from current positions
        if has_bonds:
            cpx, cpy, cpz = read_cell_positions()

            # NaN watchdog: catch the first step where positions go bad
            n_nan = np.sum(~np.isfinite(cpx[:, 0]))
            if n_nan > 0 and not hasattr(run_md, '_nan_reported'):
                run_md._nan_reported = True
                # Check which came first: pos_update or force kernel
                print(f"  *** NaN detected at step {step}: {n_nan} bad rows after pos_update ***")
                # Check forces
                fx_np = ttnn.to_torch(tt_fx).float().numpy()
                n_nan_f = np.sum(~np.isfinite(fx_np[:, 0]))
                print(f"  *** Forces have {n_nan_f} NaN rows (from previous step) ***")
                # Check bond forces specifically
                bfx_np = ttnn.to_torch(tt_bfx).float().numpy()
                n_nan_bf = np.sum(~np.isfinite(bfx_np[:, 0]))
                print(f"  *** Bond forces have {n_nan_bf} NaN rows ***")
                # Check max force magnitude
                f_mag = np.sqrt(fx_np[:, 0]**2)
                finite_f = f_mag[np.isfinite(f_mag)]
                if len(finite_f) > 0:
                    print(f"  *** Max finite force: {np.max(finite_f):.2f} ***")

            p1_x, p1_y, p1_z, m1, p2_x, p2_y, p2_z, m2 = \
                gather_bond_partners(cpx, cpy, cpz, bp1_rows, bp2_rows, n_cells_total)
            tt_p1x = to_f32(p1_x); tt_p1y = to_f32(p1_y); tt_p1z = to_f32(p1_z)
            tt_p2x = to_f32(p2_x); tt_p2y = to_f32(p2_y); tt_p2z = to_f32(p2_z)

        # Forces
        force_kernel(tt_px, tt_py, tt_pz, tt_eps, tt_q, tt_masks, tt_scaler,
                     tt_fx, tt_fy, tt_fz)
        if has_bonds:
            bond_kernel(tt_px, tt_py, tt_pz,
                        tt_p1x, tt_p1y, tt_p1z, tt_m1,
                        tt_p2x, tt_p2y, tt_p2z, tt_m2,
                        tt_bfx, tt_bfy, tt_bfz)
            tt_fx = ttnn.add(tt_fx, tt_bfx)
            tt_fy = ttnn.add(tt_fy, tt_bfy)
            tt_fz = ttnn.add(tt_fz, tt_bfz)

        vel_half(tt_vx, tt_vy, tt_vz, tt_fx, tt_fy, tt_fz)

        # Thermostat every step (we already do host I/O for bond refresh)
        target_ke = 0.5 * n_atoms * 3 * kT
        vel_atom = read_velocities()
        # Remove COM drift (force clamp breaks Newton's 3rd law slightly)
        vel_atom -= vel_atom.mean(axis=0)
        ke = 0.5 * np.sum(vel_atom ** 2)
        if ke > 0 and np.isfinite(ke):
            scale = np.sqrt(target_ke / ke)
            vel_atom *= scale
            vx, vy, vz = pack_cell_layout(vel_atom, cell_atom_map, n_cells_total)
            tt_vx = to_f32(vx); tt_vy = to_f32(vy); tt_vz = to_f32(vz)

        total_steps_done = step
        step_times.append(time.time() - t0)

        # Dump trajectory frame
        if step % dump_every == 0 or step == n_steps:
            ttnn.synchronize_device(device)
            pos_atom = read_positions()
            write_xyz_frame(dump_path,
                            unwrap_chains(pos_atom, bonds, box_length,
                                          lipid_types=[0, 1], types=types),
                            types, box_length, step=step)
            recent = step_times[-dump_every:] if len(step_times) >= dump_every else step_times
            avg_ms = np.mean(recent) * 1000

            # Bond length diagnostic
            if has_bonds:
                cpx_d, cpy_d, cpz_d = read_cell_positions()
                has_p1_d = bp1_rows >= 0
                if np.any(has_p1_d):
                    dx_d = cpx_d[has_p1_d, 0] - cpx_d[bp1_rows[has_p1_d], 0]
                    dy_d = cpy_d[has_p1_d, 0] - cpy_d[bp1_rows[has_p1_d], 0]
                    dz_d = cpz_d[has_p1_d, 0] - cpz_d[bp1_rows[has_p1_d], 0]
                    dx_d = dx_d - box_length * np.floor(dx_d / box_length + 0.5)
                    dy_d = dy_d - box_length * np.floor(dy_d / box_length + 0.5)
                    dz_d = dz_d - box_length * np.floor(dz_d / box_length + 0.5)
                    dists_d = np.sqrt(dx_d**2 + dy_d**2 + dz_d**2)
                    n_broken = np.sum(dists_d > 2.0)
                    print(f"  Step {step}: KE={ke:.4f}, bonds: min={np.min(dists_d):.3f} "
                          f"max={np.max(dists_d):.3f} mean={np.mean(dists_d):.3f} "
                          f"broken(>2)={n_broken}, {avg_ms:.3f}ms/step")
                else:
                    print(f"  Step {step}: KE={ke:.4f}, {avg_ms:.3f}ms/step")
            else:
                print(f"  Step {step}: KE={ke:.4f}, "
                      f"pos=[{np.min(pos_atom):.2f}, {np.max(pos_atom):.2f}], "
                      f"{avg_ms:.3f}ms/step, rebuilds={n_rebuilds}")

    total_time = sum(step_times)
    avg_ms = total_time / len(step_times) * 1000
    print(f"\nTiming: {avg_ms:.3f}ms/step avg over {len(step_times)} steps")
    print(f"Trajectory: {dump_path}")


if __name__ == "__main__":
    import os
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(device_id=0,
                              worker_l1_size=default_size - 80000)

    runs = [
        ("very_cold", 0.1),
        ("cold",      0.4),
        ("warm",      1.0),
        ("hot",       3.0),
        ("very_hot",  8.0),
    ]
    traj_dir = "/tmp/micelle_trajs"
    os.makedirs(traj_dir, exist_ok=True)

    for name, kT in runs:
        path = f"{traj_dir}/{name}.xyz"
        print(f"\n{'='*60}")
        print(f"  Run: {name}  (kT={kT})")
        print(f"{'='*60}")
        run_md(device, n_steps=1000, n_lipids=80, n_water=500, dt=0.001,
               dump_every=5, rebuild_every=25,
               dump_path=path, kT=kT)

    ttnn.close_device(device)
