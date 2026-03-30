"""XYZ trajectory writer for Ovito/VMD visualization.

Writes standard XYZ format with atom type names and optional properties.
Each frame is appended to the same file for trajectory viewing.
"""
import numpy as np

try:
    from .params import TYPE_NAMES
except ImportError:
    from params import TYPE_NAMES


def write_xyz_frame(filepath, positions, types, box_length, step=0,
                    mode='a', charges=None):
    """Write one XYZ frame.

    Args:
        filepath: output file path
        positions: (N, 3) atom positions
        types: (N,) int type IDs
        box_length: float, for comment line
        step: int, step number for comment
        mode: 'w' for first frame, 'a' to append
        charges: optional (N,) charges for extended XYZ
    """
    n = len(positions)
    with open(filepath, mode) as f:
        f.write(f"{n}\n")
        f.write(f"Step={step} Box={box_length:.4f}\n")
        for i in range(n):
            name = TYPE_NAMES[types[i]] if types[i] < len(TYPE_NAMES) else "X"
            x, y, z = positions[i]
            if charges is not None:
                f.write(f"{name} {x:.6f} {y:.6f} {z:.6f} {charges[i]:.4f}\n")
            else:
                f.write(f"{name} {x:.6f} {y:.6f} {z:.6f}\n")


def write_lammps_dump(filepath, positions, types, box_length, step=0, mode='a'):
    """Write one LAMMPS dump frame (better for Ovito with box info)."""
    n = len(positions)
    with open(filepath, mode) as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{n}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0.0 {box_length:.6f}\n")
        f.write(f"0.0 {box_length:.6f}\n")
        f.write(f"0.0 {box_length:.6f}\n")
        f.write("ITEM: ATOMS id type x y z\n")
        for i in range(n):
            f.write(f"{i+1} {types[i]+1} {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
