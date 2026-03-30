"""Harmonic bond force reference implementation (host-side).

F = -k * (r - r0) * r_hat for each bonded pair.
Bond list is static (lipid connectivity), ~320 bonds for 80 lipids.

TODO: Move to a TT-Lang kernel using pre-sorted per-atom bond layout
so bonds can run on-device inside the traced step loop. Approach: group
bonds by atom, stream per-atom with ttl scatter or pipe-based accumulation.
Host-side version kept here as reference and for validation.
"""
import numpy as np
import torch
import ttnn


def compute_bond_forces_host(positions, bonds, k, r0, box_length):
    """Compute harmonic bond forces on the host.

    Args:
        positions: (N, 3) atom positions
        bonds: (n_bonds, 2) int pairs of bonded atom indices
        k: spring constant
        r0: equilibrium distance
        box_length: simulation box length

    Returns:
        forces: (N, 3) bond forces
    """
    forces = np.zeros_like(positions)
    if len(bonds) == 0:
        return forces

    i_idx = bonds[:, 0]
    j_idx = bonds[:, 1]
    dr = positions[i_idx] - positions[j_idx]
    dr -= box_length * np.floor(dr / box_length + 0.5)
    r = np.linalg.norm(dr, axis=1, keepdims=True)
    r = np.maximum(r, 1e-10)  # avoid division by zero
    f_mag = -k * (r - r0)
    f_vec = f_mag * dr / r

    # Accumulate forces
    np.add.at(forces, i_idx, f_vec.reshape(-1, 3))
    np.add.at(forces, j_idx, -f_vec.reshape(-1, 3))
    return forces


def test_bond_forces():
    """Verify bond force computation against simple reference."""
    np.random.seed(42)

    # Simple test: 2 atoms connected by a bond
    positions = np.array([[1.0, 1.0, 1.0], [2.5, 1.0, 1.0]])
    bonds = np.array([[0, 1]])
    k = 100.0
    r0 = 1.0
    box_length = 10.0

    forces = compute_bond_forces_host(positions, bonds, k, r0, box_length)

    # Reference: dr = (-1.5, 0, 0), r = 1.5, f_mag = -100*(1.5-1.0) = -50
    # f_vec = -50 * (-1.5, 0, 0) / 1.5 = (50, 0, 0)
    # force on atom 0: (50, 0, 0), force on atom 1: (-50, 0, 0)
    expected_0 = np.array([50.0, 0.0, 0.0])
    expected_1 = np.array([-50.0, 0.0, 0.0])

    assert np.allclose(forces[0], expected_0, atol=1e-6), f"Atom 0: {forces[0]} != {expected_0}"
    assert np.allclose(forces[1], expected_1, atol=1e-6), f"Atom 1: {forces[1]} != {expected_1}"
    print("Bond force test 1 (2 atoms): PASS")

    # Test with PBC: atoms across boundary
    positions2 = np.array([[0.5, 1.0, 1.0], [9.5, 1.0, 1.0]])
    forces2 = compute_bond_forces_host(positions2, bonds, k, r0, box_length)
    # PBC distance: 0.5 - 9.5 = -9.0, wrapped = 1.0. r = 1.0 = r0, so force = 0
    assert np.allclose(forces2, 0.0, atol=1e-6), f"PBC test: {forces2}"
    print("Bond force test 2 (PBC, r=r0): PASS")

    # Test lipid chain: 5 atoms in a line
    positions3 = np.array([
        [1.0, 1.0, 1.0],  # head
        [2.0, 1.0, 1.0],  # tail1
        [3.0, 1.0, 1.0],  # tail2
        [4.0, 1.0, 1.0],  # tail3
        [5.0, 1.0, 1.0],  # tail4
    ])
    bonds3 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    forces3 = compute_bond_forces_host(positions3, bonds3, k, r0, box_length)
    # All bonds at r0=1.0, so all forces should be zero
    assert np.allclose(forces3, 0.0, atol=1e-6), f"Chain test: {forces3}"
    print("Bond force test 3 (chain at equilibrium): PASS")

    # Test stretched chain
    positions4 = np.array([
        [1.0, 1.0, 1.0],
        [2.5, 1.0, 1.0],  # stretched by 0.5
        [3.5, 1.0, 1.0],  # at equilibrium from tail1
        [5.0, 1.0, 1.0],  # stretched by 0.5
        [6.0, 1.0, 1.0],  # at equilibrium from tail3
    ])
    forces4 = compute_bond_forces_host(positions4, bonds3, k, r0, box_length)
    print(f"Stretched chain forces:\n{forces4}")
    # Head pulled toward tail1: positive x force
    assert forces4[0, 0] > 0, "Head should be pulled toward tail1"
    print("Bond force test 4 (stretched chain): PASS")

    print("\nAll bond force tests PASS")


if __name__ == "__main__":
    test_bond_forces()
