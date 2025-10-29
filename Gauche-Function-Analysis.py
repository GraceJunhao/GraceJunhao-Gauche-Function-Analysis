#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import sys

TPR = "Surfactant-C1-C18.tpr"
XTC = "Surfactant-C1-C18.xtc"
SKIP_FRAMES = 5000
ATOMS_PER_CHAIN = 18

def main():
    u = mda.Universe(TPR, XTC)
    all_atoms = u.atoms
    n_atoms = len(all_atoms)

    if n_atoms % ATOMS_PER_CHAIN != 0:
        print(f"Error: total atoms {n_atoms} is not divisible by {ATOMS_PER_CHAIN}!")
        sys.exit(1)

    n_chains = n_atoms // ATOMS_PER_CHAIN
    n_dihedrals = ATOMS_PER_CHAIN - 3  # 15
    gauche_counts = np.zeros(n_dihedrals, dtype=int)
    total_samples = 0

    print(f"Total atoms: {n_atoms}")
    print(f"Number of chains: {n_chains}")
    print("Analyzing dihedral distributions along the chain...")

    frame_count = 0
    for ts_index, ts in enumerate(u.trajectory):
        if ts_index < SKIP_FRAMES:
            continue
        frame_count += 1

        for i in range(n_chains):
            start = i * ATOMS_PER_CHAIN
            C = all_atoms[start:start + ATOMS_PER_CHAIN].positions  # (18, 3)

            for j in range(n_dihedrals):
                dih_rad = mda.lib.distances.calc_dihedrals(C[j], C[j+1], C[j+2], C[j+3])
                angle_deg = np.rad2deg(dih_rad)
                angle_deg = (angle_deg + 180) % 360 - 180  # [-180, 180]

                if (-90 <= angle_deg <= -30) or (30 <= angle_deg <= 90):
                    gauche_counts[j] += 1

    total_samples = frame_count * n_chains
    gauche_probs = gauche_counts / total_samples
    
    print("\n" + "="*60)
    print("Gauche probability per dihedral position (C_i to C_{i+3}):")
    print("="*60)
    for idx, prob in enumerate(gauche_probs):
        c_start = idx + 1
        c_end = idx + 4
        print(f"C{c_start:2d}鈥揅{c_end:2d} : {prob:6.4f}  ({prob*100:5.2f}%)")

    overall = np.mean(gauche_probs)
    print("-"*60)
    print(f"Overall gauche ratio: {overall:.4f} ({overall*100:.2f}%)")

    np.savetxt("gauche_per_position.txt", gauche_probs, fmt="%.6f", header="Gauche probability for C1-C4, C2-C5, ..., C15-C18")
    print("\n馃捑 Detailed data saved to: gauche_per_position.txt")

if __name__ == "__main__":
    main()
