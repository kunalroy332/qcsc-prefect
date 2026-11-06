"""Active-orbital labels: polar (Löwdin-matching) rotation onto the targets, or Boys + argmax."""

import numpy as np
from pyscf import lo
from scipy import linalg


def polar_rotation(c_act, targets, ovlp, target_labels):
    """Rotate ``c_act`` to the closest orthonormal set to ``targets``; orbital j gets label j.

    Returns (rotated orbitals, labels, weights |<φ_j|t_j>|²). Labels are unique by construction.
    """
    u, _, vt = linalg.svd(c_act.T @ ovlp @ targets)
    rotated = c_act @ u @ vt
    weights = np.diag(rotated.T @ ovlp @ targets) ** 2
    return rotated, list(target_labels), weights


def boys_rotation(mol, c_act, targets, ovlp, target_labels):
    """Boys localisation, label = target with the largest overlap; error on duplicates."""
    rotated = lo.Boys(mol, c_act).kernel()
    w = (rotated.T @ ovlp @ targets) ** 2
    idx = np.argmax(w, axis=1)
    labels = [target_labels[j] for j in idx]
    if len(set(labels)) != len(labels):
        dup = sorted({lab for lab in labels if labels.count(lab) > 1})
        raise RuntimeError(f"Boys labelling gives duplicate labels {dup}; use labels = 'polar'")
    # reorder to target order
    order = np.argsort(idx)
    return rotated[:, order], [labels[k] for k in order], w[order, idx[order]]


def check_weights(labels, weights, fe_warn=0.85, s_warn=0.3):
    out = []
    for lab, w in zip(labels, weights):
        el = lab.split("_")[1]
        if (el == "Fe" and w < fe_warn) or (el == "S" and w < s_warn):
            out.append(f"{lab} weight {w:.3f}")
    return out
