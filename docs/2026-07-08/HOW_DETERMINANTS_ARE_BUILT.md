# From 1.9M Kingston bitstrings to alpha/beta determinants — step by step

This traces exactly how the SQD subspace (the alpha determinants and beta determinants the Davidson
solver diagonalizes) is built from the raw quantum-sampled bitstrings, for the Fe4S4 72-qubit case
(norb = 36 spatial orbitals, 27 alpha electrons, 27 beta electrons). Grounded in
`algorithms/sbd/sbd/sqd.py` (`subsample_open_shell` / `_subsample_one_spin`).

---

## 0. What one "bitstring" is

The circuit measures **72 qubits = 2 × 36**. Under the Jordan–Wigner mapping, each spin-orbital is
one qubit: the first 36 qubits are the **beta** spin-orbitals, the next 36 are the **alpha**
spin-orbitals. So one measured shot is a 72-bit string:

```
        beta half (36 bits)            alpha half (36 bits)
   [ b35 b34 ... b1 b0 ]          [ a35 a34 ... a1 a0 ]
     ^ beta occupations              ^ alpha occupations
```

A `1` means that spin-orbital is occupied in that shot, `0` means empty. The Kingston run gave
**1,955,531 such 72-bit shots** (after 5×10⁶ shots, reset-mitigation, and post-selection). Each is
one row of `bitstring_matrix`; column `i` (0..35) is a beta orbital, column `36+i` is the matching
alpha orbital (`sqd.py:1094-1095`).

## 1. Split each 72-bit shot into an alpha half and a beta half

A determinant is a *product* of an alpha part and a beta part — they are independent. So the first
thing the code does is **cut every 72-bit shot down the middle** into two 36-bit halves
(`sqd.py:1104-1106`):

```
ci_strs_b = the left  36 bits  (beta  occupations)   ->  a 36-bit integer
ci_strs_a = the right 36 bits  (alpha occupations)   ->  a 36-bit integer
```

Concretely, for each shot the loop packs the 36 bits into one integer:
`ci_strs_a += bit[36+i] * 2**(35-i)`. So orbital `i` sits at bit position `35-i`. Example (toy,
norb small): if the alpha half reads `000...0111` (lowest 3 orbitals occupied) that becomes the
integer with its 3 lowest bits set.

**Key point:** after this split we have, for the 1.9M shots, **1.9M alpha-half integers and 1.9M
beta-half integers**. These are processed *separately* from here on — this is what "unrestricted /
open-shell" means: alpha and beta get their own independent determinant pools.

## 2. Deduplicate each spin's pool and accumulate probabilities

Many of the 1.9M shots repeat the same alpha (or beta) occupation pattern — that repetition is the
quantum probability. For each spin independently (`_deduplicate_and_accumurate_probs`):

- collapse identical spin-half integers to **unique** ones,
- add up how often each occurred → that sum is its **probability weight**.

So the 1.9M alpha halves collapse to, say, a few hundred thousand *unique* alpha determinants, each
with a probability. Same for beta. (In the Fe4S4 run each spin had ~1.9M unique after dedup because
the sample was very diffuse — lots of distinct noisy configs.)

## 3. Post-selection: keep only the right electron count

A physical alpha determinant for this system must have **exactly 27 alpha electrons** (27 bits set);
beta must have exactly 27. Noise flips bits, so many sampled halves have 26 or 28 electrons — those
are unphysical and are thrown out (`postselect_bitstrings`, Hamming-weight = num_elec). What
survives is a pool of alpha determinants all with popcount 27, and beta determinants all with
popcount 27. (This is also what fixes Sz = (Na−Nb)/2 exactly.)

## 4. Build each spin's determinant LIST for the subspace (`_subsample_one_spin`)

Now, per spin, assemble the actual list of determinants the solver will use. The list is built in a
fixed priority order (`sqd.py` `_merge_with_seed`):

```
[ Hartree-Fock ] + [ carryover ] + [ CISD seed ] + [ sampled ]
```

with a total length of `floor(sqrt(sqd_dim))` (the CI matrix is that many alpha × that many beta).

1. **Hartree-Fock at index 0** (required by the solver): the integer with the lowest 27 bits set,
   `(1<<27)-1` — the 27 lowest orbitals occupied.
2. **Carryover**: the highest-weight determinants the *previous* recovery step's diagonalization
   found important (top `carryover_ratio` by CI coefficient). These are forced back in so the loop
   is self-consistent.
3. **CISD seed** (when `seed_cisd>0`): classically-generated determinants — the HF determinant with
   1 electron moved (singles) or 2 electrons moved (doubles) from an occupied to a virtual orbital.
   These are the same for alpha and beta shape-wise but injected into each spin's list. They are
   forced in so the low-excitation configs noise dropped are always present. (`seed_budget_frac`
   controls how much of the budget they may take.)
4. **Sampled determinants**: the remaining slots are filled by drawing from the deduplicated,
   post-selected sampled pool **weighted by probability** (`rng.choice(..., p=probs)`), excluding
   any already added as HF/carryover/seed. These are mostly *higher-excitation* configs the device
   sampled.

The result is `ci_strs_a` (a list of ~N alpha determinant integers) and `ci_strs_b` (a list of ~N
beta determinant integers), each with HF at index 0.

## 5. The subspace the solver diagonalizes

The Davidson solver takes the **outer product**: every alpha determinant paired with every beta
determinant defines one many-body basis state |alpha> ⊗ |beta>. So `N` alpha × `N` beta gives an
`N²`-dimensional subspace — that is the `net dim` in the logs (e.g. `N=3000 -> net dim = 9,000,000`;
`N=12,922 -> net dim ≈ 1.67e8`). The Hamiltonian is projected onto this subspace and diagonalized;
its lowest eigenvalue is the SQD energy for that step.

---

## Worked micro-example (norb = 5, 3 alpha + 3 beta electrons, for intuition)

Say a shot measured as (beta | alpha) = `01110 | 00111`:
- **beta half** `01110` → orbitals {1,2,3} occupied → 3 beta electrons ✓ (a valid beta determinant)
- **alpha half** `00111` → orbitals {0,1,2} occupied → 3 alpha electrons ✓

The alpha determinant `00111` is the alpha Hartree–Fock (lowest 3 orbitals). The beta determinant
`01110` is a **single excitation** of beta-HF (`00111`): one electron moved from orbital 0 → orbital
3. If instead a shot gave alpha `01011` that's a single excitation on alpha (orbital 2 → orbital 3),
and it becomes one entry in `ci_strs_a`.

- **Singles** (what CISD level 1 injects): move ONE electron occ→virt, e.g. `00111 -> 01011`.
- **Doubles** (CISD level 2): move TWO, e.g. `00111 -> 11001`.

Each such pattern is an integer in the per-spin list. The solver then diagonalizes over all
alpha×beta pairs of these integers.

## Why alpha and beta are handled separately (UHF vs RHF)

- **UHF / open-shell** (`subsample_open_shell`): alpha and beta pools are deduplicated and built
  **independently** — the alpha list and beta list can differ, and each carries its own HF,
  carryover, and CISD seed. This lets the two spins polarize differently (the spin density
  n_alpha − n_beta the logs report).
- **RHF / closed-shell** (`subsample_close_shell`): the two halves are **merged and averaged** into a
  single list reused for both spins — appropriate when the state is spin-restricted.

For Fe4S4 we ran UHF, so the logs show `alpha dets:` and `beta dets:` separately (they had the same
counts here because it's a singlet: 27 alpha, 27 beta, symmetric).

## Where the numbers in the logs come from

```
[diag] recovery 1/12 alpha dets: n=12922 unique=12922 HF=1 singles+doubles=12879 higher(>2)=42 max_exc=8
```
- `n=12922` — length of the alpha determinant list (= √sqd_dim after HF/carryover/seed/sampled).
- `HF=1` — the Hartree–Fock determinant (index 0).
- `singles+doubles=12879` — determinants that are ≤2 excitations from alpha-HF (here, the full CISD
  seed).
- `higher(>2)=42` — determinants ≥3 excitations (here, the few sampled configs that fit).
- `max_exc=8` — the most-excited determinant in the list is 8 electrons away from HF.

The beta line is the analogous count for the beta list. The product `n_alpha × n_beta` is the
diagonalized subspace dimension.
