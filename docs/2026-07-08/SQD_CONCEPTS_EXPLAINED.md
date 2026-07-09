# SQD run concepts, explained (Fe4S4 72q, Kingston UHF)

Two things from the live log / circuit, explained from the actual code:

1. What `fractional=64` means in the recovery log line.
2. Why the sampling circuit measures twice, skips some qubits, and applies stray
   `Y`-like gates — i.e. **reset error mitigation**.

---

## 1. `fractional=64` — the occupancy-based recovery signal

### The log line

```
[diag] recovery 1/1 input occ: sum_a=27.000 sum_b=27.000 fractional=64
```

This is printed at the top of every recovery step in `algorithms/sbd/sbd/sqd.py`
(lines ~493–506):

```python
occ_a_arr, occ_b_arr = np.asarray(avg_occ[0]), np.asarray(avg_occ[1])
n_frac = int(np.sum((occ_a_arr > 0.01) & (occ_a_arr < 0.99))) + int(
         np.sum((occ_b_arr > 0.01) & (occ_b_arr < 0.99)))
```

So:

- `avg_occ` is a **pair of per-orbital occupancy vectors** — one for α (spin-up),
  one for β (spin-down). Each vector has `norb = 36` entries (the Fe4S4 active
  space has 36 spatial orbitals). Each entry is the **average occupation of that
  spatial orbital**, a real number in `[0, 1]`.
- `sum_a = 27.000` / `sum_b = 27.000` — the α vector sums to 27 and the β vector
  sums to 27. That is exactly `nelec = (27, 27)`: 27 α electrons + 27 β electrons =
  54 electrons total. Occupancy must conserve particle number, so each spin's
  vector always sums to its electron count. This is a **sanity check** that the
  occupancies are physical.
- `fractional = 64` — the number of orbitals (counting α and β separately) whose
  occupancy is **strictly between 0.01 and 0.99**, i.e. **neither clearly empty
  nor clearly full**.

### The math

For each spin σ ∈ {α, β} and each spatial orbital `i`, let `n_{σ,i}` be its average
occupancy. Define an orbital as **fractional** if

```
0.01 < n_{σ,i} < 0.99
```

Then

```
fractional = #{ i : 0.01 < n_{α,i} < 0.99 } + #{ i : 0.01 < n_{β,i} < 0.99 }
```

The maximum possible value is `2 × norb = 2 × 36 = 72` (every α and β orbital
fractional). The minimum is `0` (every orbital either ~empty or ~full, i.e. a
single Slater determinant — pure Hartree–Fock, no correlation).

Here `fractional = 64` out of a possible 72.

### Why it matters

**Fractional occupancy is the fingerprint of electron correlation**, and it is
literally what configuration recovery uses to steer sampling (see the comment at
`sqd.py:493`: *"fractional values are what configuration recovery needs to bias
toward correlated configs"*).

- If a mean-field (HF) state were exact, each orbital would be either fully
  occupied (`n = 1`) or empty (`n = 0`) → `fractional = 0`. There would be nothing
  to recover.
- A **fractional** occupancy like `n_{α,17} = 0.6` means: across the sampled
  wavefunction, orbital 17 (spin-up) is occupied 60% of the time and empty 40% of
  the time. That can only happen if multiple determinants contribute — i.e. the
  true state is a **superposition** of configurations. That is correlation.
- The recovery step (`recover_configurations`) uses these fractional occupancies
  as **per-orbital Bernoulli probabilities** to repair noisy bitstrings: a bit on a
  near-integer orbital is trusted; a bit on a fractional orbital is re-drawn in a
  spin/particle-number-preserving way. So a **high `fractional` count = lots of
  orbitals carry correlation signal = recovery has a lot of useful structure to
  exploit.**

`fractional = 64 / 72` is a **strongly multireference** signature — expected for an
iron–sulfur cluster, which is exactly why SQD (not coupled cluster) is the right
tool here.

### The 27-α / 27-β example (this run, norb = 36)

Take a toy but faithful picture of the α occupancy vector (36 entries, summing to
27). Split the 36 orbitals into three groups:

```
core / clearly-occupied :  n ≈ 1.00   (e.g. 20 orbitals)   → NOT fractional
active / correlated     :  0.01<n<0.99 (e.g. 32 across α+β) → FRACTIONAL
virtual / clearly-empty :  n ≈ 0.00   (rest)               → NOT fractional
```

A concrete illustrative α vector (36 numbers, sum = 27):

```
index:  0 .. 19   →  1.00  each   (20 fully-occupied "core-like" orbitals)  = 20.00
index: 20 .. 29   →  0.55 0.62 0.48 0.71 0.40 0.66 0.52 0.59 0.44 0.63       ≈  5.60
index: 30 .. 35   →  0.20 0.15 0.10 0.05 0.08 0.02                          ≈  0.60
                                                              α sum ≈ 20.00 + 5.60 + 0.60 ... 
```

(The exact toy numbers are just to show the structure; the real vector sums to
exactly 27.000, which is why the log prints `sum_a=27.000`.)

- **Fully occupied** orbitals (`n ≈ 1.00`): the 27 electrons "want" to fill the
  lowest orbitals, so ~20 sit at essentially 1.0. These are **not** fractional.
- **Empty** orbitals (`n ≈ 0.00`): the high virtuals. **Not** fractional.
- **Fractional** orbitals: the ~10 α orbitals near the Fermi level with `n` between
  0.05 and 0.95. Those near the HOMO/LUMO region are where electrons are genuinely
  delocalized across configurations.

Count the α orbitals with `0.01 < n < 0.99`: in the toy vector that's indices
20–34 that fall in range — say **32** of them. Do the **same for β** (also ~32 by
symmetry, since this is a spin-singlet: `sum_b = 27.000` too). Then:

```
fractional = (α fractional) + (β fractional) ≈ 32 + 32 = 64
```

That is the `64` in the log: **out of 72 spin-orbital-occupancy slots (36 α + 36
β), 64 carry fractional — i.e. correlated — character.** Only 8 are "boringly"
integer (fully core or fully empty on both spins). For an Fe–S cluster that is a
very high correlation load, and it tells you the recovery step and a large `sqd_dim`
are both doing real work.

> Note on the α/β symmetry: because we ran **UHF** (unrestricted), α and β are
> tracked independently — `sum_a` and `sum_b` are reported separately and could in
> principle differ (that difference is the spin polarization `2·Sz = Na − Nb`).
> Here both are 27.000, so `2Sz = 0` — a **singlet**, as expected for this ground
> state. Later in the log you'll see the `spin density (n_a − n_b)` line confirming
> `sum = 0`.

---

## 2. Reset error mitigation — why the circuit measures twice and skips qubits

### What you saw in the QASM

The submitted OpenQASM has two classical registers:

```
creg meas[72];   // the real measurement of the 72 active qubits
creg test[72];   // the "reset test" register
```

…a first sparse layer of single-qubit gates on **just a few qubits**
(`q[14], q[32], q[38], q[46], q[87], q[97]` get `rz/rx/rz` = a `Y`/`√X`-type
rotation), a `measure ... -> test[k]` on every active qubit, then a `barrier`, then
(later, not in your excerpt) the Hartree–Fock prep + LUCJ ansatz + the final
`measure ... -> meas[k]`.

This is **reset error mitigation**, built in `algorithms/sbd/sbd/lucj.py`
(lines 143–154, 187):

```python
creg_test = ClassicalRegister(2 * norb, name="test")   # 72 bits
creg_meas = ClassicalRegister(2 * norb, name="meas")   # 72 bits
...
if use_reset_mitigation:
    circ.measure(qreg, creg_test)   # <-- measure BEFORE state prep
    circ.barrier()
circ.append(PrepareHartreeFockJW(...))   # then prepare HF
circ.append(UCJOp...JW(...))             # then the LUCJ ansatz
circ.measure(qreg, creg_meas)            # then the REAL measurement
```

### Why measure *before* preparing the state?

On real hardware a qubit is **not guaranteed to start in |0⟩**. Superconducting
qubits are "reset" between shots, but the reset is imperfect — a qubit can leak in
still excited (|1⟩) from the previous shot or from thermal population. If your
circuit assumes a clean |0⟩ start (which `PrepareHartreeFockJW` does), a dirty
starting qubit silently corrupts that shot.

Reset mitigation catches this:

1. **`measure(qreg, creg_test)` first** — immediately measure all qubits *before*
   doing anything. If the reset worked, every `test` bit should read **0**.
2. **`barrier()`** — stop the transpiler from moving gates across this point, so the
   test truly reflects the pre-circuit state.
3. Then prepare HF, apply the ansatz, and `measure(qreg, creg_meas)` for the real
   result.

### How the test register is used (post-selection)

Back in `sqd.py` (lines 340–361), after the job returns:

```python
meas_bits = pub_result[0].data.meas
if circuit_params.use_reset_mitigation:
    test_bits = pub_result[0].data.test
    kept = meas_bits.get_bitstrings(test_bits.bitcount() == 0)   # keep shots where test == all-zeros
    ...
    batch_array = BitArray.from_samples(kept, num_bits=meas_bits.num_bits)
```

**We keep only the shots whose entire `test` register read all-zeros** — i.e. shots
where every qubit was verified clean at the start. Any shot where `test` had a `1`
somewhere (a qubit that failed to reset) is **thrown away**. `test_bits.bitcount()`
counts the 1s; `== 0` means a perfectly-reset shot.

This is *post-selection*: it trades some shots (throughput) for higher-fidelity
retained shots. It composes with the other two mitigations we enable
(dynamical decoupling XY4 + measurement twirling) and the Hamming-weight
post-selection done later in recovery.

(There's a guard: on a simulator that doesn't model the reset ancilla, `test` can
reject *every* shot; the code detects `len(kept)==0` and falls back to the
unmitigated shots for that batch with a warning, so a mock backend can't crash the
flow.)

### "Why are some qubits not measured, and why the stray Y-gates?"

Two separate things are going on, both normal:

**(a) 156 qubits, only 72 measured.** `ibm_kingston` is a **156-qubit** device
(`qreg q[156]`), but our problem needs only `2 × norb = 2 × 36 = 72` qubits (36 for
α spin-orbitals + 36 for β, Jordan–Wigner mapping). The transpiler **lays out** our
72 logical qubits onto 72 specific physical qubits chosen for good connectivity/low
error (that's why the measured indices look scattered: `q[13], q[12], q[11], q[18],
q[31]...` — those are the physical qubits the router picked). The other 84 physical
qubits are **idle spectators** — never used, never measured. So "some qubits are not
being measured" just means they aren't part of our 72-qubit subgraph.

**(b) The sparse `rz(-pi/2) rx(pi/2) rz(pi/2)` layer on a handful of qubits.** That
three-gate combo is the hardware-native decomposition of a **basis-change / √Y-type
single-qubit rotation** (`rz·rx·rz` is the standard Euler decomposition; this
particular set of angles is a `Y`-ish rotation), and the lone `sx` gates
(`sx q[46], sx q[97]`) are √X. These appear on only a few qubits because they are
**measurement twirling** gates (we enabled `twirling.enable_measure = true`): before
each measurement, a randomly-chosen Pauli-like frame rotation is applied to a subset
of qubits and undone in classical post-processing. Twirling converts coherent
readout errors into (more benign) stochastic errors, averaged away across shots.
Different shots/batches get different random twirl layers, which is why only some
qubits carry them in any single circuit snapshot. Gate twirling is **off** for us (it
breaks on the fractional `rzz` gates of the LUCJ ansatz — see the error-mitigation
notes), so what you see is measurement twirling + DD only.

### One-line summary

- `meas[72]` = the real answer; `test[72]` = a pre-flight "was every qubit actually
  reset to |0⟩?" check. Keep only shots where `test` is all-zeros.
- 156→72: only our 72 JW spin-orbital qubits are measured; the rest of the chip is
  idle.
- The stray `rz/rx/rz` and `sx` on a few qubits = measurement-twirling frame
  rotations, not part of the physics — they're randomized per shot and undone in
  post-processing.

---

## Where this shows up in the pipeline

```
sample (Kingston)                     recovery (ROQUO GPU)
─────────────────                     ────────────────────
create_lucj_circuit  ── test/meas ──▶ recover_configurations   ← uses avg_occ
  reset mitigation                       (fractional occupancies steer the repair)
  DD (XY4)                             subsample → Davidson diag
  measure twirling                     carryover ← top dets by CI weight
  post-select test==0                  repeat n_recovery_steps
  merge 5×1M shots  ── pool.npz ──▶    best_energy
```

- Reset mitigation, DD, and twirling act at **sample** time (this doc §2).
- `fractional` is a **recovery**-time diagnostic on the occupancies feeding each
  repair pass (this doc §1).
