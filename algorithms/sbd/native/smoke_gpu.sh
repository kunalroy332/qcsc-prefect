#!/bin/bash
#SBATCH --job-name=sbd-smoke
#SBATCH --partition=roquo
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=smoke/sbd-smoke-%j.log
# Smoke test of the GPU SBD solvers, as a batch job on ONE node:
#   g1  the 1-GPU binary diag-gpu_uhf, run directly (no srun), as one solve of a 1-GPU task;
#   g4  the multi-GPU binary diag-gpu_uhf-mpi-hpcx250 with 4 ranks on a 2 x 2 grid, launched as a
#       step of this job with `srun --mpi=pmix -n4` (no step inside a step).
# Both diagonalize the same small N2 subspace (300 x 300 determinants, inputs written by the flow's
# own writers, smoke_inputs.py). Each energy must equal the exact lowest eigenvalue of that space,
# E_REF, computed with PySCF (selected_ci.kernel_fixed_space; it cannot be lower).
#
# Submit from algorithms/sbd/native on the login node (account / reservation on the command line):
#   sbatch --account=$ACCT ${RESV:+--reservation=$RESV} smoke_gpu.sh
#   GPUS=1 sbatch ... smoke_gpu.sh        # only the 1-GPU binary
# Output: smoke/sbd-smoke-<jobid>.log (the whole report) and smoke/<jobid>/g{1,4}/out.txt.
set -uo pipefail
N="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[[ -f "$N/smoke_inputs.py" ]] || N="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$N/../../.." && pwd)"
cd "$N"
[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "SMOKE FAILED: submit it with sbatch (see the header of this script)"; exit 2; }
W="$N/smoke/$SLURM_JOB_ID"
mkdir -p "$W"

NV=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5
module load cuda/13.2 nvhpc/26.5 2>/dev/null || true
module load hpcx/2.50 2>/dev/null || true
export PATH="$NV/compilers/bin:$PATH"
export LD_LIBRARY_PATH="$NV/compilers/lib:$NV/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"
PY="${SBD_PYTHON:-$REPO/algorithms/sbd/.venv/bin/python}"
CPT="${SLURM_CPUS_PER_TASK:-36}"
# Exact lowest eigenvalue of the 300 x 300 subspace of data/fcidump_N2_MO.txt (ECORE = 0) with the
# sorted strings of smoke_inputs.py: PySCF 2.14 selected_ci.kernel_fixed_space, tol 1e-12; equal
# to qiskit-addon-sqd solve_fermion and to a dense diagonalization (checked on 30 x 30).
# (-134.7744464944 was a wrong reference: selected_ci needs sorted strings.)
E_REF="-134.8967813586"
TOL="1e-6"
echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname -s): $(git -C "$REPO" log -1 --oneline)"
echo "allocation: ${SLURM_NTASKS:-?} tasks x $CPT CPUs, GPUs ${SLURM_GPUS_PER_NODE:-?}, E_REF $E_REF"

ARGS=(--task_comm_size 1 --block 10 --iteration 5 --tolerance 1e-8 --carryover_ratio 0.5
      --carryover_type 1 --rdm 0 --adetfile AlphaDets.bin --bdetfile BetaDets.bin)

run() {  # <name> <ranks> <binary> <adet> <bdet>
    local d="$W/$1" n=$2 bin=$3 rc=0
    echo "== $1: $n rank(s), grid 1 x $4 x $5, $(basename "$bin")"
    [[ -x "$bin" ]] || { echo "   binary missing: $bin"; return 1; }
    file -L "$bin" | sed 's/,.*//'
    "$PY" smoke_inputs.py "$d" 300 || return 1
    if (( n == 1 )); then  # one solve of a 1-GPU task: the binary itself, on GPU 0
        (cd "$d" && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=$CPT timeout 600 \
            "$bin" --adet_comm_size 1 --bdet_comm_size 1 "${ARGS[@]}" > out.txt 2>&1) || rc=$?
    else                   # a step of this job: n ranks, one GPU each (rank % #GPUs)
        (cd "$d" && OMP_NUM_THREADS=$CPT timeout 600 srun --input=none --mpi=pmix \
            -N1 -n "$n" --cpus-per-task="$CPT" --gpus-per-node="$n" \
            "$bin" --adet_comm_size "$4" --bdet_comm_size "$5" "${ARGS[@]}" > out.txt 2>&1) || rc=$?
    fi
    echo "   exit $rc"
    grep -h '^\[gpu\]' "$d/out.txt" | sort | sed 's/^/   /'
    grep -hiE 'Loaded|Davidson energy|error' "$d/out.txt" | tail -6 | sed 's/^/   /'
    [[ $rc == 0 && -s "$d/davidson_energy.txt" ]]
}

check() {  # <name>: energy vs E_REF
    local e
    e="$(cat "$W/$1/davidson_energy.txt" 2>/dev/null)"
    [[ -n "$e" ]] || { echo "$1: no energy"; return 1; }
    "$PY" - "$1" "$e" "$E_REF" "$TOL" <<'EOF'
import sys
name, e, ref, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
d = e - ref
verdict = "OK" if abs(d) < tol else ("BELOW the exact minimum: wrong space or Hamiltonian"
                                     if d < -tol else "above the exact minimum")
print(f"{name}: E = {e:.10f}  E - E_REF = {d:+.2e}  {verdict}")
sys.exit(0 if abs(d) < tol else 1)
EOF
}

ok=0
run g1 1 "$N/diag-gpu_uhf" 1 1 || ok=1
if [[ "${GPUS:-4}" != 1 ]]; then
    run g4 4 "$N/diag-gpu_uhf-mpi-hpcx250" 2 2 || ok=1
fi
echo "== energies (E_REF = $E_REF, tolerance $TOL Eh)"
check g1 || ok=1
[[ "${GPUS:-4}" == 1 ]] || check g4 || ok=1
if [[ $ok == 0 ]]; then echo "SMOKE OK"; else echo "SMOKE FAILED (see above)"; fi
echo "work dirs: $W"
exit $ok
