#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory to avoid accidental cleanup in cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"
CCCOM="${CCCOM:-mpic++}"
CCFLAGS="${CCFLAGS:--std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST}"
SYSLIB="${SYSLIB:--lblas -llapack}"

# UHF=1 builds the unrestricted GPU binary diag-gpu_uhf (adds -D_UHF on top of -DSBD_THRUST,
# which selects the _UHF path in integrals_thrust.h); otherwise the restricted diag-gpu.
if [ "${UHF:-0}" = "1" ]; then
    CCFLAGS="$CCFLAGS -D_UHF"
    OUTBIN="diag-gpu_uhf"
else
    OUTBIN="diag-gpu"
fi

# MPI=1 builds the multi-GPU binary <name>-mpi-hpcx250 (one MPI rank per GPU, launched with
# `srun --mpi=pmix`): same main.cc and SBD checkout, plus NCCL collectives, host-staged MPI (this
# MPI is not CUDA-aware) and the rank distributions, the options of the earlier cmake recipe
# (build_sbd_mpi_uhf_hpcx250.sh) except cuBLAS. Load hpcx/2.50 AFTER nvhpc so its mpic++
# (OpenMPI 5, PMIx) comes first on PATH.
# No -DSBD_USE_CUBLAS: on upstream main (2026-10) that path does not compile (DavidsonCublas
# passes device_vectors to Determine_kernel; subspace_thrust.h and cublas_utility.h both define
# sbd::raw_ptr). Without it Davidson runs DavidsonImpl, which still uses NCCL across ranks.
# CUBLAS=1 adds it back once upstream fixes it.
if [ "${MPI:-0}" = "1" ]; then
    command -v nvc++ >/dev/null 2>&1 || {
        echo "MPI=1 needs nvc++ on PATH (module load cuda/13.2 nvhpc/26.5)" >&2; exit 1; }
    NV="${NVHPC_ROOT:-$(dirname "$(dirname "$(readlink -f "$(command -v nvc++)")")")}"
    NV="${NV%/compilers}"
    NCCL_H="$(find "$NV" -name nccl.h -path '*nccl*' 2>/dev/null | head -1)"
    NCCL_SO="$(find "$NV" -name 'libnccl.so' -path '*comm_libs*' 2>/dev/null | head -1)"
    if [[ -z "$NCCL_H" || -z "$NCCL_SO" ]]; then
        echo "MPI=1: NCCL not found under $NV (nccl.h: '${NCCL_H}', libnccl.so: '${NCCL_SO}')" >&2
        exit 1
    fi
    echo "NCCL: $(dirname "$NCCL_H") / $(dirname "$NCCL_SO")"
    CCFLAGS="$CCFLAGS -DSBD_USE_NCCL -DSBD_NON_CUDA_AWARE_MPI"
    CCFLAGS="$CCFLAGS -DSBD_THRUST_SAFE_MPI_ALLREDUCE -DSBD_USE_RANK_DISTRIBUTION"
    CCFLAGS="$CCFLAGS -DSBD_USE_BLOCK_RANK_DISTRIBUTION -I$(dirname "$NCCL_H")"
    SYSLIB="$SYSLIB -L$(dirname "$NCCL_SO") -Wl,-rpath,$(dirname "$NCCL_SO") -lnccl"
    if [ "${CUBLAS:-0}" = "1" ]; then
        CCFLAGS="$CCFLAGS -DSBD_USE_CUBLAS"
        SYSLIB="$SYSLIB -cudalib=cublas"
    fi
    OUTBIN="${OUTBIN}-mpi-hpcx250"
fi

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load the Miyabi-G compiler/MPI environment first." >&2
    exit 1
fi

# The default flags (-mp -cuda -fast -gpu=...) are NVHPC flags: the MPI wrapper must call nvc++.
# On Miyabi-G it already does. On ROQUO the first mpic++ on PATH (e.g. module hpcx/2.50) wraps
# g++ by default, so point the OpenMPI wrapper at nvc++, as build_sbd_mpi_uhf_hpcx250.sh does.
if [[ "$CCFLAGS" == *"-gpu="* || "$CCFLAGS" == *"-cuda"* ]]; then
    if command -v nvc++ >/dev/null 2>&1; then
        export OMPI_CXX="${OMPI_CXX:-nvc++}"
        export OMPI_CC="${OMPI_CC:-nvc}"
    fi
    backend="$("$CCCOM" -show 2>/dev/null | awk '{print $1}')" || backend=""
    if [[ -n "$backend" && "$(basename "$backend")" != nvc++* ]]; then
        echo "'$CCCOM' wraps '$backend', but CCFLAGS are NVHPC flags (nvc++ needed)." >&2
        echo "Load the NVIDIA HPC SDK so nvc++ is on PATH, e.g. on ROQUO:" >&2
        echo "  module load cuda/13.2 nvhpc/26.5" >&2
        echo "  export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5/compilers/bin:\$PATH" >&2
        exit 1
    fi
    echo "MPI wrapper: $CCCOM -> ${backend:-unknown}"
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists: $SBD_DIR"
fi
echo "SBD source: $(git -C "$SBD_DIR" log -1 --format='%h %ad %s' --date=short 2>/dev/null || echo '?')"

# The multi-GPU path needs three upstream pieces (all on r-ccs-cms/sbd main since 2026-08-28).
# An older checkout lacks them: update it (git -C native/sbd pull) rather than build it.
if [ "${MPI:-0}" = "1" ]; then
    inc="$SBD_DIR/include/sbd"
    missing=""
    grep -q "h_size == 1" "$inc/chemistry/basic/davidson_thrust.h" 2>/dev/null \
        || missing+=" PR79 (davidson_thrust.h: GetTotalD deadlock with a 1-rank h_comm);"
    grep -q "cr.reserve(2)" "$inc/chemistry/tpb/helper.h" 2>/dev/null \
        || missing+=" PR84 (helper.h: GenerateExcitation OpenMP livelock);"
    [[ -f "$inc/framework/nccl_utility.h" ]] || missing+=" framework/nccl_utility.h;"
    if [[ -n "$missing" ]]; then
        echo "MPI=1: $SBD_DIR lacks:$missing" >&2
        echo "  update it: git -C $SBD_DIR pull   (then build again)" >&2
        exit 1
    fi
    echo "upstream fixes present: PR79, PR84, NCCL helpers"
fi

# Clean previous build artifacts for this target only.
rm -f "$SCRIPT_DIR"/*.o "$SCRIPT_DIR/$OUTBIN"

# nvc++ prints many harmless warnings for the upstream SBD headers: unused variables, and
# "CUDA C++ compilation is unsupported in nvc++" for the <<<>>> kernels (nvc++ -cuda builds
# them; it only recommends nvcc). Silence those so real errors stand out; WARNFLAGS="" shows all.
if [[ -z "${WARNFLAGS+x}" ]]; then
    WARNFLAGS=""
    if [[ "$("$CCCOM" -show 2>/dev/null | awk '{print $1}')" == *nvc++* ]]; then
        WARNFLAGS="--diag_suppress declared_but_not_referenced,set_but_not_used,cuda_compile"
    fi
fi

# Compile and link
echo "$CCCOM $CCFLAGS $WARNFLAGS -c main.cc -o main.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS $WARNFLAGS -c main.cc -o main.o -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o $OUTBIN main.o"
$CCCOM $CCFLAGS $SYSLIB -o "$OUTBIN" main.o

[[ -x "$SCRIPT_DIR/$OUTBIN" ]] || { echo "Build FAILED: $SCRIPT_DIR/$OUTBIN not created" >&2; exit 1; }
if [ "${MPI:-0}" = "1" ]; then
    echo "--- linked MPI / NCCL / cuBLAS / CUDA libraries ---"
    ldd "$SCRIPT_DIR/$OUTBIN" | grep -iE "libmpi|nccl|cublas|cudart|pmix|not found" || true
    if ldd "$SCRIPT_DIR/$OUTBIN" | grep -q "not found"; then
        echo "Build FAILED: $SCRIPT_DIR/$OUTBIN has unresolved libraries (above)" >&2; exit 1
    fi
fi
echo "Build completed: $SCRIPT_DIR/$OUTBIN"
