#!/bin/bash
# Build gpu4pyscf 1.9.0 from source on ROQUO (aarch64 GB200, sm_100, CUDA 13).
# PyPI gpu4pyscf wheels are x86_64 only; cupy-cuda13x / cutensor-cu13 ship aarch64 wheels.
# Setup of docs/tutorials/run_bs_uno_avas_roquo.md. REVIEW BEFORE RUNNING. Run on a GPU node
# (interactive srun), not on the login node:
#   srun --account=$ACCT --partition=roquo --gres=gpu:1 --cpus-per-task=32 --time=02:00:00 --pty bash
#   (on reserved nodes add --reservation=<resv>; --gres alone gives 1 CPU for the -j 32 build)
#   bash build_gpu4pyscf_roquo.sh
# Env: MODULES (default "cuda/13.2 gcc/14"), VENV, SRC, TAG, ARCH, LIBXC. cmake comes from pip.
# Verified on ROQUO (2026-10-31): all 1.9.0 libraries + libxc compile for sm_100 with CUDA 13.2 /
# gcc 14 (ARCH=100-real). The GPU smoke test at the end is the remaining check; if it fails,
# use ENGINE=cpu with roquo/run_pdb.sh (same code path). Re-running is safe (venv, clone and
# cmake build dir are reused).
set -euo pipefail

VENV="${VENV:-$HOME/venvs/gpu4pyscf}"
SRC="${SRC:-$HOME/src}"
TAG="${TAG:-v1.9.0}"
ARCH="${ARCH:-100-real}"
LIBXC="${LIBXC:-ON}"   # only needed for UKS; set OFF if the libxc build fails

# Modules: CUDA (nvcc) and a C/C++ compiler only. ROQUO has no "cmake" module, so cmake comes
# from pip below (aarch64 wheel). Each module is loaded on its own so one unknown name does not
# abort the others; the tools are checked right after.
for m in ${MODULES:-cuda/13.2 gcc/14}; do
    module load "$m" 2>/dev/null || echo "note: module '$m' not available; continuing"
done
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found (module load cuda/13.2)"; exit 1; }
command -v gcc >/dev/null || { echo "ERROR: gcc not found"; exit 1; }
echo "nvcc $(nvcc --version | grep -o 'release [0-9.]*'), gcc $(gcc -dumpversion)"

python3.12 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "pyscf==2.14.0" cupy-cuda13x cutensor-cu13 geometric basis-set-exchange h5py
pip install "cmake>=3.26"   # build tool only (CUDA 13 / sm_100 aware); no module needed
cmake --version | head -1

mkdir -p "$SRC"
cd "$SRC"
[[ -d gpu4pyscf ]] || git clone --branch "$TAG" --depth 1 https://github.com/pyscf/gpu4pyscf
cmake -S gpu4pyscf/gpu4pyscf/lib -B gpu4pyscf/build \
    -DCUDA_ARCHITECTURES="$ARCH" -DBUILD_LIBXC="$LIBXC" -DCMAKE_BUILD_TYPE=Release
cmake --build gpu4pyscf/build -j 32

# Make the source tree importable. Not `pip install -e`: gpu4pyscf 1.9.0's setup.py defines
# package_name only for sdist or GPU4PYSCF_BUILD_CUDA_WHEEL builds (NameError otherwise), and
# would rebuild the libraries. The cmake step above already put every lib*.so (and libxc in
# lib/deps/lib) inside the source tree, where gpu4pyscf loads them from.
ls gpu4pyscf/gpu4pyscf/lib/libgvhf_rys.so gpu4pyscf/gpu4pyscf/lib/libgint.so >/dev/null \
    || { echo "ERROR: gpu4pyscf libraries were not built"; exit 1; }
pip install packaging   # gpu4pyscf runtime dependency (pip resolution is skipped here)
site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$SRC/gpu4pyscf" > "$site/gpu4pyscf-src.pth"
python -c "import gpu4pyscf; print('gpu4pyscf', gpu4pyscf.__version__, 'from', gpu4pyscf.__file__)"

# smoke test: small UHF on the GPU vs CPU
python - <<'EOF'
import cupy
from pyscf import gto, scf
from gpu4pyscf import scf as gscf
print("devices", cupy.cuda.runtime.getDeviceCount(), cupy.cuda.Device(0).compute_capability)
mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="def2-svp", charge=1, spin=1)
e_gpu = gscf.UHF(mol).density_fit().kernel()
e_cpu = scf.UHF(mol).density_fit().kernel()
print(f"UHF GPU {e_gpu:.10f} CPU {e_cpu:.10f} diff {abs(e_gpu - e_cpu):.1e}")
assert abs(e_gpu - e_cpu) < 1e-7
EOF
echo "OK: next (login node): bash roquo/run_pdb.sh check"
