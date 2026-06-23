# Custom Program for Selected Basis Diagonalization

This submodule provides a custom C++ implementation for configuration recovery,
adapted from the [selected basis diagonalization sample](https://github.com/r-ccs-cms/sbd/tree/main/samples/selected_basis_diagonalization) in the SBD project.
Refer to the upstream repository for detailed information about the original software.

## Output Files

The modified program generates the following output files:
- `davidson_energy.txt`: Final energy computed by the Davidson solver
- `occ_a.txt`: Average occupancies of alpha orbitals
- `occ_b.txt`: Average occupancies of beta orbitals
- `carryover.bin`: Carryover bitstrings in `uint8` format with big-endian ordering

These files are consumed by the Prefect integration block to construct a corresponding Python data class.

## Building the Executable on Miyabi

To build the executable, run the provided build script:

```bash
bash ./build_sbd.sh
```

> [!NOTE]
> This build script is adapted for the Miyabi-C environment.
> Build in other environments may require modification of the compiler command and flags.

## Building the GPU Executable on Miyabi-G

For Miyabi-G, you can build `diag-gpu` with the dedicated shell script:

```bash
./build_sbd_gpu.sh
```

This script compiles `main.cc` directly, so a separate `CMakeLists.txt` is not required.
It defaults to a local checkout under:

```text
algorithms/sbd/native/sbd
```

and produces:

```text
algorithms/sbd/native/diag-gpu
```

> [!NOTE]
> The default compiler is `mpic++`, assuming it is backed by the Miyabi-G NVHPC toolchain.
> If the local `sbd` checkout does not exist yet, the script clones `https://github.com/r-ccs-cms/sbd.git`.
> If your site uses a different wrapper, source tree, or BLAS/LAPACK link flags, override them via `CCCOM`, `CCFLAGS`, `SYSLIB`, `SBD_DIR`, or `SBD_REPO_URL`.

## Building the Executable on Fugaku

For Fugaku, use the dedicated script:

```bash
bash ./build_sbd_fugaku.sh
```

> [!NOTE]
> Ensure a Fugaku MPI C++ compiler is available in `PATH` (for example `mpiFCCpx`).
> The script defaults to `mpiFCCpx` with Fugaku-oriented flags and `-SSL2`.
> You can override `CCCOM`, `CCFLAGS`, and `SYSLIB` via environment variables if your site requires different settings.

Upon successful compilation, an executable named `diag` will be created in this directory:

```text
algorithms/sbd/native/diag
```

Set the absolute path to this executable in the SBD block configuration file:

```toml
# algorithms/sbd/sbd_blocks.toml
sbd_executable = "/abs/path/to/qcsc-prefect/algorithms/sbd/native/diag"
```

## Building the UHF (open-shell) Executable

All three build scripts accept `UHF=1`, which appends `-D_UHF` and emits a separate `diag_uhf`
(`diag-gpu_uhf` for the GPU script) binary, leaving the restricted `diag` target untouched. The
`-D_UHF` build switches the integral container and FCIDUMP parser to the spin-resolved
(interleaved spin-orbital) layout and reads a beta determinant list distinct from alpha.

Build both restricted and unrestricted binaries side by side, e.g. on Fugaku:

```bash
bash ./build_sbd_fugaku.sh           # -> diag      (restricted, unchanged)
UHF=1 bash ./build_sbd_fugaku.sh     # -> diag_uhf  (unrestricted / open-shell)
```

The UHF binary additionally accepts `--adetfile` / `--bdetfile` to load separate alpha and beta
determinant files and writes a `carryover_b.bin` alongside `carryover.bin`. Without `--bdetfile`
it falls back to `bdet = adet` (identical to the restricted binary). Register the UHF path under a
distinct executable key (`sbd_diag_uhf`) so RHF and UHF blocks can coexist; the `SBDSolverJob`
selects it when `method="uhf"`.

### Optional environment overrides

You can override upstream repository location:

```bash
SBD_REPO_URL="https://github.com/r-ccs-cms/sbd.git" \
SBD_DIR="/path/to/local/sbd" \
bash ./build_sbd.sh
```

For Fugaku build settings:

```bash
CCCOM="mpiFCCpx" \
CCFLAGS="-Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp" \
SYSLIB="-SSL2" \
bash ./build_sbd_fugaku.sh
```

For Miyabi-G build settings:

```bash
SBD_REPO_URL="https://github.com/r-ccs-cms/sbd.git" \
SBD_DIR="/path/to/local/sbd" \
CCCOM="mpic++" \
CCFLAGS="-std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST" \
SYSLIB="-lblas -llapack" \
./build_sbd_gpu.sh
```
