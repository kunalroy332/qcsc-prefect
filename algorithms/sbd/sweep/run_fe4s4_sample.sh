#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "node=1"
#PJM -L "elapse=2:00:00"
#PJM -j
#PJM -S
# 4Fe-4S 72q sampling on Fugaku (kobe), 5M shots -> save pool.
#   Real:    pjsub run_fe4s4_sample.sh
#   Dry-run: FE2S2_QSRC=random pjsub run_fe4s4_sample.sh
# ONE Fugaku job at a time.
source ~/load_env.sh
export UV_PYTHON_INSTALL_DIR="$MY_SPACE/uv_python"
cd "$MY_PROJECT/algorithms/sbd"
export PATH="$MY_PROJECT/algorithms/sbd/.venv/bin:$PATH"

# 4Fe-4S FCIDUMP on Fugaku + molecule selection.
export FE_MOL="fe4s4"
export FE4S4_FCIDUMP="$MY_SPACE/sweep/fcidump_Fe4S4_MO.txt"

# IBM credentials (kobe) from the gitignored sweep/.env.local (never committed).
if [[ -f "$MY_SPACE/sweep/.env.local" ]]; then
    # shellcheck disable=SC1091
    source "$MY_SPACE/sweep/.env.local"
else
    echo "ERROR: $MY_SPACE/sweep/.env.local not found (needs IBM_API_KEY/IBM_CRN/IBM_BACKEND)." >&2
    exit 1
fi

export FE2S2_QSRC="${FE2S2_QSRC:-real-device}"
export PREFECT_HOME="$MY_SPACE/sweep/runs/fe4s4_rhf/prefect_home"
export PREFECT_LOCAL_STORAGE_PATH="$PREFECT_HOME/storage"
mkdir -p "$PREFECT_HOME"

"$MY_PROJECT/algorithms/sbd/.venv/bin/python" "$MY_SPACE/sweep/run_fe4s4_sample.py"
echo "EXIT_fe4s4_sample=$?"
