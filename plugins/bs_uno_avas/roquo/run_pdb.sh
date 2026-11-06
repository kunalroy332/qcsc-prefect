#!/bin/bash
# Active spaces from PDB files on ROQUO (docs/tutorials/
# run_bs_uno_avas_roquo.md, Step 2A). Run from anywhere.
#
#   bash roquo/run_pdb.sh check    dry run (structure, charge, sizes), one short job per PDB
#   bash roquo/run_pdb.sh build    BS-UHF + UNO-AVAS + manifest, one job per (PDB, state)
#   bash roquo/run_pdb.sh report   result of the newest jobs, manifest paths
#
# PDB files: every workspace/data/*.pdb (not the generated *_prot.pdb / *_cap.pdb), or the files
# given after the command. build asks per PDB for the redox states (suggested: every
# [states.*] of the config) and stops if the newest check of that PDB did not PASS.
# Output: workspace/output/<pdb stem>_E<state>_R<geometry>_S<tag>/ (+ chkfiles in workspace/work).
# Logs: workspace/logs/pdb-check-<stem>.<jobid>.{out,err}, pdb-build-<stem>-<state>.<jobid>.*,
# and workspace/logs/pdb-submit-<time>.log for every check / build.
# Optional env: STATE (e.g. red: same states for every PDB, no question), CLUSTER_RESNAME (the
# cluster residue name if not FES / SF4), GEOMETRY (R label, default pdb), CONFIG (default
# configs/batch_dzvp.toml), TIME, ENGINE=cpu, EXCLUSIVE=0 (share the node), SKIP_CHECK=1,
# VENV, BS_UNO_AVAS_WORKSPACE (another workspace folder).
set -eo pipefail
# shellcheck disable=SC1091
[[ -f "$HOME/.qcsc_env" ]] && source "$HOME/.qcsc_env"
set -u
CMD="${1:-}"
[[ $# -gt 0 ]] && shift

if [[ "$CMD" == run ]]; then
    # inside the job (sbatch runs a spool copy of this file: PLUGIN / WS come from submit)
    PLUGIN="${PLUGIN:?not set: use run_pdb.sh check or build}"
    WS="${WS:?not set: use run_pdb.sh check or build}"
    VENV="${VENV:-$HOME/venvs/gpu4pyscf}"
    module load cuda/13.2 2>/dev/null || module load cuda 2>/dev/null || true
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export PYTHONPATH="$PLUGIN/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's bs_uno_avas
    export BS_UNO_AVAS_WORKSPACE="$WS"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-72}}"
    export PYTHONUNBUFFERED=1
    export CUPY_CACHE_DIR="/tmp/cupy_cache_$USER"
    export PYSCF_MAX_MEMORY="${PYSCF_MAX_MEMORY:-200000}"
    export PYSCF_TMPDIR="$WS/work/pyscf_tmp/pdb.${SLURM_JOB_ID:-$$}"
    mkdir -p "$PYSCF_TMPDIR" "$CUPY_CACHE_DIR"
    cd "$WS"
    echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  pdb $PDB  state ${STATE:-check}" \
         " config $CONFIG  OMP $OMP_NUM_THREADS"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    python -c "import gpu4pyscf" 2>/dev/null || echo "gpu4pyscf not importable: CPU engine"
    args=(build --config "$CONFIG" --pdb "$PDB")
    [[ -n "${ENGINE:-}" ]] && args+=(--set "scf.engine=$ENGINE")
    [[ -n "${CLUSTER_RESNAME:-}" ]] && args+=(--cluster-resname "$CLUSTER_RESNAME")
    [[ -n "${GEOMETRY:-}" ]] && args+=(--geometry "$GEOMETRY")
    rc=0
    if [[ -z "${STATE:-}" ]]; then
        python -m bs_uno_avas.cli "${args[@]}" --dry-run || rc=$?
        echo "check: $([[ $rc -eq 0 ]] && echo PASS || echo FAIL)"
    else
        time python -m bs_uno_avas.cli "${args[@]}" --state "$STATE" \
            | tee "$PYSCF_TMPDIR/build.out" || rc=$?
        m="$(sed -n 's/^wrote \(.*_manifest\.json\) (.*/\1/p' "$PYSCF_TMPDIR/build.out" | tail -1)"
        if [[ $rc -eq 0 && -n "$m" ]]; then
            # the same reference check the SBD flow makes, from the written files
            if python -m bs_uno_avas.cli verify --manifest "$m" > "$PYSCF_TMPDIR/verify.out"; then
                echo "manifest: OK  $m"
            else
                tail -n 3 "$PYSCF_TMPDIR/verify.out"
                echo "manifest: FAIL  $m"
                rc=1
            fi
        fi
    fi
    rm -rf "$PYSCF_TMPDIR"   # chkfiles live in [output] work_dir, not here
    exit $rc
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="$(cd "$HERE/.." && pwd)"
WS="${BS_UNO_AVAS_WORKSPACE:-$PLUGIN/workspace}"
case "$CMD" in
    check | build | report) ;;
    *)
        sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac

# PDB files and config as absolute paths (arguments are relative to the current directory)
FILES=()
for a in "$@"; do
    [[ -f "$a" ]] || { echo "STOP: no file $a"; exit 1; }
    FILES+=("$(cd "$(dirname "$a")" && pwd)/$(basename "$a")")
done
CONFIG="${CONFIG:-configs/batch_dzvp.toml}"
if [[ -f "$CONFIG" ]]; then
    CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
elif [[ -f "$PLUGIN/$CONFIG" ]]; then
    CONFIG="$PLUGIN/$CONFIG"
else
    echo "STOP: config $CONFIG not found (here or in $PLUGIN)"
    exit 1
fi
mkdir -p "$WS"
cd "$WS"
if [[ ${#FILES[@]} -eq 0 ]]; then
    shopt -s nullglob
    for f in "$WS"/data/*.pdb; do
        [[ "$f" == *_prot.pdb || "$f" == *_cap.pdb ]] || FILES+=("$f")
    done
    shopt -u nullglob
    [[ ${#FILES[@]} -gt 0 ]] || { echo "STOP: no PDB in $WS/data: copy them there"; exit 1; }
fi
# output folders are named after the PDB stem: two PDBs with one stem would collide
dup="$(for f in "${FILES[@]}"; do basename "$f" .pdb; done | sort | uniq -d)"
[[ -z "$dup" ]] || { echo "STOP: two PDB files named $dup: rename one"; exit 1; }
mkdir -p logs

# the redox states of the config ([states.<name>] tables)
CFG_STATES=()
while read -r s; do CFG_STATES+=("$s"); done < <(sed -nE 's/^\[states\.([^]]+)\].*/\1/p' "$CONFIG")
[[ ${#CFG_STATES[@]} -gt 0 ]] || { echo "STOP: no [states.*] table in $CONFIG"; exit 1; }

newest() {  # newest logs/<name>.<jobid>.out
    local f
    for f in $(ls -t logs/"$1".*.out 2>/dev/null); do
        [[ "${f#logs/"$1".}" =~ ^[0-9]+\.out$ ]] && { echo "$f"; return; }
    done
    return 0
}

job_state() {  # pending / running / ended, for the job id in the log name $1
    [[ "$1" =~ \.([0-9]+)\.out$ ]] || { echo unknown; return; }
    local st
    st="$(squeue -h -j "${BASH_REMATCH[1]}" -o %T 2>/dev/null || true)"
    echo "${st:-ended}" | tr 'A-Z' 'a-z'
}

err_tail() {  # last lines of the .err file, without the `time` summary
    echo "    last lines of ${1%.out}.err:"
    grep -vE '^(real|user|sys)[[:space:]]|^$' "${1%.out}.err" 2>/dev/null | tail -n 5 \
        | cut -c1-160 | sed 's/^/      /' || true
}

if [[ "$CMD" == report ]]; then
    manifests=()
    for f in "${FILES[@]}"; do
        stem="$(basename "$f" .pdb)"
        echo "===== $stem"
        log="$(newest "pdb-check-$stem")"
        if [[ -z "$log" ]]; then
            echo "  check: not submitted"
        else
            res="$(grep -oE '^check: (PASS|FAIL)' "$log" | tail -1 || true)"
            echo "  ${res:-check: no result yet: job $(job_state "$log")}   ($log)"
            pat='^(cluster |atoms |Fe-bound|[0-9]+_Fe first shell|state [^ ]+: n_Fe|DF:|WARNING)'
            grep -E "$pat" "$log" | cut -c1-160 | sed 's/^/    /' || true
            [[ "$res" == *PASS* || "$(job_state "$log")" != ended ]] || err_tail "$log"
        fi
        for s in "${CFG_STATES[@]}"; do
            log="$(newest "pdb-build-$stem-$s")"
            [[ -n "$log" ]] || continue
            res="$(grep -oE '^manifest: (OK|FAIL)' "$log" | tail -1 || true)"
            echo "  build $s: ${res:-no result yet: job $(job_state "$log")}   ($log)"
            grep -E '^(  HS UHF guess|  BS UHF|selected|  UNO-AVAS|verify:|  WARNING|.*<- low)' \
                "$log" | cut -c1-160 | sed 's/^/    /' || true
            m="$(sed -nE 's/^manifest: OK  (.*)/\1/p' "$log" | tail -1)"
            if [[ -n "$m" ]]; then
                echo "    manifest $m"
                manifests+=("$m")
            elif [[ "$res" == *FAIL* || "$(job_state "$log")" == ended ]]; then
                err_tail "$log"
            fi
        done
    done
    if [[ ${#manifests[@]} -gt 0 ]]; then
        echo "===== manifests (for the SQD tasks.txt)"
        printf '%s\n' "${manifests[@]}"
    fi
    exit 0
fi

: "${ACCT:?STOP: set ACCT in ~/.qcsc_env}"
if [[ "$CMD" == check ]]; then TIME="${TIME:-0:30:00}"; else TIME="${TIME:-12:00:00}"; fi

# build: redox states per PDB (STATE, or asked; suggested: all states of the config), and the
# newest check of that PDB must be PASS (SKIP_CHECK=1 skips this)
STATES=()
if [[ "$CMD" == build ]]; then
    for f in "${FILES[@]}"; do
        stem="$(basename "$f" .pdb)"
        if [[ -z "${SKIP_CHECK:-}" ]]; then
            log="$(newest "pdb-check-$stem")"
            [[ -n "$log" ]] || { echo "STOP: $stem has no check yet: run check first"; exit 1; }
            grep -q '^check: PASS' "$log" \
                || { echo "STOP: the check of $stem did not PASS ($log): see report"; exit 1; }
        fi
        s="${STATE:-}"
        if [[ -z "$s" ]]; then
            read -rp "redox states of $stem [${CFG_STATES[*]}]: " s
            s="${s:-${CFG_STATES[*]}}"
        fi
        for x in $s; do
            [[ " ${CFG_STATES[*]} " == *" $x "* ]] \
                || { echo "STOP: state '$x' for $stem: use ${CFG_STATES[*]} ($CONFIG)"; exit 1; }
        done
        STATES+=("$s")
    done
fi

slog="logs/pdb-submit-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$slog") 2>&1
echo "run_pdb.sh $CMD  $(date)  config $CONFIG  time $TIME  workspace $WS"
opts=(-A "$ACCT" -t "$TIME" -p "${PARTITION:-roquo}" -N 1 --gres=gpu:1)
if [[ -n "${RESV:-}" ]]; then
    if scontrol show reservation "$RESV" 2>/dev/null | grep -q 'State=ACTIVE'; then
        opts+=("--reservation=$RESV")
        echo "reservation $RESV"
    else
        echo "reservation $RESV is not active: submitting without it"
    fi
fi
# whole node for a build: memory for the DF tensors (a check needs little)
[[ "$CMD" == build && "${EXCLUSIVE:-1}" == 1 ]] && opts+=(--exclusive)

exp0="ALL,PLUGIN=$PLUGIN,WS=$WS,CONFIG=$CONFIG,ENGINE=${ENGINE:-}"
exp0+=",CLUSTER_RESNAME=${CLUSTER_RESNAME:-},GEOMETRY=${GEOMETRY:-}"
for i in "${!FILES[@]}"; do
    f="${FILES[$i]}"
    stem="$(basename "$f" .pdb)"
    if [[ "$CMD" == check ]]; then
        echo "--- $stem"
        sbatch "${opts[@]}" -J "pdb-check-$stem" -o "logs/%x.%j.out" -e "logs/%x.%j.err" \
            --export="$exp0,PDB=$f,STATE=" "$HERE/run_pdb.sh" run
        continue
    fi
    for s in ${STATES[$i]}; do
        echo "--- $stem  state $s"
        sbatch "${opts[@]}" -J "pdb-build-$stem-$s" -o "logs/%x.%j.out" -e "logs/%x.%j.err" \
            --export="$exp0,PDB=$f,STATE=$s" "$HERE/run_pdb.sh" run
    done
done
echo "when the jobs are done: bash roquo/run_pdb.sh report"
