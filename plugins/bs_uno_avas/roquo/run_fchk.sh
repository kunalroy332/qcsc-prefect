#!/bin/bash
# Active spaces from Gaussian BS-DFT fchk files on ROQUO (docs/tutorials/
# run_bs_uno_avas_roquo.md, Step 2B).
# Run from anywhere.
#
#   bash roquo/run_fchk.sh check    conversion check, one job per fchk
#   bash roquo/run_fchk.sh build    active space + manifest (asks ox/red per fchk)
#   bash roquo/run_fchk.sh report   result of the newest jobs, manifest paths
#
# fchk files: every workspace/data/*.fchk, or the files given after the command.
# Account / reservation: ACCT / RESV from ~/.qcsc_env (a reservation that is not active is left
# out). Output: workspace/output/<stem>_E<state>_R<geometry>_S<tag>_BS_OPT/.
# Logs: workspace/logs/fchk-{check,build}-<fchk stem>.<jobid>.{out,err}; every check / build is
# recorded in workspace/logs/fchk-submit-<time>.log.
# Optional env: STATE=ox|red (same state for every fchk, no question), GEOMETRY (R label),
# CONFIG (default configs/fchk_dgdzvp.toml), TIME, EXCLUSIVE=0 (share the node), VENV,
# BS_UNO_AVAS_WORKSPACE (another workspace folder).
set -eo pipefail
# shellcheck disable=SC1091
[[ -f "$HOME/.qcsc_env" ]] && source "$HOME/.qcsc_env"
set -u
CMD="${1:-}"
[[ $# -gt 0 ]] && shift

if [[ "$CMD" == run ]]; then
    # inside the job (sbatch runs a spool copy of this file: PLUGIN / WS come from submit)
    PLUGIN="${PLUGIN:?not set: use run_fchk.sh check or build}"
    WS="${WS:?not set: use run_fchk.sh check or build}"
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
    export PYSCF_TMPDIR="$WS/work/pyscf_tmp/fchk.${SLURM_JOB_ID:-$$}"
    mkdir -p "$PYSCF_TMPDIR" "$CUPY_CACHE_DIR"
    cd "$WS"
    echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  fchk $FCHK  state ${STATE:-check}" \
         " config $CONFIG  OMP $OMP_NUM_THREADS"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    if [[ -z "${STATE:-}" ]]; then
        args=(check-fchk --fchk "$FCHK" --config "$CONFIG")
    else
        args=(build-fchk --fchk "$FCHK" --config "$CONFIG" --state "$STATE")
        [[ -n "${GEOMETRY:-}" ]] && args+=(--geometry "$GEOMETRY")
    fi
    rc=0
    time python -m bs_uno_avas.cli "${args[@]}" || rc=$?
    rm -rf "$PYSCF_TMPDIR"
    exit $rc
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="$(cd "$HERE/.." && pwd)"
WS="${BS_UNO_AVAS_WORKSPACE:-$PLUGIN/workspace}"
case "$CMD" in
    check | build | report) ;;
    *)
        sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac

# fchk files and config as absolute paths (arguments are relative to the current directory)
FILES=()
for a in "$@"; do
    [[ -f "$a" ]] || { echo "STOP: no file $a"; exit 1; }
    FILES+=("$(cd "$(dirname "$a")" && pwd)/$(basename "$a")")
done
CONFIG="${CONFIG:-configs/fchk_dgdzvp.toml}"
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
    FILES=("$WS"/data/*.fchk)
    shopt -u nullglob
    [[ ${#FILES[@]} -gt 0 ]] || { echo "STOP: no *.fchk in $WS/data"; exit 1; }
fi
mkdir -p logs

newest() { ls -t logs/fchk-"$1"-"$2".*.out 2>/dev/null | head -1 || true; }

job_state() {  # pending / running / ended, for the job id in the log name $1
    [[ "$1" =~ \.([0-9]+)\.out$ ]] || { echo unknown; return; }
    local st
    st="$(squeue -h -j "${BASH_REMATCH[1]}" -o %T 2>/dev/null || true)"
    echo "${st:-ended}" | tr 'A-Z' 'a-z'
}

if [[ "$CMD" == report ]]; then
    manifests=()
    for f in "${FILES[@]}"; do
        stem="$(basename "$f" .fchk)"
        echo "===== $stem"
        for kind in check build; do
            res=""
            log="$(newest "$kind" "$stem")"
            if [[ -z "$log" ]]; then
                echo "  $kind: not submitted"
                continue
            fi
            m=""
            if [[ "$kind" == check ]]; then
                res="$(grep -oE '^check-fchk: (PASS|FAIL)' "$log" || true)"
            else
                m="$(sed -n 's/^wrote \(.*_manifest\.json\) (.*/\1/p' "$log" | tail -1)"
                [[ -z "$m" ]] || res="DONE"
            fi
            [[ -n "${res:-}" ]] || res="no result yet: job $(job_state "$log")"
            echo "  $kind: $res   ($log)"
            if [[ "$kind" == check ]]; then
                grep -E '^(cluster|check:|spins:|WARNING)' "$log" | sed 's/^/    /' || true
            elif [[ -n "$m" ]]; then
                grep -E '^(verify:|  UNO-AVAS|  WARNING)' "$log" | sed 's/^/    /' || true
                echo "    manifest $m"
                manifests+=("$m")
            fi
            if [[ "$res" == "no result yet: job ended" || "$res" == *FAIL* ]]; then
                echo "    last lines of ${log%.out}.err:"
                grep -vE '^(real|user|sys)[[:space:]]|^$' "${log%.out}.err" 2>/dev/null \
                    | tail -n 5 | cut -c1-160 | sed 's/^/      /' || true
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
if [[ "$CMD" == check ]]; then TIME="${TIME:-2:00:00}"; else TIME="${TIME:-4:00:00}"; fi

# redox state per fchk (build): STATE, or asked; the suggestion follows the multiplicity
STATES=()
if [[ "$CMD" == build ]]; then
    for f in "${FILES[@]}"; do
        stem="$(basename "$f" .fchk)"
        mult="$(awk '/^Multiplicity /{print $NF; exit}' "$f")"
        case "$mult" in 1) guess=ox ;; 2) guess=red ;; *) guess="" ;; esac
        s="${STATE:-}"
        if [[ -z "$s" ]]; then
            read -rp "redox state of $stem (multiplicity $mult): ox or red [$guess]: " s
            s="${s:-$guess}"
        fi
        case "$s" in
            ox | red | 3+ | 2+ | 1+) ;;
            *) echo "STOP: state '$s' for $stem: use ox or red (Fe4S4 also 3+, 2+, 1+)"; exit 1 ;;
        esac
        STATES+=("$s")
    done
fi

slog="logs/fchk-submit-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$slog") 2>&1
echo "run_fchk.sh $CMD  $(date)  config $CONFIG  time $TIME  workspace $WS"
opts=(-A "$ACCT" -t "$TIME" -p "${PARTITION:-roquo}" -N 1 --gres=gpu:1)
if [[ -n "${RESV:-}" ]]; then
    if scontrol show reservation "$RESV" 2>/dev/null | grep -q 'State=ACTIVE'; then
        opts+=("--reservation=$RESV")
        echo "reservation $RESV"
    else
        echo "reservation $RESV is not active: submitting without it"
    fi
fi
[[ "${EXCLUSIVE:-1}" == 1 ]] && opts+=(--exclusive)   # whole node: memory for the DF tensors

for i in "${!FILES[@]}"; do
    f="${FILES[$i]}"
    stem="$(basename "$f" .fchk)"
    exp="ALL,PLUGIN=$PLUGIN,WS=$WS,FCHK=$f,CONFIG=$CONFIG"
    if [[ "$CMD" == check ]]; then
        exp+=",STATE="
        echo "--- $stem"
    else
        exp+=",STATE=${STATES[$i]},GEOMETRY=${GEOMETRY:-}"
        echo "--- $stem  state ${STATES[$i]}"
    fi
    sbatch "${opts[@]}" -J "fchk-$CMD-$stem" -o "logs/%x.%j.out" -e "logs/%x.%j.err" \
        --export="$exp" "$HERE/run_fchk.sh" run
done
echo "logs: $WS/logs/fchk-$CMD-*.out"
echo "when the jobs are done: bash roquo/run_fchk.sh report"
