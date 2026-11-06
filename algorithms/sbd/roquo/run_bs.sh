#!/bin/bash
# BS-UHF SQD on ROQUO from active-space manifests (docs/tutorials/run_bs_uhf_sqd_roquo.md).
#
#   bash run_bs.sh check  [manifest ...]   login checks + a short 1-node check job
#   bash run_bs.sh job    [manifest ...]   one SQD run per manifest, [flow].sqd_dim
#   bash run_bs.sh sweep  [manifest ...]   one SQD run per size of [sweep].sqd_dims
#   bash run_bs.sh report [manifest ...]   check result, state of every task and size
#
# Manifests: every plugins/bs_uno_avas/workspace/output/**/*_manifest.json, or the files given
# (or TASKS=file, one per line). job / sweep list them with their state and ask which to run
# (suggested: all not done), then one action per task, suggested from its state:
#   run     new task, or samples but no checkpoint (sampling only if it has no samples yet)
#   resume  continue a cut-off recovery from its checkpoint (same samples and settings)
#   new     recovery from step 1 again, on the same samples
#   skip    leave it out
# job / sweep stop unless the newest check passed for those manifests (SKIP_CHECK=1).
# Settings: run_bs.toml. Results: runs/<run_name>/d<size>/result.json, samples in
# runs/<run_name>/samples/. Logs: logs/bs-check.<jobid>.out, logs/bs-sqd[-sweep].<jobid>.out,
# logs/<jobid>/<k>_<run_name>.log (one per task), logs/submit-<time>.log.
# Optional env: CONFIG (default run_bs.toml), TASKS, RESUME=run|resume|new|skip (every task,
# no question), RESAMPLE=1 (new device samples), SKIP_CHECK=1, MODE=direct|cloud (job only),
# TIME, ALLOC=1 (print salloc; inside it: bash run_bs.sh run), RESV_MARGIN, SKIP_REACH=1.
set -eo pipefail
# shellcheck disable=SC1091
[[ -f "$HOME/.qcsc_env" ]] && source "$HOME/.qcsc_env"
set -u
CMD="${1:-}"
[[ $# -gt 0 ]] && shift

# Under sbatch the script runs from a spool copy: check / job pass this folder as ROQUO_DIR.
HERE="${ROQUO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "$HERE/../../.." && pwd)"
SBD="$REPO/algorithms/sbd"
PY="${SBD_PYTHON:-$SBD/.venv/bin/python}"
WS="${BS_UNO_AVAS_WORKSPACE:-$REPO/plugins/bs_uno_avas/workspace}"
SECRETS="${SECRETS:-$REPO/secrets/secrets.env}"
CONFIG="${CONFIG:-run_bs.toml}"
if [[ "$CONFIG" != /* ]]; then
    if [[ -f "$CONFIG" ]]; then CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
    else CONFIG="$HERE/$CONFIG"; fi
fi

# ---- helpers --------------------------------------------------------------------------------
# run_bs.toml is read with awk: the login node has no Python >= 3.11 (tomllib).
toml() {  # toml <section.key> [default]; e.g. slurm.time, flow.sampler.shots; "" = default
    local val
    val="$(awk -v sec="${1%.*}" -v key="${1##*.}" '
        { sub(/\r$/, "") }
        /^[[:space:]]*\[/ { cur = $0; gsub(/^[[:space:]]*\[|\].*$/, "", cur); next }
        cur == sec && match($0, /^[[:space:]]*[A-Za-z0-9_]+[[:space:]]*=/) {
            k = substr($0, RSTART, RLENGTH); sub(/=$/, "", k); gsub(/[[:space:]]/, "", k)
            if (k != key) next
            v = substr($0, RSTART + RLENGTH); sub(/^[[:space:]]+/, "", v)
            if (v ~ /^"/) { v = substr(v, 2); v = substr(v, 1, index(v, "\"") - 1) }
            else if (v ~ /^\[/) { sub(/\].*$/, "]", v) }
            else { sub(/[[:space:]]*#.*$/, "", v); sub(/[[:space:]]+$/, "", v) }
            print v; exit
        }' "$CONFIG")"
    echo "${val:-${2:-}}"
}
num() { tr -d '_' <<< "$1"; }   # TOML 1_000_000 -> 1000000

sweep_sizes() {  # [sweep].sqd_dims as "N1,N2,..." (one line in the toml), or nothing
    local d; d="$(toml sweep.sqd_dims | tr -d '[:space:]_')"
    [[ "$d" =~ ^\[[0-9]+(,[0-9]+)*,?\]$ ]] || return 0
    d="${d#[}"; d="${d%]}"; echo "${d%,}"
}

kind_sizes() {  # $1 = job | sweep | report
    case "$1" in
        job) num "$(toml flow.sqd_dim 1000000)" ;;
        *) sweep_sizes ;;
    esac
}

runs_root() {  # [run].runs_root ("" = this folder's runs/), absolute
    local r; r="$(toml run.runs_root)"
    if [[ -z "$r" ]]; then echo "$HERE/runs"; elif [[ "$r" = /* ]]; then echo "$r"
    else echo "$REPO/$r"; fi
}

run_name() {  # manifest.json is written with indent=2: one key per line
    sed -nE 's/^[[:space:]]*"run_name":[[:space:]]*"([^"]*)".*/\1/p' "$1" | head -1
}

find_manifests() {  # MANIFESTS=(...): args, else TASKS, else the bs_uno_avas workspace
    MANIFESTS=()
    local a line m
    for a in "$@"; do
        [[ -f "$a" ]] || { echo "STOP: no file $a"; exit 1; }
        MANIFESTS+=("$(cd "$(dirname "$a")" && pwd)/$(basename "$a")")
    done
    [[ ${#MANIFESTS[@]} -gt 0 ]] && return 0
    if [[ -n "${TASKS:-}" ]]; then
        [[ -f "$TASKS" ]] || { echo "STOP: TASKS file $TASKS not found"; exit 1; }
        while IFS= read -r line || [[ -n "$line" ]]; do
            m="$(echo "${line%%#*}" | xargs)"
            [[ -z "$m" ]] && continue
            [[ "$m" = /* ]] || m="$REPO/$m"
            MANIFESTS+=("$m")
        done < "$TASKS"
        return 0
    fi
    shopt -s nullglob
    for m in "$WS"/output/*_manifest.json "$WS"/output/*/*_manifest.json; do
        MANIFESTS+=("$m")
    done
    shopt -u nullglob
}

STATE=()  # task_state.py lines: index run size state detail samples (tab-separated)
load_state() {  # $1 = sizes "N1,N2"; $2 = 1: also every existing d<size> (report)
    STATE=()
    [[ ${#MANIFESTS[@]} -gt 0 ]] || return 0
    local args=(--runs-root "$(runs_root)" --sizes "$1"
                --n-steps "$(num "$(toml flow.n_recovery_steps 1)")"
                --n-batches "$(num "$(toml flow.n_batches 1)")"
                --source "$(toml flow.quantum_source real-device)"
                --seed "$(num "$(toml flow.random_seed 24)")"
                --shots "$(num "$(toml flow.sampler.shots 0)")")
    local saved; saved="$(toml flow.sampler.saved_samples | sed -nE 's/^\["([^"]*)".*/\1/p')"
    [[ -n "$saved" ]] && args+=(--saved-pool "$saved")
    [[ "${2:-0}" == 1 ]] && args+=(--all-sizes)
    [[ -n "${RESAMPLE:-}" ]] && args+=(--resample)
    local line
    if command -v python3 >/dev/null; then
        while IFS= read -r line; do STATE+=("$line"); done \
            < <(python3 "$HERE/task_state.py" "${args[@]}" "${MANIFESTS[@]}")
    fi
}

field() { cut -f"$1" <<< "$2"; }   # field <n> <state line>

states_of() {  # the state lines of task index $1
    local s
    for s in ${STATE[@]+"${STATE[@]}"}; do [[ "$(field 1 "$s")" == "$1" ]] && echo "$s"; done
    return 0
}

first_state() {  # the first state line of task index $1
    local s
    for s in ${STATE[@]+"${STATE[@]}"}; do
        [[ "$(field 1 "$s")" == "$1" ]] && { echo "$s"; return 0; }
    done
    return 0
}

sizes_text() {  # "d<size> <detail>; ..." of task index $1
    states_of "$1" | awk -F'\t' '{printf "%sd%s %s", (NR > 1 ? "; " : ""), $3, $5}'
}

newest() {  # newest logs/<name>.<jobid>.out
    local f
    for f in $(ls -t logs/"$1".*.out 2>/dev/null); do
        [[ "${f#logs/"$1".}" =~ ^[0-9]+\.out$ ]] && { echo "$f"; return; }
    done
    return 0
}

job_state() {  # pending / running / ended, for job id $1
    local st
    st="$(squeue -h -j "$1" -o %T 2>/dev/null || true)"
    echo "${st:-ended}" | tr 'A-Z' 'a-z'
}

err_tail() {  # last lines of a job's .err file
    echo "    last lines of ${1%.out}.err:"
    grep -v '^$' "${1%.out}.err" 2>/dev/null | tail -n 5 | cut -c1-160 | sed 's/^/      /' \
        || true
}

why_failed() {  # <task log>: the lines that say why a task failed (indented, cut to 150 chars)
    local log="$1" exc slurm
    [[ -f "$log" ]] || { echo "       (no task log $log)"; return; }
    # the last Python exception line is the error the task ended with; the first `srun: error:`
    # / `slurmstepd: error:` line is usually its cause (e.g. a refused step)
    exc="$(grep -hE '^[A-Za-z_.]*(Error|Exception|Exit|Interrupt)( |:|$)|FAILED:' "$log" \
           | tail -1 || true)"
    slurm="$(grep -hE '(srun|slurmstepd): error:' "$log" | grep -v 'Exited with exit code' \
             | head -1 || true)"
    [[ -n "$exc" ]] && echo "       ${exc:0:150}"
    [[ -n "$slurm" ]] && echo "       ${slurm:0:150}"
    [[ -n "$exc$slurm" ]] || echo "       last line: $(tail -1 "$log" | cut -c1-150)"
    echo "       log: $log"
}

load_env() {  # ROQUO runtime (in a job); secrets.env only if it exists (IBM block fallback)
    NV=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5
    module load cuda/13.2 nvhpc/26.5 2>/dev/null || true
    module load hpcx/2.50 2>/dev/null || true
    export PATH="$NV/compilers/bin:$SBD/.venv/bin:$PATH"
    export LD_LIBRARY_PATH="$NV/compilers/lib:$NV/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"
    export UV_CACHE_DIR="/tmp/uvcache_$USER" TMPDIR="/tmp/uvtmp_$USER"
    mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
    export MY_PROJECT="$REPO" SBD_TASK_RUNNER=concurrent SBD_BS_STRICT="${SBD_BS_STRICT:-1}"
    export PYTHONUNBUFFERED=1   # task logs are files: print() lines appear as they happen
    # a file still holding template values (EDIT...) is ignored: the ibm-runner block is used
    if [[ -f "$SECRETS" ]] && ! grep -qE '^[[:space:]]*[A-Z_]+="[^"]*EDIT' "$SECRETS"; then
        set -a; source "$SECRETS"; set +a
    fi
    # an empty PREFECT_API_URL / _KEY would hide the `prefect cloud login` profile
    [[ -n "${PREFECT_API_URL:-}" ]] || unset PREFECT_API_URL
    [[ -n "${PREFECT_API_KEY:-}" ]] || unset PREFECT_API_KEY
    CPN="${SLURM_CPUS_ON_NODE:-$(nproc)}"
    GPN="$(toml slurm.gpus_per_node 4)"
}

check_layout() {  # [run].gpus_per_task vs the [solver] MPI grid and the binary (pure bash)
    local gpt gpn t a b bin sfx
    gpt="$(toml run.gpus_per_task 4)"; gpn="$(toml slurm.gpus_per_node 4)"
    t="$(toml solver.task_comm_size 1)"; a="$(toml solver.adet_comm_size)"
    b="$(toml solver.bdet_comm_size)"
    if (( gpt > gpn )); then
        echo "STOP: [run].gpus_per_task=$gpt > [slurm].gpus_per_node=$gpn"; return 1
    fi
    if [[ -n "$a" && -n "$b" ]] && (( t * a * b != gpt )); then
        echo "STOP: [solver] task x adet x bdet = $t x $a x $b = $((t * a * b)) ranks," \
             "but [run].gpus_per_task = $gpt (one rank per GPU)"
        return 1
    fi
    sfx="$(toml run.binary_suffix)"; [[ -n "$sfx" ]] || { (( gpt > 1 )) && sfx=-mpi-hpcx250; }
    bin="$SBD/native/diag-gpu_uhf$sfx"
    [[ -x "$bin" ]] || { echo "STOP: solver binary $bin missing (setup how-to, Step 4)"; return 1; }
    echo "solver: $gpt GPU(s) per task, grid $t x ${a:-auto} x ${b:-auto}, $bin"
}

check_files() {  # login node: config, manifests, run_names (pure bash)
    local bad=0 m rn names=""
    [[ -f "$CONFIG" ]] || { echo "STOP: config $CONFIG not found"; return 1; }
    if grep -q '"EDIT' "$CONFIG"; then
        echo "STOP: EDIT placeholders left in $CONFIG:"; grep -n '"EDIT' "$CONFIG"; bad=1
    fi
    for m in ${MANIFESTS[@]+"${MANIFESTS[@]}"}; do
        if [[ ! -f "$m" ]]; then echo "STOP: manifest missing: $m"; bad=1; continue; fi
        rn="$(run_name "$m")"
        [[ -n "$rn" ]] || { echo "STOP: no run_name in $m"; bad=1; continue; }
        names+="$rn"$'\n'
    done
    local dup; dup="$(printf '%s' "$names" | sort | uniq -d | xargs)"
    [[ -z "$dup" ]] || { echo "STOP: two manifests with run_name $dup (one run dir)"; bad=1; }
    return $bad
}

slurm_secs() {  # Slurm time (D-HH:MM:SS, HH:MM:SS, MM:SS, MM) -> seconds
    local t="$1" d=0 h=0 m=0 s=0 IFS=:
    if [[ "$t" == *-* ]]; then
        d="${t%%-*}"; t="${t#*-}"
        read -r h m s <<< "$t"
    else
        local p; read -ra p <<< "$t"
        case ${#p[@]} in
            1) m="${p[0]}" ;;
            2) m="${p[0]}"; s="${p[1]}" ;;
            *) h="${p[0]}"; m="${p[1]}"; s="${p[2]}" ;;
        esac
    fi
    echo $(( 10#${d:-0} * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0} ))
}

slurm_time() {  # D-HH:MM:SS from seconds (rounded down to the minute)
    local x=$(( $1 / 60 * 60 ))
    printf '%d-%02d:%02d:%02d' $(( x / 86400 )) $(( x % 86400 / 3600 )) $(( x % 3600 / 60 )) 0
}

SLURM_OPTS=()
slurm_opts() {  # account, partition, reservation (only if ACTIVE), wall time $1
    local acct resv info end left t="$1"
    acct="$(toml slurm.account)"; acct="${acct:-${ACCT:-}}"
    [[ -n "$acct" ]] || { echo "STOP: set ACCT in ~/.qcsc_env (or [slurm].account)"; exit 1; }
    SLURM_OPTS=(-A "$acct" -p "$(toml slurm.partition roquo)")
    resv="$(toml slurm.reservation)"; resv="${resv:-${RESV:-}}"
    [[ -n "$resv" ]] || { SLURM_TIME="$t"; return 0; }
    info="$(scontrol show reservation "$resv" 2>/dev/null || true)"
    if ! grep -q 'State=ACTIVE' <<< "$info"; then
        echo "reservation $resv is not active: submitting without it"
        SLURM_TIME="$t"; return 0
    fi
    SLURM_OPTS+=("--reservation=$resv")
    echo "reservation $resv"
    # a job whose -t runs past the reservation end never starts: shorten it
    end="$(grep -oE 'EndTime=[^ ]+' <<< "$info" | head -1 | cut -d= -f2)"
    end="$(date -d "$end" +%s 2>/dev/null)" || { SLURM_TIME="$t"; return 0; }
    left=$(( end - $(date +%s) - ${RESV_MARGIN:-600} ))
    (( left >= 60 )) || { echo "STOP: reservation $resv ends within ${RESV_MARGIN:-600} s"; exit 1; }
    if (( $(slurm_secs "$t") > left )); then
        t="$(slurm_time "$left")"
        echo "time: shortened to $t (end of reservation $resv)"
    fi
    SLURM_TIME="$t"
}

reach_and_preflight() {  # in a job: every node reaches Prefect Cloud / IBM; pre-flight on one
    local jobdir="$1"; shift
    local nodes=("$@")
    if [[ -z "${SKIP_REACH:-}" ]]; then
        if ! srun --nodes="${#nodes[@]}" --ntasks-per-node=1 --ntasks="${#nodes[@]}" \
                --exclusive "$PY" "$HERE/check_reach.py" > "$jobdir/reach.log" 2>&1; then
            cat "$jobdir/reach.log"
            echo "REACHABILITY CHECK FAILED (see $jobdir/reach.log)"
            return 2
        fi
        grep -h '^\[reach\]' "$jobdir/reach.log" | sort | uniq -c | sed 's/^/  /' | head -4
    fi
    # solver binary + libraries, the nested solver launch on every rank, Prefect block types,
    # the IBM block ibm-runner (kept if it exists)
    local plog="$jobdir/preflight.log"
    echo "pre-flight on ${nodes[0]} -> $plog"
    if ! timeout 900 srun --input=none --nodes=1 --ntasks=1 --exclusive -w "${nodes[0]}" \
            --job-name=preflight --cpus-per-task="$CPN" --gpus-per-node="$GPN" --mem=0 \
            "$PY" "$HERE/preflight.py" --config "$CONFIG" > "$plog" 2>&1; then
        cat "$plog"
        echo "PRE-FLIGHT FAILED on ${nodes[0]} (see $plog)"
        return 2
    fi
    grep -h '^\[preflight\]\|^\[probe\]' "$plog" | sed 's/^/  /'
}

# ---- inside the check job -------------------------------------------------------------------
if [[ "$CMD" == checkjob ]]; then
    cd "$HERE"
    load_env
    MANIFESTS=()
    if [[ -n "${CHECK_LIST:-}" && -f "$CHECK_LIST" ]]; then
        while IFS= read -r m; do MANIFESTS+=("$m"); done < "$CHECK_LIST"
    fi
    echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  config $CONFIG" \
         " manifests ${#MANIFESTS[@]}"
    bad=0
    "$PY" -c 'import sbd, qcsc_workflow_utility, prefect_qiskit' \
        || { echo "FAILED: the SBD venv $PY does not import sbd (setup how-to Step 3)"; bad=1; }
    for m in ${MANIFESTS[@]+"${MANIFESTS[@]}"}; do
        line="$("$PY" "$HERE/run_task.py" --config "$CONFIG" --manifest "$m" --check \
            | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("reference %(run_name)s: norb %(norb)s nelec %(nelec)s tag %(manifest_tag)s -> %(reference_tag)s (%(reference_source)s) dE %(de_vs_manifest).2e" % d)')" \
            || { echo "FAILED: reference of $m"; bad=1; continue; }
        echo "$line"
    done
    if [[ ${#MANIFESTS[@]} -gt 0 ]]; then
        "$PY" "$HERE/run_task.py" --config "$CONFIG" --manifest "${MANIFESTS[0]}" \
            --print-params > /dev/null \
            || { echo "FAILED: [flow] of $CONFIG is not valid FlowParameters"; bad=1; }
    fi
    jobdir="logs/${SLURM_JOB_ID:-check}"
    mkdir -p "$jobdir"
    reach_and_preflight "$jobdir" "$(hostname -s)" || bad=1
    for m in ${MANIFESTS[@]+"${MANIFESTS[@]}"}; do echo "checked: $m"; done
    echo "check: $([[ $bad == 0 ]] && echo PASS || echo FAIL)"
    exit $bad
fi

# ---- inside the job / sweep allocation -------------------------------------------------------
if [[ "$CMD" == run ]]; then
    cd "$HERE"
    JOBTASKS="${JOBTASKS:-logs/tasks-last.txt}"
    [[ -f "$JOBTASKS" ]] || { echo "STOP: no task list $JOBTASKS (run job or sweep first)"; exit 1; }
    KIND="$(sed -n 's/^# kind //p' "$JOBTASKS")"
    load_env
    MODE="${MODE:-$(toml run.mode direct)}"
    [[ "$KIND" == job ]] || MODE=direct
    TK=(); TM=(); TA=(); TR=()
    while IFS=$'\t' read -r k m a r; do
        [[ "$k" == "#"* || -z "$k" ]] && continue
        TK+=("$k"); TM+=("$m"); TA+=("$a"); TR+=("$r")
    done < "$JOBTASKS"
    nodes=($(scontrol show hostnames "${SLURM_JOB_NODELIST:?not inside an allocation}"))
    jobdir="logs/${SLURM_JOB_ID}"
    mkdir -p "$jobdir"
    check_layout || exit 1
    echo "job $SLURM_JOB_ID: $KIND, ${#nodes[@]} nodes x $CPN CPUs x $GPN GPUs," \
         "${#TK[@]} tasks, mode $MODE"
    # IBM only if a task samples on the device (set by job / sweep from the task states)
    export SAMPLES_DEVICE="${SAMPLES_DEVICE:-1}"
    reach_and_preflight "$jobdir" "${nodes[@]}" || { echo "no task was started"; exit 2; }

    declare -a node_pid node_task task_rc
    for i in "${!nodes[@]}"; do node_pid[$i]=""; done
    start_task() {  # <node index> <list index>
        local i=$1 j=$2 log act=continue
        [[ "${TA[$j]}" == new ]] && act=new
        local targs=(--config "$CONFIG" --manifest "${TM[$j]}" --kind "$KIND" --action "$act")
        [[ -n "${RESAMPLE:-}" ]] && targs+=(--resample)
        [[ "$KIND" == job ]] && targs+=(--mode "$MODE")
        log="$jobdir/${TK[$j]}_${TR[$j]}.log"
        echo "$(date '+%F %T') start task ${TK[$j]} (${TA[$j]}) on ${nodes[$i]} -> $log"
        # whole node for this step: all CPUs, all GPUs, all memory (else 1 CPU, 0 GPU)
        srun --nodes=1 --ntasks=1 --exclusive -w "${nodes[$i]}" --job-name="t${TK[$j]}" \
            --cpus-per-task="$CPN" --gpus-per-node="$GPN" --mem=0 \
            "$PY" "$HERE/run_task.py" "${targs[@]}" > "$log" 2>&1 &
        node_pid[$i]=$!; node_task[$i]=$j
    }
    reap() {  # 0 if a node is free
        local i rc free=1
        for i in "${!nodes[@]}"; do
            [[ -z "${node_pid[$i]}" ]] && { free=0; continue; }
            if ! kill -0 "${node_pid[$i]}" 2>/dev/null; then
                rc=0; wait "${node_pid[$i]}" || rc=$?
                task_rc[${node_task[$i]}]=$rc
                echo "$(date '+%F %T') task ${TK[${node_task[$i]}]} finished: exit $rc"
                node_pid[$i]=""; free=0
            fi
        done
        return $free
    }
    for j in "${!TK[@]}"; do
        until reap; do sleep "${POLL:-30}"; done
        for i in "${!nodes[@]}"; do
            if [[ -z "${node_pid[$i]}" ]]; then start_task "$i" "$j"; break; fi
        done
    done
    while :; do
        reap || true; busy=0
        for i in "${!nodes[@]}"; do [[ -n "${node_pid[$i]}" ]] && busy=1; done
        [[ $busy == 0 ]] && break
        sleep "${POLL:-30}"
    done
    echo "===== summary (job $SLURM_JOB_ID, $KIND, mode $MODE) ====="
    failed=0
    for j in "${!TK[@]}"; do
        rc="${task_rc[$j]:-?}"; [[ "$rc" == 0 ]] || failed=$((failed + 1))
        log="$jobdir/${TK[$j]}_${TR[$j]}.log"
        printf '%3s exit %-3s %s\n' "${TK[$j]}" "$rc" "${TR[$j]}"
        grep -h '^CAPTURE_BS_SQD' "$log" 2>/dev/null | sed 's/^/      /' || true
        [[ "$rc" == 0 ]] || why_failed "$log"
    done
    echo "$failed task(s) failed"
    [[ $failed == 0 ]]
    exit
fi

case "$CMD" in
    check | job | sweep | report) ;;
    *)
        sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac

cd "$HERE"
mkdir -p logs
find_manifests "$@"

# ---- report ---------------------------------------------------------------------------------
if [[ "$CMD" == report ]]; then
    log="$(newest bs-check)"
    if [[ -z "$log" ]]; then
        echo "===== check: not submitted (bash run_bs.sh check)"
    else
        id="${log%.out}"; id="${id##*.}"
        res="$(grep -oE '^check: (PASS|FAIL)' "$log" | tail -1 || true)"
        echo "===== ${res:-check: no result yet: job $(job_state "$id")}   ($log)"
        grep -E '^(reference |FAILED|.*(PRE-FLIGHT|REACHABILITY CHECK) FAILED|  \[preflight\] OK)' \
            "$log" \
            | cut -c1-160 | sed 's/^/  /' || true
        [[ "$res" == *FAIL* || ( -z "$res" && "$(job_state "$id")" == ended ) ]] \
            && err_tail "$log"
    fi
    [[ ${#MANIFESTS[@]} -gt 0 ]] || { echo "no manifests in $WS/output"; exit 0; }
    load_state "$(kind_sizes job)" 1
    [[ ${#STATE[@]} -gt 0 ]] || echo "(python3 not found: no task states)"
    energies=()
    for k in "${!MANIFESTS[@]}"; do
        m="${MANIFESTS[$k]}"; rn="$(run_name "$m")"
        echo "===== $k ${rn:-?}   ($m)"
        first=1
        while IFS= read -r s; do
            [[ -n "$s" ]] || continue
            [[ $first == 1 ]] && echo "  samples: $(field 6 "$s")"; first=0
            echo "  d$(field 3 "$s"): $(field 5 "$s")"
            [[ "$(field 4 "$s")" == done ]] \
                && energies+=("$rn d$(field 3 "$s") $(field 5 "$s" | sed 's/^done //')")
        done < <(states_of "$k")
        tlog="$(ls -t logs/[0-9]*/[0-9]*_"$rn".log 2>/dev/null | head -1 || true)"
        [[ -n "$tlog" ]] || continue
        id="$(basename "$(dirname "$tlog")")"; st="$(job_state "$id")"
        echo "  newest task log: $tlog (job $id $st)"
        grep -h '^CAPTURE_BS_SQD\|^\[task\] \(checkpoint\|size\)' "$tlog" | tail -3 \
            | cut -c1-160 | sed 's/^/    /' || true
        [[ "$st" == ended ]] && ! grep -q '^CAPTURE_BS_SQD' "$tlog" && why_failed "$tlog"
    done
    if [[ ${#energies[@]} -gt 0 ]]; then
        echo "===== energies (runs/<run_name>/d<size>/result.json)"
        printf '  %s\n' "${energies[@]}"
    fi
    exit 0
fi

# ---- check / job / sweep: login-node checks -------------------------------------------------
: "${ACCT:?STOP: set ACCT in ~/.qcsc_env}"
check_files || exit 1
check_layout || exit 1
KIND="$CMD"
if [[ "$KIND" == sweep && -z "$(sweep_sizes)" ]]; then
    echo "STOP: [sweep] sqd_dims in $CONFIG must be one line, e.g. [100_000, 1_000_000]"
    exit 1
fi
[[ "$KIND" == check || ${#MANIFESTS[@]} -gt 0 ]] \
    || { echo "STOP: no manifest in $WS/output: run bs_uno_avas first"; exit 1; }

# ---- check: one short job ------------------------------------------------------------------
if [[ "$KIND" == check ]]; then
    slog="logs/submit-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee "$slog") 2>&1
    echo "run_bs.sh check  $(date)  config $CONFIG  manifests ${#MANIFESTS[@]}"
    list="logs/check-list-$(date +%Y%m%d-%H%M%S).txt"
    : > "$list"
    for m in ${MANIFESTS[@]+"${MANIFESTS[@]}"}; do
        echo "$m" >> "$list"
        echo "  $(run_name "$m")  ($m)"
    done
    [[ ${#MANIFESTS[@]} -gt 0 ]] || echo "  no manifests yet: environment checks only"
    slurm_opts "${TIME:-00:30:00}"
    sbatch "${SLURM_OPTS[@]}" -t "$SLURM_TIME" -N 1 --exclusive \
        --gpus-per-node="$(toml slurm.gpus_per_node 4)" --mem=0 -J bs-check \
        -o "logs/%x.%j.out" -e "logs/%x.%j.err" \
        --export="ALL,ROQUO_DIR=$HERE,CONFIG=$CONFIG,CHECK_LIST=$HERE/$list" \
        "$HERE/run_bs.sh" checkjob
    echo "when the job is done: bash run_bs.sh report"
    exit 0
fi

# ---- job / sweep: the newest check must cover these manifests -------------------------------
if [[ -z "${SKIP_CHECK:-}" ]]; then
    log="$(newest bs-check)"
    [[ -n "$log" ]] || { echo "STOP: no check yet: bash run_bs.sh check"; exit 1; }
    grep -q '^check: PASS' "$log" \
        || { echo "STOP: the newest check ($log) did not PASS: see report"; exit 1; }
    for m in "${MANIFESTS[@]}"; do
        grep -qxF "checked: $m" "$log" && continue
        echo "STOP: $(run_name "$m") was not in the newest check ($log):" \
             "bash run_bs.sh check"
        exit 1
    done
fi
if [[ -n "${MODE:-}" && "$MODE" != direct && "$MODE" != cloud ]]; then
    echo "STOP: MODE=$MODE: use direct or cloud"; exit 1
fi
[[ "$KIND" == sweep && "${MODE:-direct}" == cloud ]] \
    && { echo "STOP: MODE=cloud works with job only"; exit 1; }

SIZES="$(kind_sizes "$KIND")"
load_state "$SIZES"
[[ ${#STATE[@]} -gt 0 ]] || { echo "STOP: python3 not found on this node (task states)"; exit 1; }

suggest() {  # suggested action of task $1 from its sizes' states
    local s st all_done=1 any_resume=0 any_stale=0
    while IFS= read -r s; do
        [[ -n "$s" ]] || continue
        st="$(field 4 "$s")"
        [[ "$st" == done ]] || all_done=0
        [[ "$st" == resume ]] && any_resume=1
        [[ "$st" == stale ]] && any_stale=1
        [[ "$st" == bad ]] && { echo bad; return; }
    done < <(states_of "$1")
    if [[ $all_done == 1 ]]; then echo skip
    elif [[ $any_resume == 1 ]]; then echo resume
    elif [[ $any_stale == 1 ]]; then echo new
    else echo run; fi
}

# 1. which tasks: asked only when the manifests were not named (args / TASKS)
echo "$KIND: sizes $SIZES, $(toml flow.n_recovery_steps) recovery steps, config $CONFIG"
pick=""
for k in "${!MANIFESTS[@]}"; do
    s="$(first_state "$k")"; sug="$(suggest "$k")"
    printf '%3d %-34s %s\n' "$k" "$(field 2 "$s")" "$(sizes_text "$k" | cut -c1-90)"
    [[ "$sug" == skip || "$sug" == bad ]] || pick+="$k "
done
pick="${pick% }"
if [[ $# -eq 0 && -z "${TASKS:-}" ]]; then
    read -rp "tasks [${pick:-none}] (numbers, all): " ans || true   # EOF = Enter
    ans="${ans:-$pick}"
    [[ "$ans" == all ]] && ans="${!MANIFESTS[*]}"
else
    ans="${!MANIFESTS[*]}"
fi
SEL=()
for k in $ans; do
    [[ "$k" =~ ^[0-9]+$ && $k -lt ${#MANIFESTS[@]} ]] \
        || { echo "STOP: task '$k': use numbers 0-$(( ${#MANIFESTS[@]} - 1 )) or all"; exit 1; }
    SEL+=("$k")
done
[[ ${#SEL[@]} -gt 0 ]] || { echo "STOP: no task chosen"; exit 1; }

# 2. one action per task (RESUME answers for every task)
ACT=()
for k in "${SEL[@]}"; do
    sug="$(suggest "$k")"
    [[ "$sug" == bad ]] && { echo "STOP: $(states_of "$k" | cut -f5)"; exit 1; }
    echo "--- $(field 2 "$(first_state "$k")")"
    echo "    samples: $(field 6 "$(first_state "$k")")"
    while IFS= read -r s; do echo "    d$(field 3 "$s"): $(field 5 "$s")"; done \
        < <(states_of "$k")
    a="${RESUME:-}"
    [[ -n "$a" ]] || read -rp "    action [$sug] (run|resume|new|skip): " a || true
    a="${a:-$sug}"
    case "$a" in
        run | resume | new | skip) ;;
        *) echo "STOP: action '$a': use run, resume, new or skip"; exit 1 ;;
    esac
    if [[ "$a" != new && "$sug" == skip ]]; then
        [[ "$a" == skip ]] || echo "    every size is done: skipped (new runs it again)"
        a=skip
    fi
    ACT+=("$a")
done

# 3. submit
slog="logs/submit-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$slog") 2>&1
echo "run_bs.sh $KIND  $(date)  config $CONFIG"
tasks_file="logs/tasks-$(date +%Y%m%d-%H%M%S).txt"
{
    echo "# kind $KIND"
    n=0
    for i in "${!SEL[@]}"; do
        k="${SEL[$i]}"; [[ "${ACT[$i]}" == skip ]] && continue
        printf '%s\t%s\t%s\t%s\n' "$k" "${MANIFESTS[$k]}" "${ACT[$i]}" \
            "$(run_name "${MANIFESTS[$k]}")"
    done
} > "$tasks_file"
cp "$tasks_file" logs/tasks-last.txt
N="$(grep -vc '^#' "$tasks_file" || true)"
[[ "$N" -gt 0 ]] || { echo "nothing to run (every chosen task skipped)"; exit 0; }
awk -F'\t' '!/^#/ {print "  task " $1 "  " $3 "  " $4}' "$tasks_file"

# IBM is needed only if a task samples on the device: no pool yet, or RESAMPLE
samples=0
if [[ "$(toml flow.quantum_source real-device)" == real-device ]]; then
    while IFS=$'\t' read -r k m a r; do
        [[ "$k" == "#"* ]] && continue
        [[ -n "${RESAMPLE:-}" || "$(field 6 "$(first_state "$k")")" == none ]] \
            && samples=1
    done < "$tasks_file"
fi
echo "IBM sampling: $([[ $samples == 1 ]] && echo yes || echo 'no (every task has samples)')"

slurm_opts "${TIME:-$(toml slurm.time 1-00:00:00)}"
name="$(toml slurm.job_name bs-sqd)"; [[ "$KIND" == sweep ]] && name="$name-sweep"
opts=("${SLURM_OPTS[@]}" -t "$SLURM_TIME" -N "$N" --exclusive
      --gpus-per-node="$(toml slurm.gpus_per_node 4)" --mem="$(toml slurm.mem 1600000M)")
exp="ALL,ROQUO_DIR=$HERE,CONFIG=$CONFIG,JOBTASKS=$HERE/$tasks_file,SAMPLES_DEVICE=$samples"
exp+=",MODE=${MODE:-},RESAMPLE=${RESAMPLE:-}"
if [[ -n "${ALLOC:-}" ]]; then
    echo "salloc ${opts[*]} -J $name"
    echo "then, in the shell salloc opens:"
    echo "  source ~/.qcsc_env; cd $HERE"
    echo "  SAMPLES_DEVICE=$samples bash run_bs.sh run"
    exit 0
fi
sbatch "${opts[@]}" -J "$name" -o "logs/%x.%j.out" -e "logs/%x.%j.err" \
    --export="$exp" "$HERE/run_bs.sh" run
echo "when the job is done: bash run_bs.sh report"
