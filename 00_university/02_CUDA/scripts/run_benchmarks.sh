#!/usr/bin/env bash
# =============================================================================
# run_benchmarks.sh — Automated benchmark suite for the Heat Diffusion Simulator
#
# Runs the simulation across all available solver modes and grid sizes,
# collects profiling data (Setup / Compute / Comm / Total), and writes a
# tab-separated results file for easy analysis.
#
# Usage:
#   ./scripts/run_benchmarks.sh            # defaults (from project root)
#   ./scripts/run_benchmarks.sh -r 3       # 3 repetitions per config
#   ./scripts/run_benchmarks.sh -o out.tsv # custom output file
# =============================================================================
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
GRID_SIZES=(64 256 1024 2048 4096)
ITERATIONS=1000
REPETITIONS=1
MPI_PROCS=4
RESULTS_FILE=""
BINARY=""

# ── Color helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Argument parsing ────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b <path>     Path to heat_sim binary (default: auto-detect)"
    echo "  -g <sizes>    Comma-separated grid sizes  (default: 64,256,1024,2048,4096)"
    echo "  -i <iters>    Number of simulation iterations (default: 500)"
    echo "  -r <reps>     Repetitions per configuration  (default: 1)"
    echo "  -n <nprocs>   Number of MPI processes         (default: 4)"
    echo "  -o <file>     Output TSV file                 (default: results_<timestamp>.tsv)"
    echo "  -h            Show this help message"
    exit 0
}

while getopts "b:g:i:r:n:o:h" opt; do
    case "$opt" in
        b) BINARY="$OPTARG" ;;
        g) IFS=',' read -ra GRID_SIZES <<< "$OPTARG" ;;
        i) ITERATIONS="$OPTARG" ;;
        r) REPETITIONS="$OPTARG" ;;
        n) MPI_PROCS="$OPTARG" ;;
        o) RESULTS_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# ── Locate project root and binary ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$BINARY" ]]; then
    if [[ -x "$PROJECT_DIR/build/heat_sim" ]]; then
        BINARY="$PROJECT_DIR/build/heat_sim"
    else
        echo -e "${RED}Error:${RESET} heat_sim binary not found in build/."
        echo "       Build the project first:  mkdir build && cd build && cmake .. && make -j\$(nproc)"
        exit 1
    fi
fi

if [[ ! -x "$BINARY" ]]; then
    echo -e "${RED}Error:${RESET} $BINARY is not executable."
    exit 1
fi

# ── Detect available modes by probing compile definitions ───────────────────
MODES=("seq")

# Probe helper: runs with a small valid grid and checks for success
probe_mode() {
    local cmd="$1"
    $cmd 2>&1 | grep -q "Simulation complete"
}

# Check for OpenMP support
if probe_mode "$BINARY 4 1 omp"; then
    MODES+=("omp")
fi

# Check for CUDA support
if probe_mode "$BINARY 4 1 cuda"; then
    MODES+=("cuda")
fi

# Check for hybrid CUDA+OpenMP support
if probe_mode "$BINARY 4 1 hybrid"; then
    MODES+=("hybrid")
fi

# Check for MPI support (need mpirun available)
HAS_MPI=false
if command -v mpirun &>/dev/null; then
    if probe_mode "mpirun --oversubscribe -np 2 $BINARY 8 1 mpi_blocking"; then
        HAS_MPI=true
        MODES+=("mpi_blocking" "mpi_nonblocking")
    fi
fi

# ── Prepare output ──────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$RESULTS_FILE" ]]; then
    RESULTS_FILE="$PROJECT_DIR/results_${TIMESTAMP}.tsv"
fi

# TSV header
echo -e "mode\tgrid_size\titeration\tsetup_s\tcompute_s\tcomm_s\ttotal_s" > "$RESULTS_FILE"

# ── Helper: parse profiling output ──────────────────────────────────────────
parse_result() {
    local output="$1"
    local setup compute comm total

    setup=$(echo "$output"  | grep "Setup Time:"   | awk '{print $(NF-1)}')
    compute=$(echo "$output" | grep "Compute Time:" | awk '{print $(NF-1)}')
    comm=$(echo "$output"   | grep "Comm Time:"    | awk '{print $(NF-1)}')
    total=$(echo "$output"  | grep "Total Time:"   | awk '{print $(NF-1)}')

    echo "${setup:-N/A} ${compute:-N/A} ${comm:-N/A} ${total:-N/A}"
}

# ── Print banner ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║        🔥  Heat Diffusion Simulation — Benchmarks  🔥       ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${CYAN}Binary:${RESET}       $BINARY"
echo -e "  ${CYAN}Grid sizes:${RESET}   ${GRID_SIZES[*]}"
echo -e "  ${CYAN}Iterations:${RESET}   $ITERATIONS"
echo -e "  ${CYAN}Repetitions:${RESET}  $REPETITIONS"
echo -e "  ${CYAN}Modes:${RESET}        ${MODES[*]}"
if $HAS_MPI; then
    echo -e "  ${CYAN}MPI procs:${RESET}    $MPI_PROCS"
fi
echo -e "  ${CYAN}Output:${RESET}       $RESULTS_FILE"
echo ""

# ── Table header ────────────────────────────────────────────────────────────
ROW_FMT="  %-16s │ %8s │ %3s │ %14s │ %14s │ %14s │ %14s"
DIVIDER=$(printf "$ROW_FMT" "" "" "" "" "" "" "" | sed 's/ /─/g; s/│/┬/g')
HEADER=$(printf "  ${BOLD}%-16s${RESET} │ ${BOLD}%8s${RESET} │ ${BOLD}%3s${RESET} │ ${BOLD}%14s${RESET} │ ${BOLD}%14s${RESET} │ ${BOLD}%14s${RESET} │ ${BOLD}%14s${RESET}" \
    "Mode" "Grid" "Run" "Setup (s)" "Compute (s)" "Comm (s)" "Total (s)")

echo "$DIVIDER"
echo "$HEADER"
echo "$DIVIDER"

# ── Run benchmarks ──────────────────────────────────────────────────────────
TOTAL_RUNS=$(( ${#MODES[@]} * ${#GRID_SIZES[@]} * REPETITIONS ))
CURRENT_RUN=0

for mode in "${MODES[@]}"; do
    for n in "${GRID_SIZES[@]}"; do

        # Skip small grids for MPI (domain decomposition needs minimum size)
        if [[ "$mode" == mpi_* && "$n" -lt 64 ]]; then
            continue
        fi

        for rep in $(seq 1 "$REPETITIONS"); do
            CURRENT_RUN=$((CURRENT_RUN + 1))

            # Build command
            if [[ "$mode" == mpi_* ]]; then
                CMD="mpirun --oversubscribe -np $MPI_PROCS $BINARY $n $ITERATIONS $mode"
            else
                CMD="$BINARY $n $ITERATIONS $mode"
            fi

            # Execute and capture output
            OUTPUT=$($CMD 2>&1) || {
                echo -e "  ${RED}✗${RESET} ${mode} N=${n} rep=${rep} — ${YELLOW}FAILED${RESET}"
                continue
            }

            # Parse timings
            read -r setup compute comm total <<< "$(parse_result "$OUTPUT")"

            # Write TSV row
            echo -e "${mode}\t${n}\t${rep}\t${setup}\t${compute}\t${comm}\t${total}" >> "$RESULTS_FILE"

            # Pretty-print table row
            printf "${ROW_FMT}\n" \
                "$mode" "${n}" "${rep}" "$setup" "$compute" "$comm" "$total"
        done
    done
done

echo "$DIVIDER"
echo ""
echo -e "${GREEN}✓${RESET} Benchmark complete. ${BOLD}${TOTAL_RUNS}${RESET} configurations executed."
echo -e "  Results saved to: ${CYAN}${RESULTS_FILE}${RESET}"
echo ""
