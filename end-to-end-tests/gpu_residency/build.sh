#!/bin/bash
# Build the GPU data-movement end-to-end test.
#
#   build.sh MODE BUILD_DIR
#
# MODE:
#   off        caps generated without --directive, CPU build (reference)
#   cpu        caps generated with --directive acc, CPU build (directives ignored)
#   gpu        caps generated with --directive acc, OpenACC build
#   gpu-debug  as gpu, plus --device-debug runtime checks
#   gpu-trace  as gpu, plus --device-trace and --device-sync-debug
#   omp, omp-debug, omp-trace, omp-unified
#              the same four GPU variants with --directive omp (OpenMP
#              target), built with -mp=gpu and the kernels' OpenMP twins
#   unified    caps generated with --directive acc --memory-model unified,
#              OpenACC build with NVHPC managed memory (UNIFIED_MEM=unified for
#              full unified memory)
#
# Needs nvfortran and nvc++ on PATH (on Derecho: module swap intel nvhpc, in
# a login shell), or ifx and icpx for COMPILER=intel.  FMA is disabled on host
# and device so CPU and GPU results can be compared exactly.  Extra generator
# flags can be passed in CAPGEN_EXTRA.
#
# GPU_ARCH: NVIDIA compute capabilities to build for (NVHPC -gpu=...).
# Default cc80,cc90: one executable for A100 (cc80) and H100/H200/GH200
# (cc90).  Set e.g. GPU_ARCH=cc70 (V100), GPU_ARCH=cc90, or GPU_ARCH=ccnative
# (the GPU on the build node).  See README.md, "Running on other systems".
set -euo pipefail
MODE=$1
BLD=$(mkdir -p "$2" && cd "$2" && pwd)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# TEST_DIR=<dir>: build another test directory that has its own test.cfg
# (default: this script's directory).
HERE=$(cd "${TEST_DIR:-$SCRIPT_DIR}" && pwd)
CAPGEN="$SCRIPT_DIR/../../capgen"
source "$HERE/test.cfg"

# COMPILER=nvhpc (default, nvfortran) or intel (ifx).  ifx has no OpenACC
# and offloads OpenMP only to Intel GPUs (spir64); on a machine without one,
# run its OpenMP builds with OMP_TARGET_OFFLOAD=DISABLED (host fallback).
COMPILER=${COMPILER:-nvhpc}
GPU_ARCH=${GPU_ARCH:-cc80,cc90}
gen_flags=""
if [ "$COMPILER" = intel ]; then
  FC=ifx
  CXX=icpx
  cxx_flags="-O1 -ffp-contract=off -fp-model=precise"
  cxx_link="-lstdc++"
  fc_flags="-O1 -fpp -free -traceback -fp-model=precise -no-fma"
  case "$MODE" in
    off) gen_flags=""; fc_flags="$fc_flags -DNO_DEVICE_SYNC" ;;
    cpu) gen_flags="--directive omp" ;;
    omp)         gen_flags="--directive omp" ;;
    omp-debug)   gen_flags="--directive omp --device-debug" ;;
    omp-trace)   gen_flags="--directive omp --device-trace --device-sync-debug" ;;
    omp-unified) gen_flags="--directive omp --memory-model unified" ;;
    *) echo "mode $MODE is not available with COMPILER=intel"; exit 2 ;;
  esac
  case "$MODE" in omp*) fc_flags="$fc_flags -fiopenmp -fopenmp-targets=spir64 -DCCPP_TEST_OMP" ;; esac
else
FC=nvfortran
CXX=nvc++
cxx_flags="-O1 -Mnofma"
cxx_link="-c++libs"
fc_flags="-O1 -Mpreprocess -Mnofma -Mfree -traceback"
case "$MODE" in
  off)       gen_flags="";                                   fc_flags="$fc_flags -DNO_DEVICE_SYNC" ;;
  cpu)       gen_flags="--directive acc" ;;
  gpu)       gen_flags="--directive acc";                    fc_flags="$fc_flags -acc=gpu -gpu=${GPU_ARCH},nofma -Minfo=accel" ;;
  gpu-debug) gen_flags="--directive acc --device-debug";     fc_flags="$fc_flags -acc=gpu -gpu=${GPU_ARCH},nofma -Minfo=accel" ;;
  gpu-trace) gen_flags="--directive acc --device-trace --device-sync-debug"; fc_flags="$fc_flags -acc=gpu -gpu=${GPU_ARCH},nofma -Minfo=accel" ;;
  unified)   gen_flags="--directive acc --memory-model unified"; fc_flags="$fc_flags -acc=gpu -gpu=${GPU_ARCH},nofma,mem:${UNIFIED_MEM:-managed} -Minfo=accel" ;;
  # OpenMP target backend (--directive omp); kernels use their OpenMP twins.
  omp)         gen_flags="--directive omp";                           fc_flags="$fc_flags -mp=gpu -gpu=${GPU_ARCH},nofma -Minfo=mp -DCCPP_TEST_OMP" ;;
  omp-debug)   gen_flags="--directive omp --device-debug";            fc_flags="$fc_flags -mp=gpu -gpu=${GPU_ARCH},nofma -Minfo=mp -DCCPP_TEST_OMP" ;;
  omp-trace)   gen_flags="--directive omp --device-trace --device-sync-debug"; fc_flags="$fc_flags -mp=gpu -gpu=${GPU_ARCH},nofma -Minfo=mp -DCCPP_TEST_OMP" ;;
  omp-unified) gen_flags="--directive omp --memory-model unified";    fc_flags="$fc_flags -mp=gpu -gpu=${GPU_ARCH},nofma,mem:${UNIFIED_MEM:-managed} -Minfo=mp -DCCPP_TEST_OMP" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
fi


# OVERRIDE=<dir>: any .F90/.meta file in <dir> replaces the file of the same
# name here (used for the deliberately broken --device-debug fixtures).
src() { if [ -n "${OVERRIDE:-}" ] && [ -f "$HERE/$OVERRIDE/$1" ]; then echo "$HERE/$OVERRIDE/$1"; else echo "$HERE/$1"; fi; }
rm -rf "$BLD/ccpp" "$BLD/obj" "$BLD/mod"
mkdir -p "$BLD/ccpp" "$BLD/obj" "$BLD/mod"
scheme_metas=""; for sc in "${SCHEMES[@]}"; do scheme_metas="$scheme_metas$(src $sc.meta),"; done; scheme_metas=${scheme_metas%,}
host_metas=""; for hm in "${HOST_METAS[@]}"; do host_metas="$host_metas$(src $hm),"; done; host_metas=${host_metas%,}
cd "$HERE"
# TIMERS=cesm|fms: generate timers (--timer-style) and link the stand-in
# timing module from timer_stubs/, whose report the driver prints.
timer_srcs=()
if [ -n "${TIMERS:-}" ]; then
  gen_flags="$gen_flags --timer-style $TIMERS"
  fc_flags="$fc_flags -DCCPP_TEST_TIMERS"
  case "$TIMERS" in
    cesm) timer_srcs=("$SCRIPT_DIR/timer_stubs/perf_mod.F90") ;;
    fms)  timer_srcs=("$SCRIPT_DIR/timer_stubs/mpp_mod.F90"); fc_flags="$fc_flags -DCCPP_TEST_TIMERS_FMS" ;;
    *) echo "unknown TIMERS=$TIMERS"; exit 2 ;;
  esac
fi
python3 "$CAPGEN/ccpp_capgen.py" --host-name test_host \
    --host-files "$host_metas" \
    --scheme-files "$scheme_metas" --suites "$HERE/$SUITE" \
    --output-root "$BLD/ccpp" $gen_flags ${CAPGEN_EXTRA:-} > "$BLD/capgen.log" 2>&1 \
  || { cat "$BLD/capgen.log"; exit 1; }

# Optional hook to edit the generated caps (used for negative controls).
if [ -n "${POSTGEN:-}" ]; then (cd "$BLD/ccpp" && eval "$POSTGEN"); fi

DT="$BLD/ccpp/datatable.xml"
deps=$(python3 "$CAPGEN/ccpp_datafile.py" "$DT" --dependencies --separator ' ')
caps=$(python3 "$CAPGEN/ccpp_datafile.py" "$DT" --capgen-files --separator ' ')
srcs=(${timer_srcs[@]+"${timer_srcs[@]}"} $BLD/ccpp/ccpp_kinds.F90 $deps)
for h in "${HOST_SRCS[@]}"; do srcs+=("$(src $h)"); done
# Schemes are Fortran (.F90) or, for language = c++, C++ (.cpp).
cxx_srcs=()
for s in "${SCHEMES[@]}"; do
  if [ -f "$(src $s.F90)" ]; then srcs+=("$(src $s.F90)"); else cxx_srcs+=("$(src $s.cpp)"); fi
done
for f in $caps; do [[ "$f" == */ccpp_kinds.F90 ]] || srcs+=("$f"); done
srcs+=("$(src $DRIVER)")

# Compile until every file builds (module dependencies resolve themselves).
remaining=("${srcs[@]}")
for pass in 1 2 3 4 5 6 7 8; do
  next=()
  for f in "${remaining[@]}"; do
    o="$BLD/obj/$(basename "${f%.*}").o"
    if ! $FC $fc_flags -module "$BLD/mod" -c "$f" -o "$o" > "$BLD/obj/$(basename "$f").log" 2>&1; then
      next+=("$f")
    fi
  done
  [ ${#next[@]} -eq 0 ] && break
  [ ${#next[@]} -eq ${#remaining[@]} ] && { echo "COMPILE FAILED:"; for f in "${next[@]}"; do echo "== $f"; cat "$BLD/obj/$(basename "$f").log"; done; exit 1; }
  remaining=("${next[@]}")
done
# C++ schemes get the same accelerator and macro flags as the Fortran build.
cxx_accel=""
for tok in $fc_flags; do
  case "$tok" in -acc*|-mp*|-gpu=*|-D*|-fiopenmp|-fopenmp-targets=*) cxx_accel="$cxx_accel $tok" ;; esac
done
for f in "${cxx_srcs[@]}"; do
  $CXX $cxx_flags $cxx_accel -I"$BLD/ccpp" -c "$f" -o "$BLD/obj/$(basename "${f%.*}").o" > "$BLD/obj/$(basename "$f").log" 2>&1 \
    || { echo "COMPILE FAILED:"; cat "$BLD/obj/$(basename "$f").log"; exit 1; }
done
link_extra=""; [ ${#cxx_srcs[@]} -gt 0 ] && link_extra="$cxx_link"
$FC $fc_flags -o "$BLD/$EXE" "$BLD"/obj/*.o $link_extra
arch=""; [[ $fc_flags == *-gpu=* ]] && arch=", GPU_ARCH=$GPU_ARCH"
echo "built $BLD/$EXE ($MODE, $COMPILER$arch)"
