# gpu_residency — end-to-end test for GPU data movement

Exercises the GPU data-movement proof of concept
(`../../doc/gpu_data_movement_plan.md`, results in
`../../doc/gpu_data_movement_report.md`).  Not part of the CMake
end-to-end suite: it needs nvfortran and, for the GPU variants, a GPU.

## What it covers

Suite `gpu_residency_suite`, two groups, run for three time steps in three
column chunks, with host code changing data between the groups:

| Group | Items | Memory space |
|---|---|---|
| `before` | subcycle ×2 of `gpu_heat`, `cpu_adjust`; then `gpu_moist` | GPU, CPU, GPU |
| `after` | `gpu_finish`, `cpu_check` | GPU, CPU |

* `temp` — `host_access = read_write` (default); the driver changes it between the groups.
* `qv` — `host_access = none`; stays on the device, read once on the CPU by `cpu_check`.
* `phys_state%ps` — a derived-type member with `host_access = read`.
* `phys_state%precl` — written on the device in `before`, read by the host between groups.
* `extra` — an optional argument, active only when `use_extra` is true.
* `work` — an interstitial written by a CPU scheme inside the subcycle and read on the device after it.

The GPU schemes' kernels use `default(present)`: they assume their arguments
are already on the device.  The generated caps make that true.

## Build and run

```bash
module swap intel nvhpc            # Derecho; elsewhere see "Running on other systems"
./build.sh off       $B/off        # CPU reference, caps without --directive
./build.sh cpu       $B/cpu        # caps with --directive acc, CPU build
./build.sh gpu       $B/gpu        # OpenACC build
./build.sh gpu-debug $B/gpu-debug  # plus --device-debug runtime checks
./build.sh gpu-trace $B/gpu-trace  # plus --device-trace and --device-sync-debug
./build.sh unified   $B/unified    # --memory-model unified, NVHPC managed memory
./build.sh omp       $B/omp        # OpenMP target backend (also omp-debug, omp-trace, omp-unified)
COMPILER=intel ./build.sh omp $B/omp-ifx   # ifx, OpenMP offload for Intel GPUs (spir64)
(cd $B/off && ./test_gpu_residency.x)
qsub -A <account> -v BUILD_ROOT=$B,BUILDS="gpu gpu-debug gpu-trace unified" run_on_gpu.pbs
```

Every variant must reproduce `off/gpu_residency_out.txt` bit for bit
(FMA is disabled on host and device).

The GPU kernels have OpenACC and OpenMP versions, selected by
`CCPP_TEST_OMP`.  ifx has no OpenACC and offloads only to Intel GPUs; without
one, run its OpenMP builds with `OMP_TARGET_OFFLOAD=DISABLED` and compare
with `COMPILER=intel ./build.sh off`.  Host fallback cannot catch the
`bad_intent` fixture: host and device are then the same memory.

## Running on other systems

The commands above are for NCAR's Derecho (A100 GPUs, PBS).  Elsewhere,
four things may need to change; the generated code does not.

* **Compilers.**  Put NVHPC's `nvfortran` and `nvc++` on `PATH` (module
  names vary by site; on Derecho `module swap intel nvhpc`).  NVHPC 25.9
  and 26.1 have been used.  `COMPILER=intel` uses `ifx`/`icpx` instead,
  for the `off`, `cpu` and OpenMP modes only (Intel GPUs, or host fallback).
* **GPU architecture.**  `GPU_ARCH` sets NVHPC's `-gpu=` target.  The
  default, `cc80,cc90`, builds one executable for both A100 and H100/H200/
  GH200.  For other GPUs, set it to the compute capability that
  `nvidia-smi --query-gpu=compute_cap --format=csv` reports, e.g.
  `GPU_ARCH=cc70` for a V100, or use `GPU_ARCH=ccnative` when building on
  the GPU node itself.  A build for the wrong capability fails at run time
  with a "no kernel image" style error.
* **Running.**  `run_on_gpu.pbs` is a plain bash script with Derecho's PBS
  settings at the top.  On another PBS system, change its queue and
  `select` line.  Anywhere else, run it directly on a GPU node, e.g. in a
  Slurm allocation:
  ```bash
  BUILD_ROOT=$B BUILDS="gpu gpu-debug omp" bash run_on_gpu.pbs
  ```
  with NVHPC's runtime libraries on `LD_LIBRARY_PATH`.
* **Unified memory.**  The `unified` and `omp-unified` modes default to
  NVHPC managed memory (`mem:managed`), which works on any NVIDIA GPU.  On
  systems with full CPU–GPU unified memory (Grace Hopper GH200, or Linux
  HMM), `UNIFIED_MEM=unified` tests that mode instead.

Keep FMA off (`-Mnofma`, `-gpu=...,nofma`, already in `build.sh`) when
changing flags: the bit-for-bit comparison with the CPU build depends on it.

## Negative controls

`OVERRIDE=fixtures/<name>` swaps in deliberately broken files (see
`fixtures/README`):

* `bad_intent` — a GPU scheme writes an `intent(in)` argument.
* `bad_host_access` — the host breaks its `host_access = read` promise.
* `bad_realloc` — the host reallocates an array after `ccpp_init`.  With
  `gpu-debug` it stops cleanly; with `gpu-trace` it warns before the
  failing statement.

Built with `gpu`, the first two give silently wrong answers; built with
`gpu-debug` they stop with an error naming the array and the call.  A third control,
stripping every generated update (`POSTGEN="sed -i '/!\$acc update/d' *_cap.F90"`),
also gives wrong answers, showing the comparison detects missing transfers.

## Other tests

`build.sh` reads the test definition from `test.cfg`; `TEST_DIR=<dir>`
builds another test directory with its own `test.cfg`
(`../gpu_constituents`).  `run_on_gpu.pbs` takes `EXE` and `OUT` for the
other test's executable and output file.
