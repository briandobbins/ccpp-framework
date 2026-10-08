# cxx_schemes — a Fortran host calling C++ schemes

Exercises `language = c++` scheme metadata (`capgen/generator/cxx_schemes.py`)
together with GPU data movement.  Uses the build and GPU run scripts in
`../gpu_residency` (`TEST_DIR=<this dir>`, see `test.cfg`) and that test's
Fortran host, driver and GPU-ported Fortran schemes.

| Group | Scheme | Language | Memory space |
|---|---|---|---|
| `before` | `gpu_heat` (in a subcycle) | Fortran, OpenACC/OpenMP | GPU |
| `before` | `cxx_adjust` (in a subcycle) | **C++** | CPU |
| `before` | `gpu_moist` | Fortran, OpenACC/OpenMP | GPU |
| `after` | `gpu_finish` | Fortran, OpenACC/OpenMP | GPU |
| `after` | `cxx_micro` | **C++, OpenACC/OpenMP** | GPU |
| `after` | `cxx_check` | **C++** | CPU |

`cxx_micro` is a made-up microphysics scheme: a saturation polynomial, a
condensation limiter, a vertical recurrence per column, latent heating, a
square root and a column sum into precipitation.  It uses only correctly
rounded operations, so with FMA off its GPU result matches its CPU result
bit for bit.

The caps reach the C++ schemes through generated `BIND(C)` interface blocks;
`<scheme>_ccpp.h` (written next to the caps) holds the C prototypes the
schemes include.  `cxx_micro`'s 2-D arrays are passed through dense
scratch arrays the caps fill and empty on the device, so the C++ kernel
sees a dense column-major chunk that is already present on the GPU.

```bash
module swap intel nvhpc
S=../gpu_residency/build.sh
for m in off cpu gpu gpu-debug gpu-trace unified omp omp-debug omp-trace omp-unified; do
  TEST_DIR=$PWD $S $m $B/$m
done
(cd $B/off && ./test_cxx_schemes.x)
qsub -A <account> -v BUILD_ROOT=$B,BUILDS="gpu omp unified",EXE=test_cxx_schemes.x ../gpu_residency/run_on_gpu.pbs
```

Every variant must reproduce `off/gpu_residency_out.txt` bit for bit.
`COMPILER=intel` builds with ifx/icpx (OpenMP variants: run with
`OMP_TARGET_OFFLOAD=DISABLED` without an Intel GPU).  `OVERRIDE=fixtures/cxx_error`
makes `cxx_check` report an error; the driver must print it intact.

Other systems (compilers, `GPU_ARCH`, running without PBS): see
`../gpu_residency/README.md`, "Running on other systems".
