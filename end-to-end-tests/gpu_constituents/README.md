# gpu_constituents — end-to-end test for GPU data movement of constituents

Exercises device-side CCPP constituents
(`../../doc/gpu_data_movement_plan.md`, results in
`../../doc/gpu_data_movement_report.md`).  Uses the build and GPU run
scripts in `../gpu_residency` (`TEST_DIR=<this dir>`, see `test.cfg`).

Suite `gpu_const_suite`, three registered constituents (water vapour,
cloud liquid, cloud ice), three time steps in three column chunks:

| Group | Scheme | Memory space | Touches |
|---|---|---|---|
| `phys` | `const_reg` | host | registers the constituents (register phase) |
| `phys` | `gpu_liq` | GPU | cloud liquid and its tendency (one constituent each) |
| `phys` | `cpu_ice` | CPU | cloud ice and its tendency |
| `phys` | `gpu_apply` | GPU | every constituent and tendency at once |
| `diag` | `cpu_check_const` | CPU | every constituent |

Between time steps the driver rescales every constituent through the
host's own pointer (a stand-in for the dycore), and it writes every final
value to `gpu_constituents_out.txt`.

```bash
module swap intel nvhpc
S=../gpu_residency/build.sh
for m in off gpu gpu-debug gpu-trace unified omp omp-debug omp-trace omp-unified; do
  TEST_DIR=$PWD $S $m $B/$m
done
(cd $B/off && ./test_gpu_constituents.x)
qsub -A <account> -v BUILD_ROOT=$B,BUILDS="gpu omp unified",EXE=test_gpu_constituents.x,OUT=gpu_constituents_out.txt \
     ../gpu_residency/run_on_gpu.pbs
```

Every variant must reproduce `off/gpu_constituents_out.txt` bit for bit.
`COMPILER=intel` builds the OpenMP variants with ifx (run with
`OMP_TARGET_OFFLOAD=DISABLED` without an Intel GPU).

Other systems (compilers, `GPU_ARCH`, running without PBS): see
`../gpu_residency/README.md`, "Running on other systems".
