# C++ scheme argument types

A Fortran host calling a host-side C++ scheme (`cxx_types`) through every
argument type added for C++ schemes:

| Argument | Type | What crosses the boundary |
|---|---|---|
| `use_extra` | logical scalar, `in` | a byte holding exactly 0 or 1; C++ sees `bool` |
| `mask` | logical array, `out` | a 0/1 byte array, converted back to `logical` by the cap |
| `extra` | optional real array, `in` | `NULL` when the host's `active` condition is false |
| `state` | derived type `gpu_state_type`, `inout` | one generated descriptor `ccpp_gpu_state_type_t`: member pointers, shapes, lower bounds |
| `work` | `allocatable` real array, `out` | an allocation handle; C++ calls `work->allocate(work->handle, lower, upper)`, which allocates in Fortran |

`cxx_types_gpu` is the GPU-side counterpart (`memory_space = device`).  It
takes the logical scalar, the optional `extra` array, `temp` and the derived
type `state`.  From OpenACC or OpenMP kernels it writes a logical array
`warm`, a logical scalar `any_warm`, and an array `gwork` that it allocates
itself; the allocation callback creates the device copy.  `gwork` uses
`state%precl`, which `cxx_types` has just written on the host, so a missing
update of the device copy shows up as a wrong `gwork`.  Its logical array and optional array go through device
scratch arrays, and the optional one is copied only when present.  Build it
with `gpu`, `omp` or `gpu-debug` and run with `run_on_gpu.pbs`: results
must match this test's CPU build bit for bit.

`ftn_types_check` (Fortran) runs next and checks both schemes.  It recomputes every result: the
mask from the updated temperature; `work`, which depends on `ps` and on
`use_extra`; and `precl`, written through the descriptor.  Any mismatch
fails the run.  It then flips `use_extra`, so `extra` is alternately present
and absent in successive calls; both sides check that presence agrees with
the flag.  `fixtures/bad_mask/` inverts the mask in C++ and must fail with
`mask wrong`.

Build with the GPU test's script (any mode; the C++ scheme is host-side):

    TEST_DIR=$PWD ../gpu_residency/build.sh off /path/to/build      # NVHPC
    COMPILER=intel TEST_DIR=$PWD ../gpu_residency/build.sh off ...  # Intel
    OVERRIDE=fixtures/bad_mask TEST_DIR=$PWD ../gpu_residency/build.sh off ...

Compiler issues found and worked around in the generator (see
`capgen/generator/cxx_schemes.py`):

* `logical(c_bool)` is not a safe C `bool` with NVHPC 25.9 or Intel ifx: both
  store `.true.` as 0xFF.  Logicals therefore cross as 0/1 bytes.
* NVHPC 25.9 and 26.1 pass invalid F2018 C descriptors for allocatable
  `BIND(C)` dummies when the Fortran side is compiled for GPU offload (and
  for `intent(out)` ones even without), so allocatable arguments use a
  generated allocation handle with a Fortran callback instead.

Other systems (compilers, `GPU_ARCH`, running without PBS): see
`../gpu_residency/README.md`, "Running on other systems".
