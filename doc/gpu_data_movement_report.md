# GPU data movement and C++ schemes in capgen v1 — proof-of-concept report

*2026-10-05.  Built locally on Derecho in a `feature/capgen-v1` working
copy; nothing is committed.  Design: `doc/gpu_data_movement_plan.md`.
Open items: `doc/followups.md` FU-036 and FU-038 to FU-049 (FU-037 and
FU-040 closed by this work).*

## 1. Summary

capgen v1 can generate every host↔device transfer for a suite that mixes
GPU-ported and CPU schemes, from two new metadata attributes.  Scheme
kernels are written as if their arrays were already on the device
(`default(present)` in OpenACC, implicit mapping in OpenMP); the generated
caps make that true and keep the host copy correct wherever CPU code reads
it.

* Two backends from one plan: OpenACC (`--directive acc`) and OpenMP target
  (`--directive omp`).
* Two test suites run on an A100 and reproduce the CPU result **bit for
  bit** in every variant — OpenACC, OpenMP, unified memory, and with each
  debug mode on.  One covers chunked columns, two groups with host code
  between them, a subcycle, an interstitial, derived-type members and an
  optional argument; the other covers **CCPP constituents**, including
  single-constituent and all-constituent access in the same group.
* Without `--directive`, the generated code was **byte-identical** to
  capgen v1's for all 12 existing end-to-end test directories when first
  built; all 1500 existing unit tests behave as before.  Later changes
  that are on by default alter some generated code (see "Changes for
  existing capgen v1 users" in the repository `README.md`): the
  register-phase constituent collection is rewritten, so 3 of the
  end-to-end suite caps now differ in text but not in behaviour.
* The change to existing capgen code is about 500 added lines across 8
  files, plus three new modules (about 1,900 lines) and 72 unit tests.
* Three debug modes: automatic detection of wrong data (stops at the exact
  call), a crash-investigation trace, and a host-mirroring sync mode.
* The OpenMP output also compiles with Intel `ifx` for Intel GPUs (run in
  host-fallback mode here; Derecho has no Intel GPU).
* A Fortran host can call **C++ schemes** (`language = c++`), on the CPU or
  on the GPU with the same data movement, ported from xdsl-ccpp and
  compiled and run here (xdsl-ccpp only generated this direction).

## 2. How it is used

**Scheme metadata.**  Mark where the scheme touches its arrays, per
argument or once for a whole phase:

```ini
[ccpp-arg-table]
  name = kessler_run
  type = scheme
  memory_space = device        # default for every argument of this phase
[ precl ]
  ...
  memory_space = host          # per-argument override
```

Constituents need nothing extra: a device-side argument that resolves to
the constituent state (`ccpp_constituents`, `ccpp_constituent_tendencies`,
or a single constituent or its tendency) is handled like any other array.

**Host metadata.**  Optionally say what host code does with a variable
between CCPP calls (the coupler between physics groups, the dycore between
steps).  Omitting it is always correct, just slower:

| `host_access` | Host promises | Valid copy whenever the host has control |
|---|---|---|
| `read_write` (default) | nothing | host |
| `read` | it reads but never writes the CPU copy | host and device |
| `none` | it never touches it | device |

A host variable marked `memory_space = device` is one the host keeps on the
GPU itself; CCPP never allocates or moves it.  Constituents always use
`read_write`, since dycores advect them between physics calls (FU-045).

**Host code.**  Two entry points appear in the host cap:
`ccpp_physics_device_update_device`, called after the host fills initial
data (it copies the `none` and `read` arrays), and
`ccpp_physics_device_update_host`, called before the host reads `none`
data for output or restarts (it copies only those).  All other arrays are
valid on the host whenever the host has control, and these routines leave
them alone.

Two rules for hosts: arrays used on the device must be allocated before
`ccpp_init` and not reallocated afterwards (the debug and trace modes
detect a violation); and variables private to a scheme's own module that
its kernels use are the scheme's own responsibility (`declare create` and
an update in its init), since no generator can see them.

**Generator flags.**

| Flag | Effect |
|---|---|
| `--directive acc\|omp` | turn the feature on with OpenACC or OpenMP target directives |
| `--memory-model unified` | hardware-coherent memory (GH200 / MI300A class): no copies, only waits where CPU code reads GPU results |
| `--device-debug` | presence checks before GPU calls; host/device content checks wherever both copies should agree; stops with the array and call |
| `--device-trace` | crash investigation (§4) |
| `--device-sync-debug` | copy every GPU result to the host right after the call that produced it |

**C++ schemes.**  A scheme implemented in C or C++ (with C linkage)
declares `language = c++` in its `[ccpp-table-properties]`.  The group caps
then reach it through a generated `BIND(C)` interface block instead of a
`use`, and capgen writes `<scheme>_ccpp.h` next to the caps with the exact
C prototypes:

```c
void cxx_micro_run(
    int ncol,
    int nlev,
    double* temp,  /* [device] */
    double* qv,  /* [device] */
    const double* ps,  /* [device] */
    double* precl,  /* [device] */
    char* errmsg,
    int* errflg);
```

Scalars with `intent(in)` pass by value, everything else by pointer; real
kinds map through the kind table (`REAL64` → `double`, `REAL32` →
`float`); arrays are dense and column-major; `errmsg` arrives blank-padded
to the host length, not NUL-terminated.  A non-contiguous column chunk is
passed as a dense copy on the CPU (the compiler makes it) and, for
device-side arguments, through a dense scratch array the caps fill and
empty on the GPU, so a C++ kernel always sees a dense array that is already
present on the device (`present()` in OpenACC, an existing mapping in
OpenMP).  Host-side C++ schemes can also take logical, derived-type,
optional and allocatable arguments (§2.1 below); the C++-host direction is
not ported (FU-047).

### 2.1 Logical, derived-type, optional and allocatable arguments (2026-10-06)

| Type | Fortran side | C++ side |
|---|---|---|
| logical | the cap converts to a byte holding exactly 0 or 1 | `bool` / `bool*` |
| derived type `T` | the cap fills one generated `bind(C)` descriptor `ccpp_T_t` (a pointer, shape and lower bounds per integer/real member) | one `ccpp_T_t*`, struct defined in the scheme header |
| optional | F2018 `optional` dummy; absent when the host's `active` condition is false | pointer, `NULL` when absent |
| allocatable | the cap moves the array into a local box and passes an allocation handle; a generated Fortran function does the `allocate` | `ccpp_alloc_T_R_t*`: current `data`/`shape`/`lbound`, and `allocate(handle, lower, upper)` returning the new data |

Derived types are not flattened: a type costs one argument however many
members it has, and the C++ code does not depend on the Fortran compiler's
layout of the type.  Three compiler problems shaped the design:

* NVHPC 25.9 and Intel ifx store `.true.` as 0xFF even in
  `logical(c_bool)`, which is not a valid C `bool`.  An Intel test computing
  `flag ? 1.0 : 0.0` got 255.
* NVHPC 25.9 and 26.1 pass invalid F2018 C descriptors (`CFI_cdesc_t`) for
  allocatable `BIND(C)` dummies whenever the Fortran file is compiled for GPU
  offload (`-acc=gpu`, `-mp=gpu`), as the caps are in GPU builds.  Without
  offload, they do so for `intent(out)` dummies, and they also reject
  keyword arguments for such dummies.  So allocatables do not use C
  descriptors.  The callback design above works with NVHPC (with and
  without `-acc`) and Intel (with and without OpenMP offload).

Test `end-to-end-tests/cxx_arg_types`: a Fortran checker recomputes every
result, and passes with NVHPC and Intel.  The optional argument is
exercised both present and absent.  A broken-on-purpose fixture fails as
it should.

**On the GPU side** (`memory_space = device`), C++ schemes take logical
scalars (any intent), logical arrays and optional arguments as well.
Logical arrays go through the dense device scratch arrays and are converted
to and from 0/1 bytes in device loops.  Optional arrays are copied only
when present; an absent argument arrives as NULL.

They take derived types and allocatable arguments too (2026-10-07):

* **Derived types.**  Each array member a GPU-side descriptor exposes is
  tracked like an ordinary host array: a device copy is created at init,
  and updates are planned from the argument's intent.  The descriptor keeps
  host addresses, and kernels use `present`.  Host-side C++ derived-type
  arguments are also tracked member by member, so a host write through a
  descriptor is seen.
* **Allocatable arguments.**  A device variant of the allocation callback
  creates the new array's device copy, deleting the old one first.  The
  array is tracked as a scheme-allocated interstitial: its updates and
  checks are guarded by `allocated()`, and `<suite>_final` deletes the
  device copy.

The test's GPU-side scheme `cxx_types_gpu` covers all of these with
OpenACC and OpenMP kernels.  It reads the derived type's `precl`, which the
host-side C++ scheme has just written through its own descriptor, and
allocates and fills `gwork` on the device.  The OpenACC, OpenMP and
OpenACC-with-checks builds match the CPU build bit for bit on an A100.  A
negative control that drops the generated `precl` update gives a wrong
`gwork`, and the checker reports it.  Remaining limit (FU-048): device-side
derived types with allocatable members.

**Planner fix found on the way.**  A phase whose only GPU work was a C++
call's scratch copies (no updates or checks) was treated as empty, so the
copies were never generated and the C++ kernel would have received host
addresses.  Such phases are now kept (`PhasePlan.is_empty`).  None of the
existing tests had hit it.

## 3. How it works

* **Allocate once.**  Each array that some device-side argument touches
  gets one device copy, created at the end of `<suite>_init` (runs once,
  outside any threaded region) and deleted in `<suite>_final`; derived-type
  parents (host DDTs, the interstitial container, the constituent object)
  are created first.  No reference-count pairing across phases, no
  threading races.
* **Then only update.**  All later movement is an update on exactly the
  slice a call uses.  Updates allocate nothing, so threads working on
  different column chunks do not interfere.
* **Groups are independent.**  Every group phase starts from each array's
  handoff state and restores it before returning.  Call order between
  groups, skipped groups and `group_name='all'` need no special handling.
  Inside a group the planner walks the calls in order and copies only
  where a call reads a stale copy; subcycles use a fixed-point analysis
  that moves a copy outside the loop when the loop keeps it valid.
* **Overlapping slices.**  Constituent arrays are reached one constituent
  at a time and all at once.  They are tracked as one unit per column
  chunk; a single-constituent write is a partial write, so the rest of the
  unit is brought up to date on that side first.
* **Planner and printers are separate.**  `generator/device_residency.py`
  builds a directive-neutral plan; `generator/device_emit.py` prints it
  through one spelling table per directive language.  The existing writers
  call into them only when a plan exists.

| Statement | OpenACC | OpenMP target |
|---|---|---|
| create device copy | `enter data create(x)` | `target enter data map(alloc: x)` |
| delete device copy | `exit data delete(x)` | `target exit data map(delete: x)` |
| copy to device | `update device(x)` | `target update to(x)` |
| copy to host | `update self(x)` | `target update from(x)` |
| wait (unified mode) | `wait` | `taskwait` |
| presence query (debug) | `acc_is_present(x)` | generated `ccpp_gpu_is_present(x)` |

The OpenMP presence helper exists because `omp_target_is_present` takes a
C address and `c_loc` is only legal on a TARGET variable (ifx enforces
this); the helper's assumed-type, assumed-rank TARGET dummy makes it
standard-conforming for any host array.

Generated code for the constituent test's physics group:

```fortran
    ! gpu_liq updates one constituent and writes its tendency
    !$acc update device(temp(lb:ub, 1:pver))
    !$acc update device(ccpp_model_constituents_obj(1)%vars_layer(lb:ub, 1:pver, :))
    !$acc update device(ccpp_model_constituents_obj(1)%vars_layer_tend(lb:ub, 1:pver, :))
    call gpu_liq_run(..., cld_liq=...%vars_layer(lb:ub, 1:pver, index_of_cloud_liquid_dry_mixing_ratio), ...)
    ! cpu_ice reads on the host
    !$acc update self(ccpp_model_constituents_obj(1)%vars_layer(lb:ub, 1:pver, :))
    !$acc update self(ccpp_model_constituents_obj(1)%vars_layer_tend(lb:ub, 1:pver, :))
    call cpu_ice_run(...)
    ! gpu_apply applies every constituent's tendency at once
    !$acc update device(...%vars_layer_tend(lb:ub, 1:pver, :))
    !$acc update device(...%vars_layer(lb:ub, 1:pver, :))
    call gpu_apply_run(..., const_tend=...%vars_layer_tend(lb:ub, 1:pver, :), const=...%vars_layer(lb:ub, 1:pver, :))
```

## 4. Debugging

| Need | Mode | What it does |
|---|---|---|
| Is the data wrong? | `--device-debug` | Before each GPU call, checks the arrays have device copies.  Wherever the plan says host and device copies agree — after a call that only reads an array, at group exit, at group entry for `host_access = read` — copies the device slice back and compares; stops with errflg naming the array, the call and the reason. |
| Why did it crash? | `--device-trace` | Before each GPU call, one line per array: the plan's expectation (valid on host / device / both), presence of the whole array **and of the slice's first and last elements**, slice shape, array bounds, values of the variables in the slice bounds, and the slice's first element **read on the host and on the device**; one line with the call's scalar arguments.  Before every copy, the copied slice's shape and a warning if the array has no device copy.  After each device copy is created, its presence and bounds.  An explicit warning before an access that will fail. |
| Keep host memory current | `--device-sync-debug` | Copies every GPU result back right after the call that produced it; the plan then skips later copies the mirror makes unnecessary. |

Example: a host that reallocates `temp` after `ccpp_init` (fixture
`bad_realloc`).  With `--device-trace`, OpenACC prints, immediately before
the abort:

```
ccpp-gpu WARNING: temp has no device copy at this copy to device; was it reallocated after ccpp_init? the runtime will abort here
```

OpenMP prints the same warning and the trace line
`... present (array, first, last): F F F ...`, then finishes correctly but
slowly, because OpenMP silently copies unmapped arrays into every kernel.
With `--device-debug` both stop cleanly:
`temp has no device copy before gpu_heat_run`.

**Compared with xdsl-ccpp's `--gpu-debug-prints` / `--gpu-debug-sync`:**

| Capability | This proof of concept | xdsl-ccpp |
|---|---|---|
| Whole-array presence before GPU calls | yes | yes |
| Presence of the exact slice | yes, first and last elements, without risking an implicit copy or abort | yes, via a `present()` region, which aborts if absent |
| Array sizes and bounds | slice shape, array bounds, slice-bound variable values | sizes and bound values, read inside device regions |
| A value read on the device | first element of each slice, host and device side by side | module scalars, host and device |
| Scalar arguments of GPU calls | yes | n/a (scalars go into kernels by value here) |
| Where the plan expects valid data | yes | no |
| Warning before the failing statement | yes, at copies and calls, with likely cause | no |
| Clean error instead of a crash | yes (`--device-debug`) | no |
| Content check: host vs device copies agree | yes (`--device-debug`) | no |
| Host-mirroring sync mode | yes | yes |
| Works with OpenMP and ifx | yes | not checked |

## 5. Results

GPU runs on one NVIDIA A100-SXM4-40GB, NVHPC 25.9, FMA disabled on host and
device so that results can be compared exactly.

| Check | Result |
|---|---|
| Existing unit tests | 1500 pass; 1 pre-existing failure on this filesystem (FU-044), also before any change |
| New unit tests (`unit-tests/test_device_residency.py`) | 72 pass |
| Flag-off output, all 12 end-to-end test directories | byte-identical to the unchanged generator when first built; later, 3 suite caps differ in their (rewritten) constituent collection, and all 13 end-to-end tests pass |
| `gpu_residency` test: OpenACC and OpenMP, each plain / debug / trace+sync / unified (managed memory) | all 8 match the CPU build bit for bit; 54 kernels launched |
| `gpu_constituents` test: the same 8 variants | all match the CPU build bit for bit; 18 kernels launched |
| Negative controls: generated updates stripped | wrong answers in both tests |
| Fixture `bad_intent`: GPU scheme writes an `intent(in)` array | without debug silently wrong; with debug stops after `gpu_heat_run` naming `phys_state%ps` (OpenACC and OpenMP) |
| Fixture `bad_host_access`: host breaks `host_access = read` | without debug silently wrong; with debug stops at entry to group `after` naming `temp` (OpenACC and OpenMP) |
| Fixture `bad_realloc`: host reallocates after `ccpp_init` | debug stops cleanly naming `temp`; trace warns before the failure (OpenACC and OpenMP) |
| Intel `ifx -fiopenmp -fopenmp-targets=spir64`, both tests | compile and link; in host-fallback mode match the ifx CPU build bit for bit; ifx and nvfortran CPU references are identical |
| Full unified memory | not supported on Derecho's A100 platform (needs GH200/MI300A class; FU-042) |
| `cxx_schemes` test (Fortran host, two CPU C++ schemes, one GPU-ported C++ scheme with OpenACC/OpenMP, three GPU-ported Fortran schemes): 8 GPU variants | all match the test's CPU build bit for bit; the C++ kernel runs on the GPU (9 launches) with no transfers of its own; the C++ CPU schemes alone reproduce the all-Fortran reference bit for bit |
| C++ error path | an error set in C++ reaches the Fortran host intact |
| `cxx_schemes` with Intel `ifx`/`icpx` | all variants match the Intel CPU build; Intel and NVIDIA references differ in 10 of 660 values by one rounding step (2.3e-16 relative), from the two C++ compilers' arithmetic |

**A bug the constituent test found.**  The first version of the host sync
routines copied every tracked array.  `ccpp_physics_device_update_host`
therefore overwrote host values the host had changed after the last
physics call with stale device data.  The `gpu_residency` test missed it
because its driver never changes data after the last call; the constituent
test's driver does, and its results differed from the CPU build by exactly
one host update.  The routines now copy only the arrays that need it (§2).

**Transfer counts** (`gpu_residency`, 3 steps × 3 column chunks): 99
transfers inside physics calls, measured with `NV_ACC_NOTIFY`.  Wrapping
every GPU call in its own copy-in/copy-out region would need 153 for the
same run (computed from the metadata).  No performance tuning has been
done (FU-041).

**Comparison run: xdsl-ccpp's Kessler test.**  On the same node type and
compiler, xdsl-ccpp's own CPU-versus-GPU parity test passes, but as
configured its "GPU" build compiles with `-mp=gpu` while the scheme and
caps use `!$acc`, which NVHPC 25.9 ignores without `-acc`; it runs entirely
on the CPU (and its link fails without an extra flag).  With `-acc` added
it runs on the GPU (5 kernels, 40 transfers) and passes.  Its generated
code is sound; its build configuration is not.

## 6. Limits of the proof of concept

Each is a clear generation-time error today, not a silent failure:

* unit, kind or vertical-flip transforms on device-side arguments (FU-038);
* instance- or thread-indexed host variables, scheme-allocated
  interstitials, derived-type device arguments (FU-039);
* multi-instance constituent objects on the device (FU-046);
* constituents always use the `read_write` handoff (FU-045);
* no real-suite or GH200/MI300A/Intel-GPU testing yet (FU-042) —
  CAM-SIMA's registry generator still emits capgen v0 host metadata, so
  this waits on the CAM-SIMA transition (FU-034);
* no validator checks yet for non-assumed-shape dummies in device phases
  or for schemes that manage their own data regions (FU-043);
* no performance work (FU-041), including the device scratch copies for
  C++ scheme arguments (FU-049);
* C++ schemes: the new argument types are host-side only, and some
  combinations remain unsupported (FU-048); a C++ host calling Fortran
  schemes is not ported (FU-047).

## 7. What changed

| File | Change |
|---|---|
| `capgen/metadata/metadata_table.py` | `memory_space`, `host_access` variable attributes; `memory_space` arg-table header default; `language` table property; table-type scoping |
| `capgen/metadata/variable_resolver.py` | carry both onto host entries (derived-type members inherit); per-phase default lookup |
| `capgen/ccpp_capgen.py` | five flags; build the plan per suite |
| `capgen/generator/suite_resolver.py` | `device_plan` field on the suite resolution |
| `capgen/generator/group_cap.py` | hooks for per-call, subcycle and group-exit updates, checks and traces |
| `capgen/generator/suite_cap.py` | create/delete device copies; per-suite sync routines |
| `capgen/generator/host_cap.py` | host sync entry points |
| `capgen/generator/datatable.py` | `<device_residency>` section |
| `capgen/generator/device_residency.py` (new) | the planner |
| `capgen/generator/device_emit.py` (new) | Fortran printers: OpenACC and OpenMP spellings, debug and trace code, device scratch copies for C++ schemes |
| `capgen/generator/cxx_schemes.py` (new) | C++ schemes: interface blocks, C prototype headers |
| `unit-tests/test_device_residency.py` (new) | 72 tests |
| `end-to-end-tests/gpu_residency/` (new) | GPU test, shared build script and GPU run script, broken fixtures |
| `end-to-end-tests/gpu_constituents/` (new) | GPU constituent test |
| `end-to-end-tests/cxx_schemes/` (new) | C++ scheme test (CPU and GPU-ported C++ schemes) |

## 8. Reproducing

```bash
python -m pytest unit-tests
module swap intel nvhpc                     # login shell on Derecho
cd end-to-end-tests/gpu_residency           # see README.md there
for m in off gpu gpu-debug gpu-trace unified omp omp-debug omp-trace omp-unified; do
  ./build.sh $m $B/$m
  TEST_DIR=../gpu_constituents ./build.sh $m $C/$m
done
qsub -A <account> -v BUILD_ROOT=$B,BUILDS="gpu omp unified" run_on_gpu.pbs
qsub -A <account> -v BUILD_ROOT=$C,BUILDS="gpu omp unified",EXE=test_gpu_constituents.x,OUT=gpu_constituents_out.txt run_on_gpu.pbs
```

## 9. Decisions for the maintainers (FU-036)

1. Attribute names and values: `memory_space = host|device` and
   `host_access = read_write|read|none`.  `memory_space` matches
   xdsl-ccpp's spelling deliberately.
2. Whether the two sync entry points belong in the host-facing API.
3. Where a host should declare `host_access` for constituents (FU-045).
4. The C++ scheme contract: dense column-major arrays via device scratch
   copies (simple, costs copies) or base address plus strides (no copies,
   more work for scheme authors) (FU-049).
5. Whether to merge as an opt-in feature now, with the limits above as
   errors.

## CAM-SIMA aux_sima check (2026-10-05, GPU feature off)

Two scratch sandboxes of `climbfuji/CAM-SIMA` branch `feature/capgen-ng`
(f9ae41c) on Derecho with Intel.  This branch carries the capgen v0 → v1
compatibility layer (`cime_config/capgen_compat/`).  Upstream CAM-SIMA
`development` still calls the v0 `capgen()` API.  *Stock* uses the capgen v1
the branch pins (`climbfuji/ccpp-framework` f37cbde).  *PoC* replaces
`ccpp_framework/` with this tree.  Both sandboxes needed the same three local
fixes to the branch, none of them in capgen:

1. `capgen_compat/_runner.py`: drop repeated host metadata paths.  The
   registry lists `radiative_aerosol_definitions.meta` more than once, and
   capgen v1 rejects the duplicate standard names.
2. Rename `cam_constituents_array` / `cam_model_const_properties` to the
   `ccpp_`-prefixed names in `cam_comp.F90`, `dp_coupling.F90` and
   `phys_comp.F90`.  These are leftovers from the merge of `development`.
3. Delete a stray merge-conflict marker in `src/dynamics/se/dp_coupling.F90`.

| Test | Stock | PoC | Generated caps | History (cprnc) |
|---|---|---|---|---|
| SMS_Ln9.mpasa480_mpasa480.FKESSLER (MPAS) | PASS | PASS | identical | IDENTICAL |
| SMS_Ln9.ne3pg3_ne3pg3_mg37.FKESSLER (SE-CSLAM) | PASS | PASS | identical | IDENTICAL |
| SMS_Ln9.ne3pg3_ne3pg3_mg37.FHS94 | PASS | PASS | identical | IDENTICAL |
| SMS_Ln9.ne3pg3_ne3pg3_mg37.FTJ16 | PASS | PASS | identical | IDENTICAL |
| SMS_Ln9.ne3pg3_ne3pg3_mt233.QPC4 (CAM4) | FAIL setup | FAIL setup | — | — |
| SMS_D_Ln9.ne3pg3_ne3pg3_mt233.F1850_C4 | FAIL setup | FAIL setup | — | — |

The CAM4 setups fail in both sandboxes with the same capgen v1 error.
`column_visible_optical_depth_per_bulk_aerosol` is a suite-owned array
allocated at init.  Its dimension is set by a scheme, so capgen v1 asks for
the producing scheme to make it `allocatable, intent(out)`.  This is a
CAM-SIMA/atmospheric_physics metadata change and does not involve the GPU
work.

## CAM-SIMA Kessler on the GPU (2026-10-05)

Real model, real suite.  Case `FKESSLER`, `ne3pg3_ne3pg3_mg37` (SE-CSLAM),
NVHPC 25.9, 4 MPI ranks sharing one A100, 9 steps.  The CPU reference is
the same source and configuration without OpenACC.  Both use `-Mnofma` on
the host; the GPU build uses `-gpu=nofma`.

**What changed in the CAM-SIMA sandbox** (all local, in scratch):

* `kessler.meta`: `memory_space = device` on the `kessler_run` table.  No
  other scheme changed, so the rest of the suite runs on the CPU.
* `kessler.F90`: one `!$acc parallel loop gang vector default(present)`
  over columns, with the per-column work arrays private.  For the device,
  the module constants are copied into locals.  The in-loop error `return`
  became a counted reduction, reported after the loop.  `precl` is zeroed
  inside the loop.
* `capgen_compat/_runner.py`: reads capgen's device flags from a
  `ccpp_device_opts` file found from the working directory or above the cap
  output directory.  It must be found from both, because buildnml runs from
  different directories at setup and at build time.
* The case's `Depends.nvhpc`: Derecho compiles only listed files with
  `-acc`.  Without listing `kessler.o` and the CCPP caps, the generated
  directives are silently compiled away.
* `src/dynamics/se/dp_mapping.F90`: NVHPC 25.9 rejects the renamed `pi` in
  a `parameter` expression, so the same value is written as a literal.  This
  is unrelated to the GPU work.
* The batch request was reduced to one node with 4 ranks and 1 GPU.  CIME's
  Derecho GPU template asks for full nodes, which the develop queue rejects.

**What capgen generated.**  In the `physics_before_coupler` group,
`kessler_run` gets 9 inputs copied to the device just before the call.
These are `cpairv`, `rairv`, `phys_state%zm` and six suite interstitials
(`rho`, `exner`, `theta`, `qv_dry`, `qc_dry`, `qr_dry`).  Right after the
call, the four arrays the following CPU schemes read (`theta` and the three
dry mixing ratios) are copied back.  `precl` is copied back only just
before `kessler_diagnostics`, its first host reader.  `relhum` stays on the
device because nothing on the host reads it.  Device copies are created
once in `kessler_init` and deleted in `kessler_final`.

**Results.**

| Check | Result |
|---|---|
| Kernel on the GPU (`NV_ACC_TIME`, rank 0) | `kessler_run` kernel launched 11 times; each generated update ran 11 times; `kessler_init` created the device copies |
| GPU vs CPU history (`cprnc`) | Initial file identical.  After 9 steps T is identical and Q differs at one time (5.9e-16 normalized RMS); U and V differ at all 9 times, at most 5.2e-10 normalized RMS |
| `--device-debug` build | Runs clean: presence and coherence checks at 25 more directive sites, none fired.  History identical to the plain GPU run |

The CPU–GPU differences are expected.  Kessler's `exp` and fractional
powers use the GPU math library rather than the host's, so bit-for-bit
agreement is not possible.  The dycore spreads those last-bit changes into
the winds.

**Pitfall found.**  The first GPU run aborted with
`data in PRESENT clause was not found on device: precl`.  CIME still
recorded the run as a success.  The device copies had never been created,
because the caps had been regenerated during the build without the device
flags.  The presence error pointed straight at it.

## Generated timers: `--timer-style cesm|fms` (2026-10-06)

`--timer-style` makes the group caps time every scheme call, and, with
`--directive`, the generated copies around it.  The timers use the host
model's own timing library:

| Timer | What it covers |
|---|---|
| `<scheme>_<phase>` | the call (compute).  For a call with a device-side argument, a `!$acc wait` / `!$omp taskwait` precedes the stop, so asynchronous kernels are counted |
| `<scheme>_<phase>_h2d` | copies to the device placed before this call |
| `<scheme>_<phase>_d2h` | copies to the host placed before or after this call |
| `<group>_<phase>_loop_h2d`, `_loop_d2h` | copies before a subcycle loop |
| `<group>_<phase>_exit_h2d`, `_exit_d2h` | copies restoring the handoff state at group exit |

The planner copies data back to the host lazily, just before its first host
reader.  A `_d2h` timer is therefore often named after the CPU scheme that
needs the data, not after the GPU scheme that produced it.  Unit and kind
transforms and `--device-debug` checks are outside the timers.  Without
`--directive` only the compute timers exist, so a CPU build and a GPU build
report the same compute timer names.

Two styles, because the hosts differ.  UFS's FV3 driver (`atmos_model.F90`)
uses FMS clocks, not GPTL:

| | `cesm` | `fms` |
|---|---|---|
| Library | GPTL via CIME `perf_mod` | FMS `mpp_mod` clocks |
| Calls | `t_startf(name)` / `t_stopf(name)` | `mpp_clock_id(name, grain=CLOCK_ROUTINE)` once per clock, then `mpp_clock_begin(id)` / `mpp_clock_end(id)` |
| Name limit | 63 characters | 32 characters |
| Threads | thread-safe | not thread-safe: only OpenMP thread 0 times (UFS runs physics blocks in threads) |
| Visibility | the timers are nested below the host's physics timer, so CESM's `TIMER_LEVEL` (maximum timer nesting depth) must be large enough; `TIMER_DETAIL` does not apply because the caps do not call `t_adj_detailf` | FMS's `clock_grain` must be `'ROUTINE'` or finer |

Over-long names are shortened deterministically: the `_h2d`/`_d2h` suffix is
kept and a 4-digit hash keeps names distinct.  The generated module holds a
name table and two helpers, `ccpp_timer_start(i)` and `ccpp_timer_stop(i)`.
For FMS, the helpers also hold the lazily registered clock ids.

Tests:

* 7 unit tests (`TestTimers`).
* Output with timers off is byte-identical across all 12 end-to-end test
  directories.
* The `gpu_residency` GPU test built with `TIMERS=cesm` and with
  `TIMERS=fms` links stand-in `perf_mod` / `mpp_mod` modules
  (`end-to-end-tests/gpu_residency/timer_stubs/`).  These abort on any
  improperly nested start/stop and print a report at the end.  Both builds
  match the CPU reference bit for bit on an A100 and report the same timer
  set.

### CAM-SIMA Kessler at ne30 with timers (2026-10-06)

`FKESSLER`, `ne30pg3_ne30pg3_mg17` (48,600 columns, 30 levels), NVHPC 25.9,
9 steps, `--timer-style cesm`, `TIMER_LEVEL=12`.  Only `kessler_run` is on
the GPU.  Times are per physics call (each 10-call total divided by 10),
averaged over ranks:

| | CPU, 4 ranks | CPU, 128 ranks (full node) | GPU, 4 ranks, 4 A100s |
|---|---|---|---|
| Columns per rank | 12,150 | ~380 | 12,150 |
| `kessler_run` compute | 52.7 ms | 1.74 ms | **0.53 ms** |
| Copies to device (`kessler_run_h2d`, 9 arrays, 26 MB) | — | — | 4.82 ms |
| Copies to host (5 `_d2h` timers, 12 MB) | — | — | 2.33 ms |
| Kessler total | 52.7 ms | 1.74 ms | 7.68 ms |
| `phys_run1` (whole group, 19 schemes) | 98.4 ms | 3.37 ms | 57.9 ms |

* **Compute.**  The GPU runs Kessler about 100× faster than one CPU core
  doing the same columns, and about 3.3× faster than a full 128-core node
  doing all of them.
* **Data movement** is 94% of Kessler's GPU time.  It runs at about
  5 GB/s, the rate for pageable host memory.  Pinned host memory, or
  porting the neighbouring conversion schemes so data stays on the device,
  are the obvious next steps.  Asynchronous transfers are FU-041.  With
  only Kessler ported, the GPU run is about 6.9× faster than 4 CPU ranks but
  4.4× slower than a full CPU node.
* **Correctness.**  GPU vs CPU (4 ranks): differences at roundoff, at most
  2e-9 normalized RMS (V), as at ne3.  The 4- and 128-rank CPU runs are
  identical.
* **Total model time** is dominated by the SE dycore on the CPU (224 s for
  10 steps with 4 ranks), so this layout is only for timing physics.

**One A100 vs one CPU node.**  Same case with 1 rank on 1 A100 (all 48,600
columns in one kernel per call), compared with 128 CPU ranks on one
Derecho CPU node.  Per physics call:

| | CPU node, 128 ranks | 1 A100, 1 rank |
|---|---|---|
| `kessler_run` compute | 1.74 ms | **1.40 ms** |
| Copies to device (9 arrays, 105 MB) | — | 7.95 ms (13 GB/s) |
| Copies to host (5 arrays, 47 MB) | — | 3.89 ms (12 GB/s) |
| Kessler total | 1.74 ms | 13.2 ms |
| `phys_run1` (rest of the group on one CPU core) | 3.37 ms | 156.5 ms |

One A100 computes Kessler about 1.2× faster than a full 128-core node.
With the copies it is about 7.6× slower, so with only Kessler ported,
offloading does not pay.  It would pay once the surrounding schemes run on
the device and data stays there.  Larger copies reach 12–13 GB/s, against
about 5 GB/s with 12,150 columns per rank.  Kessler's compute time grew
only 2.6× for 4× the columns (0.53 ms at 12,150 columns on one GPU), so one
rank's share at 4 GPUs did not saturate a GPU.  Output is identical to the
4-GPU run, and differs from the CPU runs only at roundoff.

## CCPP single-column model (SCM) regression tests (2026-10-07)

DTC `ccpp-scm` `main` (a52680b0), which already uses capgen v1 (submodule
`feature/capgen-v1`, pinned at `c4b7d9a`).  Four scratch copies share the
case data:

* **stock**: the pinned capgen v1;
* **base**: capgen v1 `e21f3db`, the commit capgen v1.1 starts from, 14
  upstream commits newer;
* **v1.1**: this framework;
* **xdsl**: xdsl-ccpp.

The procedure follows the SCM's CI (`run_scm_rts.yml`): the 10 suites of
the `supported` Intel list, Release build with Intel oneAPI 2025.3.1
(`derecho_intel/2.1.1` modules), and the `supported_ifx` run list (31
cases) at 10% runtime.

| | stock | base | v1.1 | xdsl-ccpp |
|---|---|---|---|---|
| Caps generated | yes | yes | yes, byte-identical to base (all 73 files) | **no** (0 of 10 suites) |
| Build | yes | yes | yes | — |
| Cases run | 31/31 pass | 31/31 pass | 31/31 pass | — |
| Output vs stock (`cmp` of `output.nc`) | — | 31/31 identical | 31/31 identical | — |

xdsl-ccpp cannot generate caps for the SCM.  Three failures were worked
around, in the scratch copy only, to see how far it gets:

1. It rejects capgen v1's `module_name` table property (3 metadata files).
2. Parser bug: any line containing `[` and `]` is taken as a section
   header, so a value such as `long_name = sea-ice concentration [0,1]`
   ends the variable early and loses its type (5 files).
3. It does not support capgen v1 `type = control` tables, which hold the
   host control variables.

It then fails for every suite:

* **7 suites:** it cannot provide the subcycle loop variables
  `ccpp_loop_counter` / `ccpp_loop_extent`, which Thompson microphysics,
  RUC land surface, the MYNN surface layer and the GFS surface loop
  control read.
* **3 prescribed-surface suites** (`SCM_GFS_v16_ps`,
  `SCM_GFS_v16_RRTMGP_ps`, `SCM_WoFS_v0_ps`): it crashes converting a
  character length (`ValueError: invalid literal for int()`).  The only
  non-numeric length in those suites is the deferred-length `len=:` of
  `filename_of_internal_namelist`, the likely cause (not confirmed).

Beyond generation, the SCM driver calls capgen v1's host interface
(`ccpp_physics_run(suite_name=..., group_name=..., <static args>)`).
xdsl-ccpp generates a different interface
(`ccpp_physics_run(suite_name, suite_part, ...)`), so the SCM host code
would also need changes before it could compile.

## Upstream CAM-SIMA (2026-10-07)

Current `ESCOMP/CAM-SIMA` `development` (`47e1e63`, `sima0_22_000`) still
calls capgen v0's interface, so capgen v1 cannot be dropped in.  The same
six Intel `aux_sima` tests (SE and MPAS Kessler, Held-Suarez, TJ2016, CAM4
`QPC4`, CAM4 `F1850_C4` debug) were run three ways and compared with
`cprnc`:

| | Code generated | Pass | Output vs upstream capgen v0 |
|---|---|---|---|
| Upstream as-is, capgen v0 | 6/6 | 6/6 | reference |
| capgen v1.1 on upstream + compatibility layer | 6/6 | 6/6 | 4/6 identical; CAM4 ×2 differ at roundoff |
| xdsl-ccpp (`johnmauff/CAM-SIMA` `xdsl-ccpp-adapter`, 1 commit behind upstream) | 6/6 | 6/6 | 4/6 identical; CAM4 ×2 differ at roundoff |

**capgen v1.1 and xdsl-ccpp agree bit for bit on the CAM4 tests**: both
differ from capgen v0 in the same 39 fields by the same amounts (up to about
1e-10 normalized for state fields).  capgen v0 is the odd one out.  With
capgen v0 on both sources, the xdsl-ccpp branch is identical to upstream, so
its one missing upstream commit and older physics do not matter here.

**Cause of the CAM4 roundoff differences: constituent ordering.**  Found by
checksumming the physics state, tendencies and constituents after every
scheme call: the whole first physics step is bit-identical, and the runs
diverge only after the first dynamics step.  Both generators fill CAM-SIMA's
constituent object, a hash table that hands out constituent indices in its
iteration order, and that order depends on the table size the cap passes to
`initialize_table` (×10, rounded up to a power of two).

* capgen v0 sizes it for the host plus the suite's *static* constituents
  only (3 here), leaving out the 9 that schemes register at run time but
  still inserting them: 32 slots for 12 entries.
* capgen v1.1 sizes it for everything it inserts (12): 128 slots.

Both orders are valid; they differ, for example cloud liquid and cloud ice
swap indices 1 and 2.  `doc/cam4_fwaut_constituent_order.md` (D.
Heinzeller, June 2026) had already traced the CAM4 difference to this swap
and gives the mechanism: `air_composition` lists the water species in
constituent-index order, `cam_thermo` sums them in that order for the
total energy, and the global energy fixer spreads the machine-epsilon
difference over every column.  What is new here is why the order differs
between generators: the table size.  Giving capgen v1.1's generated code
capgen v0's table size makes CAM4 bit-for-bit identical to capgen v0.  We
keep capgen v1.1's sizing and accept the roundoff differences:

* it sizes the table for what it holds, so there are fewer collisions;
* by reading the code, capgen v0's count can be zero (a suite whose
  constituents all come from register phases, and no host constituents);
  `initialize_table` then takes `log(0)`.  Not tested.

Results that depend on a hash table's size are fragile with any generator.
A deterministic order (for example, sorting by standard name, or keeping
insertion order) would be worth raising with the CCPP and CAM-SIMA
maintainers.

**Full `aux_sima` list (46 Derecho tests: 15 Intel, 30 GNU, 1 NVHPC).**

| | Pass | Failures | Output vs capgen v0 |
|---|---|---|---|
| capgen v0 | 44 | CAM7 ×2 (run) | reference |
| capgen v1.1 (first run) | 43 | CAM7 ×2 (run); NVHPC RRTMGP GPU snapshot test (two causes, both since fixed, below) | 31 identical; the 12 CAM4 tests differ at roundoff (constituent order) |
| capgen v1.1 (after both fixes) | 44 | CAM7 ×2 (run) | 32 identical (now including the NVHPC test); the same 12 CAM4 tests differ. Identical to the first v1.1 run on every test that ran in both |
| xdsl-ccpp | 42 of 45 | CAM7 ×2 (run); PUMAS/BAM snapshot test (code generation: local-name collision on `constituents`) | 30 identical; the same 12 CAM4 tests differ |

**The NVHPC RRTMGP GPU snapshot test** (`FPHYStest ... outfrq_rrtmgp_derecho_gpu`)
failed for two independent reasons:

1. **NVHPC bug, fixed in the generator.**  The constituent table failed to
   lock with garbage names: NVHPC corrupts strings in `items = [items, new]`
   (`doc/compiler_issues.md` issue 5).  The cap now appends with
   `move_alloc` and section copies.
2. **Object lifetime, fixed in the generator.**  The model then
   aborted in RRTMGP with OpenACC "variable in data clause is partially
   present on the device" errors (CIME still reported the run as passing).
   capgen v0 makes RRTMGP's objects (`gas_concs_lw/sw`, `atm_optics_*`,
   `aer_*`, `sources_lw`, `cloud_*`, the flux objects) locals of the
   run-group subroutine, fresh on every call.  capgen v1.1 keeps them in
   persistent suite data.  RRTMGP's `rrtmgp_inputs` run phase takes them
   `intent(out)`, so Fortran frees their allocatable components on entry
   without RRTMGP's `!$acc exit data`, leaving stale device mappings that
   new allocations then overlap.  capgen v1.1 now gives such objects a
   per-call lifetime (`assign_run_local_ddts`; rule in `doc/migration.md`
   §1.3.3).  It selects exactly capgen v0's 13 objects, and the run
   completes, **bit-for-bit identical to capgen v0**.  The change leaves
   all 73 CCPP SCM caps byte-identical.

CAM7 (`F2000_C7`) fails at run time with every generator ("Longwave down
sent from the atmosphere model..." in the coupler): an upstream problem.
xdsl-ccpp skipped one GNU test whose test mods are missing on its branch.

**What running capgen v1(.1) on current upstream took** (scratch sandbox,
local commits only):

1. Merge `climbfuji/CAM-SIMA` `feature/capgen-ng` (the capgen v1
   compatibility layer) into current upstream `development`.
2. Physics: the fork's `atmospheric_physics` (`f76d5d1`, 28 commits of
   metadata fixes capgen v1 needs) merged with upstream's
   `atmos_phys0_29_001`.  Upstream's physics alone is rejected by capgen v1's
   stricter checks (a duplicate `dimensions` attribute, a constituent
   written directly instead of through a tendency, a character-length
   mismatch); stock capgen v1 fails identically.
3. Add `CapDatabase.suite_list()` to the compatibility layer.  Upstream's
   `write_init_files.py` uses it since 10 September (skip IC reads for
   variables a suite writes before use).
4. The three fixes needed earlier on the fork branch: duplicate
   metadata paths, old constituent function names, a stray merge marker.
5. **New in capgen v1.1:** CCPP allocates a suite variable sized by an
   init-phase scheme right after that scheme, instead of rejecting it.
   CAM4's `odv_col_aod` is sized by `aerosol_optics_init`.
   `--strict-allocatable` restores the previous rule; the default is open
   for discussion (FU-050).
