# GPU data movement in capgen v1 — proof-of-concept plan

*Written 2026-10-05.  Proof of concept, built locally in
`ccpp-framework_gpu/` (no commits).  To be taken to the CCPP maintainers
afterwards.  Scope: OpenACC data-movement directives generated from
metadata.  Out of scope: C++/multi-language interop.*

## 1. Goal

Let a host run a CCPP suite in which some schemes are GPU-ported and some
are not, with capgen v1 generating every host↔device transfer from
metadata.  Scheme authors write kernels that assume their array arguments
are already on the device (`present` / `default(present)`); the generated
caps guarantee that, and keep the host copy correct wherever CPU code
reads it.

With the feature switched off (no `--directive` flag) the generator's
output must be **byte-identical** to today's.

## 2. Design

### 2.1 Metadata vocabulary

| Attribute | Where | Values | Meaning |
|---|---|---|---|
| `memory_space` | scheme argument | `host` (default), `device` | Where this scheme accesses the array: in GPU kernels (`device`) or CPU code (`host`). Same spelling as xdsl-ccpp. |
| `memory_space` | `[ccpp-arg-table]` header of a scheme phase | `host`, `device` | Default for every argument of that phase, so a fully ported phase is marked once. A per-argument value overrides it. |
| `memory_space` | host variable | `host` (default), `device` | `device` = the host allocates and maintains the device copy itself (e.g. a GPU dycore). CCPP never creates, deletes or moves it; it only relies on it being present. |
| `host_access` | host variable | `read_write` (default), `read`, `none` | What host code outside the suite does with the variable while the host has control (between CCPP calls). Drives the handoff state below. |

`memory_space` describes where code runs, not the hardware.  Whether
memory is unified is a property of the build target and is selected on
the command line (§2.6).

### 2.2 Residency model: allocate once, then only update

* **Allocation.**  Every array that any device-side scheme argument
  touches gets a device copy created once (`!$acc enter data create`) when
  the suite is initialised (`<suite>_init`, reached from `ccpp_init`, which
  takes no column or thread arguments and runs once).  It is deleted in
  `<suite>_final`.  One create/delete pair per array per run: no
  reference-count pairing problems, no threading races.
  Framework-owned interstitials (`ccpp_suite_data(i)%x`) get their device
  copy right after the framework allocates them.
* **Coherence.**  All movement afterwards is `!$acc update device(...)` /
  `!$acc update self(...)` on exactly the slice a call uses (the resolver's
  existing call expression, e.g. `x(col_start:col_end, 1:nlev)`).  Updates
  allocate nothing and are not reference-counted, so OpenMP threads working
  on different column chunks do not interfere.

### 2.3 Handoff states make groups independent

Each tracked array has a fixed *handoff state*: where valid data must be
whenever the host has control.  Every group phase subroutine starts from
that state and restores it before returning.  The analysis inside a group
is exact; call order between groups, skipped groups (e.g. radiation not
called every step) and threading do not matter.

| Variable | Handoff state | Consequence |
|---|---|---|
| host var, `host_access = read_write` (default) | valid on host only | update device before the first GPU read in a group; update self at group exit if a GPU scheme wrote it |
| host var, `host_access = read` | valid on both | update self at exit if the device copy is newer; update device at exit if a CPU scheme wrote it |
| host var, `host_access = none` | valid on device | no boundary traffic; update device at exit if a CPU scheme wrote it last |
| host var, host `memory_space = device` | valid on device | as `none`, but CCPP does not allocate it |
| framework interstitial | valid on device | as `none` |

Only arrays that some device-side scheme argument touches are tracked.
Everything else is untouched by this feature.

### 2.4 The planner

New module `capgen/generator/device_residency.py`, pure Python.  For each
suite, group and phase it walks the resolved calls in order, tracking for
each tracked array the set of locations holding valid data (`{H}`, `{D}`,
`{H,D}`):

* before a call that **reads** an array on side *s* where *s* is not valid:
  emit an update to *s*;
* after a call that **writes** an array on side *s*: only *s* is valid;
* **subcycles**: the loop-entry state is the meet (intersection) of the
  pre-loop state and the loop-body exit state, iterated to a fixed point;
  after the loop the state is that same meet (covers zero iterations);
* **runtime-active (optional) arguments**: updates are wrapped in the
  argument's active condition; a conditional write is followed by a
  guarded update that restores the pre-call state, so the state stays
  path-independent;
* **group exit**: emit the updates needed to restore each touched array's
  handoff state.

The plan is a data structure (per-call pre/post update lists, per-phase
entry/exit lists, the set of arrays to create) attached to the suite
resolution.  Emitters only print it.

### 2.5 Generated code

| Where | What |
|---|---|
| group cap phase subroutines | group-exit updates; per-call pre/post updates next to each scheme call |
| suite cap `<suite>_init` / `<suite>_final` | `enter data create` / `exit data delete` of tracked host arrays, derived-type parents before members |
| suite cap interstitial alloc/dealloc | create/delete of tracked interstitials |
| host cap | two new entry points, `ccpp_physics_device_update_device(suite_name, ...)` and `ccpp_physics_device_update_host(suite_name, ...)`.  The host calls the first after it fills initial data (it copies the `host_access = none` and `read` arrays) and the second before it reads `none` data for output or restarts (it copies only those).  Other arrays are valid on the host whenever the host has control; copying them would destroy newer host values. |
| `datatable.xml` | a residency section listing every tracked array, its handoff state, and where it is touched on the device |

### 2.6 Command line

* `--directive acc|omp` — enable the feature with OpenACC or OpenMP target
  directives (OpenMP added after the phases below).  Absent: `memory_space`
  and `host_access` are parsed and validated but ignored.
* `--memory-model discrete|unified` (default `discrete`).  `unified` (for
  GH200 / MI300A-class machines): no creates, deletes or updates; at every
  point where `discrete` would update the host copy, emit `!$acc wait`
  instead, so asynchronous kernels finish before CPU code reads.
* `--device-debug` — emit runtime checks (§2.7).
* `--device-trace` — print presence, shape, bounds and slice-bound values
  before every GPU call, and announce every update (added after the
  phases below, modelled on xdsl-ccpp's `--gpu-debug-prints`).
* `--device-sync-debug` — copy every GPU result back to the host right
  after the call that produced it (modelled on xdsl-ccpp's
  `--gpu-debug-sync`).

### 2.7 Debug tooling

* **Presence checks**: before each device-side call, verify each tracked
  array is present on the device (`acc_is_present`), failing through
  `errflg`/`errmsg`.
* **Coherence checks**: wherever the plan says host and device copies are
  both valid (at group exit, and at group entry for `host_access = read`),
  copy the device slice back into a scratch array and compare it with the
  host slice; report the array, suite, group and phase on mismatch.  This
  catches wrong intents (an `intent(in)` argument written on the device)
  and broken `host_access` promises.

### 2.8 Proof-of-concept restrictions (hard errors with clear messages)

* A device-side argument that needs a unit, kind or vertical-flip
  transform.
* A device-side argument in the `register` phase or in a suite-level
  `<init>`/`<final>` scheme.
* A device-side argument whose host variable is subscripted by an
  instance or thread index, or a multi-instance constituent object.
  (Single-instance constituent state arrays are supported: they are
  tracked as whole units per column chunk, added after the phases below.)
* A tracked array accessed with different slices inside one phase
  subroutine.

## 3. Phases

Status as of 2026-10-05.  Results: `doc/gpu_data_movement_report.md`.

| # | Phase | Gate | Status |
|---|---|---|---|
| 0 | Baseline: environment, run the existing unit tests, generate caps for every end-to-end test suite and keep them as reference output | Baseline recorded | done |
| 1 | Metadata and CLI plumbing: parse/validate `memory_space` and `host_access`, carry them to the resolver, add the three flags | All existing unit tests pass; flag-off output byte-identical to the baseline | done |
| 2 | Planner (`device_residency.py`) with unit tests: straight-line, mixed host/device, two groups, every handoff state, subcycles, active arguments, the PoC error cases | Planner tests pass | done |
| 3 | Discrete-mode emission: group caps, suite init/final, interstitials, the two sync entry points, datatable section | Generated code compiles with nvfortran `-acc`; flag-off output still byte-identical | done |
| 4 | GPU end-to-end test on Derecho A100 (`end-to-end-tests/gpu_residency`) | GPU run matches a CPU-only build of the same test | done, bit for bit |
| 5 | Debug tooling: presence and coherence checks; deliberately wrong-intent and wrong-`host_access` fixtures | Both fixtures are caught at runtime on the GPU | done |
| 6 | Unified mode: `wait`-only emission; build and run with NVHPC managed memory on the A100 | Unified-mode run matches the CPU build | done (full unified memory unsupported on this platform; FU-042) |
| 7 | Report for maintainers; open items recorded in `doc/followups.md` | Report written | done (FU-036…FU-044) |
| 8 | Optional: generate caps for real suites from CAM-SIMA host metadata | — | not done: CAM-SIMA's registry still emits capgen v0 metadata, and real suites need device-side constituents (FU-037, FU-042) |
