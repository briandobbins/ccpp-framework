TODO UPDATE ME

# CCPP Framework

## This branch: capgen v1.1, a rapid prototype

> **Not production code, and not reviewed.**  `feature/capgen-v1.1` is a
> proof of concept built on capgen v1 (`e21f3db`), shared so others can
> examine and test it.  Do not merge it as-is.

It adds, all off unless asked for:

* **GPU data movement** (`--directive acc|omp`, OpenACC or OpenMP): the
  caps move data between host and device from scheme `memory_space`
  metadata, with a unified-memory mode and runtime checks.
* **A Fortran host calling C++ physics** (`language = c++` in scheme
  metadata), on the CPU and the GPU.
* **Generated timers** (`--timer-style cesm|fms`): compute and data
  movement timed per scheme.

### Changes for existing capgen v1 users

A model that uses capgen v1 can point its build at this branch's `capgen/`
and run its own tests; the command line and Python interface are unchanged.
Generated code is unchanged except for these, which are on by default:

1. **Constituent collection in the register phase** is rewritten:
   `items = [items, scheme_consts]` became `move_alloc` plus section copies,
   because NVHPC 25.9 and 26.1 miscompile the original
   (`doc/compiler_issues.md`, issue 5).  Same behaviour, different text:
   every suite whose schemes register constituents gets a different suite
   cap.
2. **Per-call derived-type objects.**  A derived-type suite object written
   first in one group's run phase, and used in no later phase, is created
   afresh on every call, as original capgen did.  CAM-SIMA's RRTMGP needs
   this on the GPU (`doc/migration.md` §1.3.3).
3. **Arrays sized in `init`.**  A suite array sized by an `init`-phase
   scheme is now allocated by CCPP right after that scheme; capgen v1
   rejected such suites.  `--strict-allocatable` restores the old rule.
4. **Scheme module names.**  When a scheme's metadata gives no
   `module_name` and its Fortran file holds a single module with another
   name, that module is used; capgen v1 generated code that did not
   compile.

### Testing and documentation

* Results: `doc/gpu_data_movement_report.md` (CAM-SIMA `aux_sima` and
  Kessler on the GPU, the CCPP single-column model, end-to-end tests).
* Design: `doc/gpu_data_movement_plan.md`; porting notes: `doc/migration.md`;
  compiler problems found: `doc/compiler_issues.md`; open items:
  `doc/followups.md`.
* GPU and C++ end-to-end tests: `end-to-end-tests/gpu_residency` (build and
  run scripts; "Running on other systems" covers GPUs other than the A100),
  `gpu_constituents`, `cxx_schemes`, `cxx_arg_types`.

## About the CCPP Framework

This repository contains the Common Community Physics Package (CCPP) Framework: The infrastructure that connects CCPP physics schemes with a host model, as well as stand-alone tools for use with CCPP.

## Documentation
The CCPP Framework is designed to be used with the [CCPP Physics repository](https://github.com/NCAR/ccpp-physics) and a host model. Links to further information about this repository and its place in the context of the CCPP as a whole can be found on the [GitHub Wiki](https://github.com/NCAR/ccpp-framework/wiki).

The CCPP Single Column Model (SCM) is designed as a simple host model for testing and development of the CCPP; information about building and using the CCPP in this context can be found in [the CCPP SCM Users Guide](https://dtcenter.org/sites/default/files/paragraph/scm-ccpp-guide-v6.0.0.pdf).

More information about the design and use of CCPP can be found in the [CCPP Technical Documentation](https://ccpp-techdoc.readthedocs.io/en/v6.0.0/).

