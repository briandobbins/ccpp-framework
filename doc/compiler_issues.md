# Compiler issues found during the GPU / C++ proof of concept

*2026-10-06.  Found while building capgen v1's GPU data-movement and
Fortran-host → C++-scheme support (`doc/gpu_data_movement_report.md`).
All were reproduced with small standalone programs on Derecho (AMD EPYC
7763 hosts, NVIDIA A100 GPUs) and are worked around in the generator; this
note records what fails, on which compilers, and what the generator does
instead.*

Compilers tested:

| Compiler | Version |
|---|---|
| NVHPC `nvfortran` / `nvc++` | 25.9 (main), 26.1 (rechecks) |
| Intel `ifx` / `icpx` | 2025.2.1 |
| GNU `gfortran` / `g++` | 14.3.0 (comparison only) |

## Summary

| # | Issue | NVHPC 25.9 | NVHPC 26.1 | Intel 2025.2.1 | gfortran 14.3 | Generator workaround |
|---|---|---|---|---|---|---|
| 1 | `logical(c_bool)` `.true.` is not a valid C `bool` | **fails** (0xFF) | not tested | **fails** (0xFF) | ok (0x01) | logicals cross as a 0/1 byte |
| 2 | C descriptors corrupted when the Fortran caller is compiled for GPU offload | **fails** | **fails** | ok without offload; with offload not tested | not tested with offload | no C descriptors at all |
| 3 | Invalid descriptor for an `allocatable, intent(out)` `BIND(C)` dummy | **fails** | **fails** | ok | ok | allocation handle (callback) |
| 4 | Keyword argument rejected for an allocatable `BIND(C)` dummy | **fails** | **fails** | ok | not tested | as 3 |
| 5 | `a = [a, b]` corrupts strings in the old elements of `a` (constituent properties) | **fails** | **fails** | ok | not tested | `move_alloc` + two section copies |

Issue 5 was found later (2026-10-07) by CAM-SIMA's NVHPC `aux_sima` test, and is
not specific to GPU builds.  Issues 2–4 all concern F2018 "C descriptors" (`ISO_Fortran_binding.h`,
`CFI_cdesc_t`), the standard way to pass assumed-shape, allocatable or
pointer arrays between Fortran and C.  NVHPC's support is unusable for
GPU-offload builds (issue 2), so the generator never relies on it.

---

## 1. `logical(c_bool)` stores `.true.` as 0xFF

**Expected.** F2018 makes `logical(c_bool)` interoperable with C `_Bool` /
C++ `bool`, whose only valid values are 0 and 1.

**Observed.** NVHPC 25.9 and Intel ifx 2025.2.1 store `.true.` as all bits
set (0xFF) in `logical(c_bool)`, both by value and in arrays.  C++ code
reading it has undefined behaviour.  In practice `b == true` is false, and
`b ? 1.0 : 0.0` evaluated to **255.0** in an Intel build.  gfortran stores
0x01.

**Reproducer.**

```cpp
// b1.cpp
#include <stdio.h>
#include <string.h>
extern "C" void showb(bool b, const bool* a) {
  unsigned char raw, rawa; memcpy(&raw, &b, 1); memcpy(&rawa, a, 1);
  printf("by value raw=0x%02x, array raw=0x%02x, (b == true)=%d\n",
         raw, rawa, (int)(b == true));
}
```

```fortran
! b1.F90
program tb
  use iso_c_binding
  implicit none
  interface
    subroutine showb(b, a) bind(C, name='showb')
      import :: c_bool
      logical(c_bool), value, intent(in) :: b
      logical(c_bool), intent(in) :: a(*)
    end subroutine
  end interface
  logical(c_bool) :: a(1)
  a(1) = logical(.true., c_bool)
  call showb(logical(.true., c_bool), a)
end program
```

| Build | Output |
|---|---|
| `nvc++ -c b1.cpp; nvfortran b1.F90 b1.o -c++libs` | `raw=0xff ... (b == true)=255` |
| `icpx -c b1.cpp; ifx b1.F90 b1.o -lstdc++` | `raw=0xff ... (b == true)=255` |
| `g++ -c b1.cpp; gfortran b1.F90 b1.o -lstdc++` | `raw=0x01 ... (b == true)=1` |
| NVHPC with `-Munixlogical` | `raw=0x01` |
| Intel with `-fpscomp logicals` | `raw=0x01` |

**Workaround in the generator.** The compiler flags fix it, but they change
the representation of every `logical` in the model, which a code generator
cannot require.  Instead the cap converts each logical argument to an
`integer(c_signed_char)` temporary holding exactly 0 or 1
(`merge(1_c_signed_char, 0_c_signed_char, x)`), and back with `t /= 0`.  The
C++ prototype still says `bool`.  A 1-byte integer and `bool` have the same
size and calling convention on the platforms tested.
(`capgen/generator/cxx_schemes.py`, `ArgSpec` / `CallWrap`.)

---

## 2. NVHPC corrupts C descriptors in files compiled for GPU offload

**Expected.** A `BIND(C)` interface with an assumed-shape (`x(:)`) or
allocatable dummy passes a `CFI_cdesc_t*` describing the actual argument.

**Observed.** With `-acc=gpu` or `-mp=gpu` on the **Fortran** side, NVHPC
25.9 and 26.1 pass a descriptor with garbage fields: type, rank and extents
are zero or random.  `CFI_allocate` then fails (return code 16) or the
program segfaults.  This happens for plain assumed-shape arrays too, not
only allocatables.  The same code works without offload flags and with
`-acc=host`.  The C++ side's flags do not matter.

**Reproducer.**

```cpp
// c.cpp
#include <ISO_Fortran_binding.h>
#include <stdio.h>
extern "C" void shape_of(CFI_cdesc_t *y) {
  printf("assumed-shape: type=%d rank=%d extent=%ld\n", (int)y->type,
         (int)y->rank, (long)(y->rank > 0 ? y->dim[0].extent : -1));
}
extern "C" void alloc_of(CFI_cdesc_t *x) {
  printf("allocatable:   type=%d rank=%d attr=%d\n", (int)x->type,
         (int)x->rank, (int)x->attribute);
}
```

```fortran
! n.F90
program t2
  use iso_c_binding
  implicit none
  interface
    subroutine shape_of(y) bind(C, name='shape_of')
      import :: c_double
      real(c_double), intent(in) :: y(:)
    end subroutine
    subroutine alloc_of(x) bind(C, name='alloc_of')
      import :: c_double
      real(c_double), allocatable, intent(inout) :: x(:)
    end subroutine
  end interface
  real(c_double) :: y(5) = 1
  real(c_double), allocatable :: x(:)
  allocate(x(3))
  call shape_of(y)
  call alloc_of(x)
end program
```

| Fortran flags (NVHPC 25.9) | Output |
|---|---|
| none | `type=28 rank=1 extent=5` / `type=28 rank=1 attr=2` (correct) |
| `-acc=host` | correct |
| `-acc=gpu -gpu=cc80` | `type=0 rank=0 extent=-1` / `type=0 rank=0 attr=0` |
| `-mp=gpu -gpu=cc80` | descriptor corrupted (`CFI_allocate` returns 16) |
| NVHPC 26.1, `-acc=gpu` | corrupted |

**Impact.** Any C/C++ interoperability built on C descriptors breaks as
soon as the Fortran side is built for an NVIDIA GPU.  CCPP caps are compiled
with `-acc` in GPU builds (CAM-SIMA's Derecho build rules add it to the
listed files).

**Workaround in the generator.** C++ scheme interfaces never use C
descriptors:

* arrays are assumed-size (`x(*)`), which pass a plain address;
* derived types pass a generated `bind(C)` struct of member addresses,
  shapes and lower bounds;
* allocatable arguments pass an *allocation handle*: a `bind(C)` struct with
  the current address, shape and bounds, plus a function pointer
  (`c_funloc`) to a generated Fortran function.  C++ calls
  `x->allocate(x->handle, lower, upper)`; the Fortran function does an
  ordinary `allocate` on a box the cap moved the array into (`move_alloc`)
  and returns `c_loc` of the new data.  Verified with NVHPC (with and
  without `-acc=gpu`) and Intel (with and without OpenMP offload).

---

## 3. NVHPC: invalid descriptor for `allocatable, intent(out)` `BIND(C)` dummies

**Observed.** Even without offload flags, NVHPC 25.9 and 26.1 pass an
invalid descriptor (type 127, random rank and element length) for an
`allocatable, intent(out)` dummy.  The NVHPC runtime then aborts with
`cfi_to_nvf_kind: unexpected CFI_type_t 127` (25.9) or `... 0` (26.1).
The same interface with `intent(inout)` works.  Intel and gfortran handle
`intent(out)` correctly, deallocating the actual argument on entry as the
standard requires.

**Reproducer.** As in issue 2, with
`real(c_double), allocatable, intent(out) :: x(:)` and an unallocated
actual argument.

**Workaround.** Superseded by the allocation handle (issue 2).  The cap
implements `intent(out)` itself by deallocating before the call.

---

## 4. NVHPC: keyword arguments rejected for allocatable `BIND(C)` dummies

**Observed.** NVHPC 25.9 and 26.1 reject a correct call that names an
allocatable dummy of a `BIND(C)` interface by keyword:

```
NVFORTRAN-S-0079-Keyword form of argument illegal in this context for x
```

The same call by position compiles.  CCPP caps call schemes by keyword.

**Workaround.** Superseded by the allocation handle (issue 2): no interface
dummy is allocatable any more, so keyword calls are kept.

---

## 5. NVHPC: self-referencing reallocating assignment corrupts strings

**Observed.** Appending to an allocatable array of CAM-SIMA's
`ccpp_constituent_properties_t` (which extends `ccpp_hashable_char_t` and has
deferred-length `character(len=:), allocatable` components) with

```fortran
items = [items, new]
```

leaves many of the **old** elements' strings with the right length but garbage
contents.  NVHPC 25.9 and 26.1, at `-O0` and at CAM's flags; Intel ifx is
correct.  It did not reproduce with a plain derived type, only with the real
constituent type, and it depended on the sizes: 41 items then 7 corrupted 32
of 48 names, while 3 then 28 was clean.

**Impact.** capgen v1.1 used this statement to collect the constituents each
scheme's register phase returns.  In CAM-SIMA's
`FPHYStest ... outfrq_rrtmgp_derecho_gpu` test (snapshot initializer: 41
constituents, then RRTMGP: 7) the constituent table then failed to lock:
`ccp_model_const_table_lock ERROR: Bad vertical dimension, '<garbage>imension'`.

**Reproducer.** `ccpp_kinds.F90`, `ccpp_hashable.F90`, `ccpp_hash_table.F90`
and `ccpp_constituent_prop_mod.F90` from CAM-SIMA's build, plus:

```fortran
program t
  use ccpp_constituent_prop_mod, only: ccpp_constituent_properties_t
  implicit none
  type(ccpp_constituent_properties_t), allocatable :: items(:), new(:)
  call make(items, 41)          ! instantiate with names 'cnst_name_<7*i>'
  call make(new, 7)
  items = [items, new]          ! NVHPC: 32 of 48 names now garbage
end program
```

(`make` allocates the array and calls `%instantiate` on each element with a
distinct `std_name` and `vertical_dim = 'vertical_layer_dimension'` or
`'vertical_interface_dimension'`.)

**Workaround in the generator.** The cap appends without the self-reference:

```fortran
block
  type(ccpp_constituent_properties_t), allocatable :: prev(:)
  call move_alloc(items, prev)
  allocate(items(size(prev) + size(new)))
  items(1:size(prev)) = prev
  items(size(prev)+1:) = new
end block
```

Verified correct with NVHPC 25.9 on the reproducer.
(`capgen/generator/suite_cap.py`, the register-phase constituent buffer.)

---

## Related pitfalls (not compiler bugs)

| Pitfall | Effect | Handling |
|---|---|---|
| NVHPC ignores `!$acc` without `-acc` | xdsl-ccpp's Kessler "GPU" build used `-mp=gpu` with OpenACC directives and ran entirely on the CPU, with no error | build with the flag that matches the directives |
| Derecho's `Depends.nvhpc` gives `-acc` only to a fixed list of CAM files | the generated data-movement directives in the caps were silently compiled away; the first CAM-SIMA GPU run then stopped with a presence error | the case adds the CCPP caps and Kessler to the list (`make_case.sh`) |
| `c_loc` of a variable without `TARGET` | ifx rejects it, correctly per the standard; nvfortran accepts it | generated helpers take an assumed-type, assumed-rank `TARGET` dummy (`ccpp_cxx_loc`, `ccpp_gpu_is_present`) |
| icpx vs ifx rounding | the same expression can differ in the last bit between C++ and Fortran | tests compare cross-language results with a 1e-14 relative tolerance |
| NVHPC 25.9 rejects a `parameter` expression in CAM-SIMA's `dp_mapping.F90` that uses a renamed `pi` from `shr_const_mod` | SE dycore fails to compile with NVHPC | a literal value in the local sandbox; cause not confirmed (possibly a different `shr_const_mod` shadowing the CESM one on the include path) |

## Suggested reports to NVIDIA

Issues 2–5 are worth reporting with the reproducers above.  Issue 2 matters
most: it blocks standard Fortran/C++ interoperability in exactly the
GPU-offload builds where C++ physics is most interesting.  Issue 1 is
long-standing documented behaviour of both NVHPC and Intel (their
`-Munixlogical` / `-fpscomp logicals` flags exist for it), but it makes
`logical(c_bool)` unsafe to pass to C++ without those flags, which is worth
a note to both vendors.
