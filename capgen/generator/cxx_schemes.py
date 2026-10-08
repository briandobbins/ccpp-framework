#!/usr/bin/env python3
"""Fortran host calling C++ schemes (``language = c++``).

A scheme whose ``[ccpp-table-properties]`` declares ``language = c++`` is
implemented in C or C++ with C linkage.  The group caps reach it through a
``BIND(C)`` interface block instead of a ``use`` statement; the call site is
unchanged apart from the conversions below (keyword arguments match the
interface's dummy names, which are the metadata local names).  For each such
scheme a C/C++ header with the exact prototypes is written next to the caps,
so the scheme author cannot get the signature wrong.

Argument mapping (metadata order is the C parameter order):

=============================  ============================  ======================
metadata                       Fortran interface dummy       C parameter
=============================  ============================  ======================
integer scalar, in             ``integer(c_int), value``     ``int``
integer scalar, out/inout      ``integer(c_int)``            ``int*``
real scalar, in                ``real(c_X), value``          ``double`` / ``float``
real scalar, out/inout         ``real(c_X)``                 ``double*`` / ``float*``
logical scalar, in             ``integer(c_signed_char),``   ``bool``
                               ``value`` (0 or 1)
logical scalar, out/inout      ``integer(c_signed_char)``    ``bool*``
integer/real/logical array     ``(*)`` assumed size          ``[const] T*``
character scalar               ``character(kind=c_char)``    ``[const] char*``
                               ``, dimension(*)``
derived type (``type = T``)    ``type(ccpp_T_t)``            ``[const] ccpp_T_t*``
``optional = True``            ``optional`` (never           pointer, ``NULL`` when
                               ``value``)                    absent
``allocatable = True``         ``type(ccpp_alloc_T_R_t)``    ``ccpp_alloc_T_R_t*``
                               (allocation handle)
=============================  ============================  ======================

Real kinds map through the generator's kind table (``--kind-type`` and
``kind_spec``): ``REAL64`` -> ``double``, ``REAL32`` -> ``float``.  Arrays are
column-major (the first dimension varies fastest).  A non-contiguous
argument (a column chunk of a 2-D array, say) is passed as a contiguous
temporary by the compiler, so the scheme always sees a dense array of the
declared shape.  Character arguments arrive blank-padded to the host's
declared length, not NUL-terminated.

**logical**: neither Fortran's default ``logical`` nor ``logical(c_bool)``
is a safe C ``bool``: NVHPC 25.9 and Intel ifx store ``.true.`` as all bits
set (0xFF) even in ``logical(c_bool)``, which is not a valid ``bool`` value
(``b ? 1.0 : 0.0`` can give 255).  So the cap converts to and from a 1-byte
integer temporary holding exactly 0 or 1 (``integer(c_signed_char)``), and
the C++ side declares ``bool``; any non-zero value written by C++ reads back
as ``.true.``.

**Derived types**: the argument is not flattened into one C parameter per
member.  The generator writes a C-interoperable descriptor ``ccpp_T_t`` from
the type's own metadata table: for each integer or real member, a pointer to
its data and, for arrays, its shape and Fortran lower bounds
(``<member>_shape``, ``<member>_lbound``).  The cap fills one descriptor per
call and passes a single pointer, so a large type still costs one argument
and the C++ code never depends on how a Fortran compiler lays out the type.
Writes through the member pointers change the host's data.  Members of other
types (logical, character, nested derived types, non-default integer kinds)
are left out and listed in the header.  Members must be contiguous
(allocatable or explicit-shape); a member whose metadata says
``allocatable = True`` is passed as ``NULL`` with zero shape when unallocated.

**Optional arguments**: F2018 ``BIND(C)`` optional dummies.  An absent
argument arrives as a null pointer, so optional scalars are passed by
reference even when ``intent(in)``.

**Allocatable arguments** (arrays the scheme allocates): an allocation
handle ``ccpp_alloc_T_R_t`` (``T`` the C element type, ``R`` the rank) with
the current allocation (``data``, ``shape``, ``lbound``; ``data`` is NULL
when unallocated) and an ``allocate`` function pointer.  The C++ scheme
calls ``x->allocate(x->handle, lower, upper)`` with Fortran bounds; a
generated Fortran function does an ordinary ``allocate`` and returns the
data pointer.  The cap moves the host array into a local box before the
call and back after it, and deallocates it first for ``intent(out)``.
F2018 C descriptors (``CFI_cdesc_t``, ``CFI_allocate``) would be the
standard route, but NVHPC 25.9 and 26.1 pass invalid descriptors for
allocatable ``BIND(C)`` dummies whenever the Fortran file is compiled for
GPU offload (``-acc=gpu``, ``-mp=gpu``), as the caps are in GPU builds, and
for ``allocatable, intent(out)`` dummies even without offload.

Device-side arguments (``memory_space = device``, with ``--directive``):
the pointer refers to host memory that has a device copy, so a kernel finds
the device data with ``present`` (OpenACC) or an implicit/``tofrom`` map
(OpenMP).  Arrays of rank 2 or more, and logical arrays of any rank, are
passed through a dense scratch array the cap creates, fills and empties on
the device (converting logicals to and from 0/1 bytes in device loops), so
column chunks work; other rank-1 slices are contiguous and passed in place.
Optional device-side arguments are passed only when present: the scratch
array is allocated only under the argument's active condition, and an
unallocated (or a disassociated in-place) actual makes the dummy absent, so
C++ sees NULL.  Logical scalars are converted on the host, as for host-side
schemes.

Device-side derived types: the descriptor still holds host addresses, like
every device-side array; each array member it exposes is tracked by the
data-movement planner as an ordinary host array (device copy created at
init, updates planned per the argument's intent), so a kernel finds it with
``present``.  Scalar members are host pointers: read them on the host
before launching kernels.  Members declared allocatable are rejected.  The
planner also sees host-side C++ derived-type arguments member by member,
so a host write through a descriptor invalidates the device copy.

Device-side allocatable arguments: the allocation callback is a device
variant that also creates the device copy of the new array (and deletes the
old one); the array is tracked as a scheme-allocated interstitial, whose
updates and checks run only while it is allocated and whose device copy
``<suite>_final`` deletes.  Every scheme that allocates such an array must
be a device-side C++ scheme.

Not supported in this proof of concept (generation-time errors): character
arrays, non-default integer kinds, arrays of derived types, optional
derived-type or allocatable arguments, allocatable logical or character
arguments, device-side derived types with allocatable members, and C++
schemes in the register phase or as suite-level ``<init>``/``<final>``
schemes.  See doc/followups.md.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from metadata.parse_tools import CCPPError
from generator.device_emit import _fit

_ISO_REAL = {'REAL64': ('c_double', 'double'), 'REAL32': ('c_float', 'float')}

#: Name of the generated module procedure returning ``c_loc`` of any array.
LOC_HELPER = 'ccpp_cxx_loc'


def is_cxx(scheme_store, scheme_name: str) -> bool:
    return scheme_store.language_for(scheme_name) == 'c++'


def _err(scheme: str, phase: str, var, msg: str) -> CCPPError:
    return CCPPError("C++ scheme '{}' phase '{}' argument '{}': {}".format(
        scheme, phase, var.local_name, msg))


def _real_kind(kind: str, kind_types: Dict[str, Tuple[str, str]],
               scheme: str, phase: str, var) -> Tuple[str, str]:
    """(Fortran C kind, C type) for a real of metadata kind *kind*."""
    if not kind:
        return 'c_float', 'float'
    entry = kind_types.get(kind)
    spec = (entry[1] if entry else kind).upper()
    if spec in _ISO_REAL:
        return _ISO_REAL[spec]
    raise _err(scheme, phase, var,
               "real kind '{}' maps to '{}', which has no known C "
               "equivalent; map it to REAL64 or REAL32 with --kind-type"
               .format(kind, spec))


def _is_ddt(typ: str) -> bool:
    return typ.strip().lower() not in ('integer', 'real', 'logical',
                                       'character', 'complex')


def _type_name(ddt: str) -> str:
    """Fortran and C name of the descriptor for derived type *ddt*."""
    return 'ccpp_{}_t'.format(ddt.strip().lower())


########################################################################
# Derived-type descriptors
########################################################################

@dataclass
class _Member:
    name: str
    ctype: str        # C element type
    rank: int
    allocatable: bool


@dataclass
class DdtDescriptor:
    """The C-interoperable descriptor of one derived type."""
    ddt: str
    members: List[_Member]
    omitted: List[Tuple[str, str]]   # (member, reason)

    @property
    def type_name(self) -> str:
        return _type_name(self.ddt)

    def fortran_lines(self, indent: str) -> List[str]:
        i2 = indent + '  '
        out = ['{}! ccpp: descriptor of derived type {} for C++ schemes'
               .format(indent, self.ddt)]
        for name, why in self.omitted:
            out.append('{}! (member {} not accessible from C++: {})'.format(
                indent, name, why))
        out.append('{}type, bind(C) :: {}'.format(indent, self.type_name))
        for m in self.members:
            out.append('{}type(c_ptr) :: {}'.format(i2, m.name))
            if m.rank:
                out.append('{}integer(c_int) :: {}_shape({})'.format(i2, m.name, m.rank))
                out.append('{}integer(c_int) :: {}_lbound({})'.format(i2, m.name, m.rank))
        out.append('{}end type {}'.format(indent, self.type_name))
        return out

    def c_lines(self) -> List[str]:
        guard = '{}_DEFINED'.format(self.type_name.upper())
        out = ['#ifndef {}'.format(guard), '#define {}'.format(guard),
               '/* Descriptor of the Fortran derived type {}: a pointer to'.format(self.ddt),
               ' * each member\'s data (column-major arrays), with the shape',
               ' * and Fortran lower bounds of array members.  Writing',
               ' * through a pointer changes the host\'s data. */']
        for name, why in self.omitted:
            out.append('/* member {} not accessible from C++: {} */'.format(name, why))
        out.append('typedef struct {')
        for m in self.members:
            out.append('    {} *{};'.format(m.ctype, m.name))
            if m.rank:
                out.append('    int {}_shape[{}];'.format(m.name, m.rank))
                out.append('    int {}_lbound[{}];'.format(m.name, m.rank))
        out += ['}} {};'.format(self.type_name), '#endif', '']
        return out


def ddt_descriptor(ddt: str, ddt_index, kind_types,
                   scheme: str, phase: str, var) -> DdtDescriptor:
    table = (ddt_index or {}).get(ddt)
    if table is None:
        raise _err(scheme, phase, var,
                   "derived type '{}' has no metadata table (type = ddt), "
                   "so its members are unknown".format(ddt))
    members, omitted = [], []
    for mv in table.variables():
        typ = mv.type.strip().lower()
        if typ == 'integer' and not mv.kind:
            ctype = 'int'
        elif typ == 'real':
            try:
                ctype = _real_kind(mv.kind, kind_types, scheme, phase, var)[1]
            except CCPPError:
                omitted.append((mv.local_name, "real kind '{}'".format(mv.kind)))
                continue
        else:
            why = ('integer kind {}'.format(mv.kind) if typ == 'integer'
                   else 'nested derived type' if _is_ddt(typ) else typ)
            omitted.append((mv.local_name, why))
            continue
        members.append(_Member(mv.local_name, ctype, len(mv.dimensions),
                               bool(getattr(mv, 'allocatable', False))))
    if not members:
        raise _err(scheme, phase, var,
                   "derived type '{}' has no integer or real members a C++ "
                   "scheme could reach".format(ddt))
    return DdtDescriptor(ddt, members, omitted)


########################################################################
# Per-argument mapping
########################################################################

@dataclass
class ArgSpec:
    """How one argument crosses the Fortran/C++ boundary."""
    name: str
    intent: str
    decl: str               # Fortran interface dummy declaration
    cpar: str               # C parameter
    note: str = ''          # header comment
    kind: str = 'plain'     # 'plain' | 'logical' | 'ddt' | 'alloc'
    rank: int = 0
    optional: bool = False
    ddt: Optional[DdtDescriptor] = None
    ftype: str = ''         # Fortran element type (allocatable arguments)
    ctype: str = ''         # C element type (allocatable arguments)
    device: bool = False    # memory_space = device

    @property
    def alloc_type(self) -> str:
        """Name of the allocation-handle type (allocatable arguments)."""
        return 'ccpp_alloc_{}_{}_t'.format(self.ctype, self.rank)

    @property
    def alloc_box(self) -> str:
        return 'ccpp_cxx_box_{}_{}'.format(self.ctype, self.rank)

    @property
    def alloc_fn(self) -> str:
        return 'ccpp_cxx_alloc_{}_{}'.format(self.ctype, self.rank)

    def alloc_fn_for(self, directive: Optional[str]) -> str:
        """Allocation function: the device variant (also creates and
        deletes the device copy) for a device-side argument when GPU data
        movement is generated, else the host one."""
        if self.device and directive:
            return 'ccpp_cxx_alloc_{}_{}_{}'.format(directive, self.ctype, self.rank)
        return self.alloc_fn

    @property
    def reads(self) -> bool:
        return self.intent in ('in', 'inout')

    @property
    def writes(self) -> bool:
        return self.intent in ('out', 'inout')


def _arg(var, scheme: str, phase: str, kind_types, memory_space: str,
         ddt_index=None) -> ArgSpec:
    """How *var* crosses the boundary (see the module docstring)."""
    typ = var.type.strip().lower()
    intent = var.intent or 'inout'
    rank = len(var.dimensions)
    name = var.local_name
    optional = bool(var.optional)
    alloc = bool(var.allocatable)
    in_only = intent == 'in'
    const = 'const ' if in_only else ''
    opt = ', optional' if optional else ''
    device = memory_space == 'device'

    def device_err(what):
        return _err(scheme, phase, var,
                    '{} arguments must be host-side (memory_space = host) '
                    'in this proof of concept'.format(what))

    if typ == 'character':
        if rank:
            raise _err(scheme, phase, var, 'character arrays are not supported')
        if alloc:
            raise _err(scheme, phase, var,
                       'allocatable character arguments are not supported')
        return ArgSpec(
            name, intent,
            'character(kind=c_char), intent({}){} :: {}(*)'.format(intent, opt, name),
            '{}char* {}'.format(const, name),
            'optional: NULL when absent' if optional else '',
            optional=optional)

    if _is_ddt(typ):
        if rank:
            raise _err(scheme, phase, var,
                       'arrays of derived types are not supported')
        if optional or alloc:
            raise _err(scheme, phase, var,
                       'optional or allocatable derived-type arguments are '
                       'not supported')
        desc = ddt_descriptor(var.type.strip(), ddt_index, kind_types,
                              scheme, phase, var)
        if device:
            alloc_members = [m.name for m in desc.members if m.allocatable]
            if alloc_members:
                raise _err(scheme, phase, var,
                           "device-side derived type: members declared "
                           "allocatable ({}) are not supported (their device "
                           "copies are created once, at init)".format(
                               ', '.join(alloc_members)))
        return ArgSpec(
            name, intent,
            'type({}), intent(in) :: {}'.format(desc.type_name, name),
            '{}{}* {}'.format(const, desc.type_name, name),
            'derived type {}: member pointers into host data{}'.format(
                var.type.strip(),
                ' (array members present on the device; read scalar '
                'members on the host)' if device else ''),
            kind='ddt', ddt=desc, device=device)

    if typ == 'logical':
        if alloc:
            raise _err(scheme, phase, var,
                       'allocatable logical arguments are not supported')
        ftype, ctype = 'integer(c_signed_char)', 'bool'
    elif typ == 'integer':
        if var.kind:
            raise _err(scheme, phase, var,
                       "integer kind '{}' is not supported (default integer "
                       "only, mapped to C int)".format(var.kind))
        ftype, ctype = 'integer(c_int)', 'int'
    elif typ == 'real':
        fkind, ctype = _real_kind(var.kind, kind_types, scheme, phase, var)
        ftype = 'real({})'.format(fkind)
    else:
        raise _err(scheme, phase, var,
                   "type '{}' is not supported".format(var.type))
    kind = 'logical' if typ == 'logical' else 'plain'

    if alloc:
        if optional:
            raise _err(scheme, phase, var,
                       'optional allocatable arguments are not supported')
        if not rank:
            raise _err(scheme, phase, var,
                       'allocatable scalars are not supported')
        spec = ArgSpec(name, intent, '', '', kind='alloc', rank=rank,
                       ftype=ftype, ctype=ctype, device=device)
        spec.decl = 'type({}), intent(in) :: {}'.format(spec.alloc_type, name)
        spec.cpar = '{}* {}'.format(spec.alloc_type, name)
        spec.note = ('allocatable: allocate with {}->allocate({}->handle, '
                     'lower, upper)'.format(name, name)
                     + ('; unallocated on entry' if intent == 'out' else '')
                     + ('; the new array is present on the device'
                        if device else ''))
        return spec
    if rank:
        decl = '{}, intent({}){} :: {}(*)'.format(ftype, intent, opt, name)
        cpar = '{}{}* {}'.format(const, ctype, name)
    elif in_only and not optional:
        decl = '{}, value, intent(in) :: {}'.format(ftype, name)
        cpar = '{} {}'.format(ctype, name)
    else:
        decl = '{}, intent({}){} :: {}'.format(ftype, intent, opt, name)
        cpar = '{}{}* {}'.format(const, ctype, name)
    return ArgSpec(name, intent, decl, cpar,
                   'optional: NULL when absent' if optional else '',
                   kind=kind, rank=rank, optional=optional, device=device)


def _specs(scheme_store, scheme: str, phase: str, kind_types,
           ddt_index=None) -> List[ArgSpec]:
    """Argument specs of one entry point, in metadata order."""
    out = []
    for var in scheme_store.variables_for(scheme, phase) or []:
        space = scheme_store.memory_space_for(scheme, phase, var.local_name)
        out.append(_arg(var, scheme, phase, kind_types, space, ddt_index))
    return out


########################################################################
# Group cap pieces
########################################################################

def _group_specs(phase_calls_by_scheme, scheme_store, kind_types, ddt_index):
    for scheme in sorted(phase_calls_by_scheme):
        for phase in phase_calls_by_scheme[scheme]:
            if phase == 'register':
                raise CCPPError(
                    "C++ scheme '{}': the register phase is not supported "
                    "for C++ schemes".format(scheme))
            yield scheme, phase, _specs(scheme_store, scheme, phase,
                                        kind_types, ddt_index)


def module_use_lines(phase_calls_by_scheme, scheme_store, kind_types,
                     ddt_index=None, indent: str = '  ') -> List[str]:
    """Module-level ``use iso_c_binding`` for the call-site conversions and
    derived-type descriptors (none when the C++ calls need neither)."""
    names = set()
    for _, _, specs in _group_specs(phase_calls_by_scheme, scheme_store,
                                    kind_types, ddt_index):
        for s in specs:
            if s.kind == 'logical':
                names.add('c_signed_char')
            elif s.kind == 'ddt':
                names.update(('c_ptr', 'c_int', 'c_null_ptr'))
            elif s.kind == 'alloc':
                names.update(('c_ptr', 'c_funptr', 'c_int', 'c_null_ptr',
                              'c_loc', 'c_funloc', 'c_double', 'c_float'))
    if not names:
        return []
    return ['{}use, intrinsic :: iso_c_binding, only: {}'.format(
        indent, ', '.join(sorted(names)))]


def interface_lines(phase_calls_by_scheme: Dict[str, List[str]],
                    scheme_store, kind_types, indent: str = '  ',
                    ddt_index=None) -> List[str]:
    """Descriptor types and ``interface`` blocks for every C++ scheme entry
    point a group calls.

    *phase_calls_by_scheme* maps scheme name -> phases called (in order).
    """
    lines: List[str] = []
    descs: Dict[str, DdtDescriptor] = {}
    blocks: List[str] = []
    allocs: Dict[str, ArgSpec] = {}
    for scheme, phase, specs in _group_specs(phase_calls_by_scheme,
                                             scheme_store, kind_types, ddt_index):
        for s in specs:
            if s.ddt is not None:
                descs.setdefault(s.ddt.type_name, s.ddt)
            if s.kind == 'alloc':
                allocs.setdefault(s.alloc_type, s)
        fname = '{}_{}'.format(scheme, phase)
        i2, i3 = indent * 2, indent * 3
        blocks.append('{}interface'.format(indent))
        blocks.append('{}! ccpp: C++ scheme entry point (language = c++)'.format(i2))
        blocks += _subroutine_header(i2, fname, [s.name for s in specs])
        blocks.append('{}use, intrinsic :: iso_c_binding, only: c_int, c_double, '
                      'c_float, c_char, c_signed_char'.format(i3))
        imports = sorted({s.ddt.type_name for s in specs if s.ddt is not None}
                         | {s.alloc_type for s in specs if s.kind == 'alloc'})
        if imports:
            blocks.append('{}import :: {}'.format(i3, ', '.join(imports)))
        blocks.append('{}implicit none'.format(i3))
        blocks += ['{}{}'.format(i3, s.decl) for s in specs]
        blocks.append('{}end subroutine {}'.format(i2, fname))
        blocks.append('{}end interface'.format(indent))
    for name in sorted(descs):
        lines += descs[name].fortran_lines(indent)
    for name in sorted(allocs):
        lines += _alloc_type_lines(allocs[name], indent)
    return lines + blocks


def _alloc_type_lines(s: ArgSpec, indent: str) -> List[str]:
    """Box (holds the array during the call) and handle types for one
    element type and rank."""
    i2 = indent + '  '
    dims = ','.join([':'] * s.rank)
    return [
        '{}! ccpp: allocation handle for C++ schemes ({} array of rank {})'
        .format(indent, s.ctype, s.rank),
        '{}type :: {}'.format(indent, s.alloc_box),
        '{}{}, allocatable :: a({})'.format(i2, s.ftype, dims),
        '{}end type {}'.format(indent, s.alloc_box),
        '{}type, bind(C) :: {}'.format(indent, s.alloc_type),
        '{}type(c_ptr) :: handle'.format(i2),
        '{}type(c_funptr) :: allocate'.format(i2),
        '{}type(c_ptr) :: data'.format(i2),
        '{}integer(c_int) :: shape({})'.format(i2, s.rank),
        '{}integer(c_int) :: lbound({})'.format(i2, s.rank),
        '{}end type {}'.format(indent, s.alloc_type),
    ]


def _alloc_c_lines(s: ArgSpec) -> List[str]:
    guard = '{}_DEFINED'.format(s.alloc_type.upper())
    return [
        '#ifndef {}'.format(guard), '#define {}'.format(guard),
        '/* Allocation handle for a {} array of rank {} that the scheme'.format(s.ctype, s.rank),
        ' * allocates: data/shape/lbound describe the current allocation',
        ' * (data is NULL when unallocated).  allocate(handle, lower, upper)',
        ' * allocates with Fortran bounds (deallocating any current',
        ' * allocation) and returns the new data, column-major. */',
        'typedef struct {',
        '    void *handle;',
        '    {} *(*allocate)(void *handle, const int *lower, const int *upper);'.format(s.ctype),
        '    {} *data;'.format(s.ctype),
        '    int shape[{}];'.format(s.rank),
        '    int lbound[{}];'.format(s.rank),
        '}} {};'.format(s.alloc_type),
        '#endif', '']


def helper_lines(phase_calls_by_scheme, scheme_store, kind_types,
                 ddt_index=None, module_name: str = '',
                 directive: Optional[str] = None) -> List[str]:
    """``contains``-section procedures the C++ calls of a group need: the
    ``c_loc`` helper (derived types) and one allocation function per
    element type and rank (allocatable arguments)."""
    specs = [s for _, _, ss in _group_specs(phase_calls_by_scheme, scheme_store,
                                           kind_types, ddt_index) for s in ss]
    out: List[str] = []
    if any(s.kind == 'ddt' for s in specs):
        out += LOC_HELPER_LINES
    from generator.device_emit import SPELLINGS, _directive
    seen = set()
    for s in specs:
        fn = s.alloc_fn_for(directive) if s.kind == 'alloc' else ''
        if s.kind != 'alloc' or fn in seen:
            continue
        seen.add(fn)
        r = s.rank
        dev = fn != s.alloc_fn
        bounds = ', '.join('lower({0}):upper({0})'.format(d) for d in range(1, r + 1))
        free = ['    if (allocated(box%a)) then']
        if dev:
            free += _directive('      ', SPELLINGS[directive].delete('box%a'), directive)
        free += ['      deallocate(box%a)', '    end if']
        create = (_directive('    ', SPELLINGS[directive].create('box%a'), directive)
                  if dev else [])
        out += [
            '  function {}(handle, lower, upper) &'.format(fn),
            "      bind(C, name='{}_{}') result(p)".format(module_name, fn[len('ccpp_cxx_'):]),
            '    ! ccpp: called from a C++ scheme to allocate an argument{}'.format(
                '; also creates the device copy (and deletes the old one)'
                if dev else ''),
            '    use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_loc, c_f_pointer',
            '    type(c_ptr), value :: handle',
            '    integer(c_int), intent(in) :: lower({}), upper({})'.format(r, r),
            '    type(c_ptr) :: p',
            '    type({}), pointer :: box'.format(s.alloc_box),
            '    call c_f_pointer(handle, box)',
        ] + free + [
            '    allocate(box%a({}))'.format(bounds),
        ] + create + [
            '    p = c_loc(box%a)',
            '  end function {}'.format(fn),
            '',
        ]
    return out


#: ``contains``-section helper: ``c_loc`` of any variable (F2018: the
#: assumed-type, assumed-rank TARGET dummy makes ``c_loc`` legal for host
#: variables that do not have the TARGET attribute themselves).
LOC_HELPER_LINES = [
    '  function {}(a) result(p)'.format(LOC_HELPER),
    '    ! ccpp: address of a derived-type member for a C++ descriptor',
    '    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc',
    '    type(*), dimension(..), target, intent(in) :: a',
    '    type(c_ptr) :: p',
    '    p = c_loc(a)',
    '  end function {}'.format(LOC_HELPER),
    '',
]


@dataclass
class CallWrap:
    """Call-site conversions for one C++ scheme call.

    *directive*: 'acc' / 'omp' when GPU data movement is generated (and the
    memory model is discrete): device-side allocatable arguments then use
    the device allocation function and drop the old device copy first.
    """
    specs: List[ArgSpec] = field(default_factory=list)
    directive: Optional[str] = None

    def lines(self, indent: str, exprs: Dict[str, str], pointers=(),
              packed=()) -> Tuple[List[str], List[str], Dict[str, str]]:
        """(lines before the call, lines after it, argument overrides).
        *exprs* maps each dummy name to its call-site actual expression;
        *pointers* names the optional arguments passed through a pointer
        (absent when it is not associated); *packed* names the device-side
        arguments already passed through a device scratch array, which
        converts logicals itself (device_emit.cxx_pack_lines)."""
        i2 = indent + '  '
        decls: List[str] = []
        pre: List[str] = []
        post: List[str] = []
        unwrapped_pre: List[str] = []
        overrides: Dict[str, str] = {}
        for s in self.specs:
            if s.name in packed:
                continue
            e = exprs[s.name]
            if s.kind == 'logical':
                t = 'ccpp_cxx_l_{}'.format(s.name)
                dims = '({})'.format(','.join([':'] * s.rank)) if s.rank else ''
                decls.append('{}integer(c_signed_char), allocatable :: {}{}'.format(i2, t, dims))
                shape = ', '.join('size({}, {})'.format(e, d)
                                  for d in range(1, s.rank + 1))
                body = ['allocate({}({}))'.format(t, shape) if s.rank
                        else 'allocate({})'.format(t)]
                if s.reads:
                    body.append('{} = merge(1_c_signed_char, 0_c_signed_char, {})'.format(t, e))
                if s.optional and s.name in pointers:
                    # Absent: the temporary stays unallocated, which makes
                    # the optional dummy absent too (F2008).
                    pre.append('{}if (associated({})) then'.format(i2, e))
                    pre += ['{}  {}'.format(i2, b) for b in body]
                    pre.append('{}end if'.format(i2))
                    if s.writes:
                        post.append('{}if (allocated({})) {} = {} /= 0_c_signed_char'
                                    .format(i2, t, e, t))
                else:
                    pre += ['{}{}'.format(i2, b) for b in body]
                    if s.writes:
                        post.append('{}{} = {} /= 0_c_signed_char'.format(i2, e, t))
                overrides[s.name] = t
            elif s.kind == 'ddt':
                d = 'ccpp_cxx_d_{}'.format(s.name)
                decls.append('{}type({}) :: {}'.format(i2, s.ddt.type_name, d))
                for m in s.ddt.members:
                    src = '{}%{}'.format(e, m.name)
                    fill = ['{}%{} = {}({})'.format(d, m.name, LOC_HELPER, src)]
                    if m.rank:
                        fill.append('{}%{}_shape = shape({})'.format(d, m.name, src))
                        fill.append('{}%{}_lbound = lbound({})'.format(d, m.name, src))
                    if m.allocatable:
                        pre.append('{}if (allocated({})) then'.format(i2, src))
                        pre += ['{}  {}'.format(i2, f) for f in fill]
                        pre.append('{}else'.format(i2))
                        pre.append('{}  {}%{} = c_null_ptr'.format(i2, d, m.name))
                        if m.rank:
                            pre.append('{}  {}%{}_shape = 0'.format(i2, d, m.name))
                            pre.append('{}  {}%{}_lbound = 1'.format(i2, d, m.name))
                        pre.append('{}end if'.format(i2))
                    else:
                        pre += ['{}{}'.format(i2, f) for f in fill]
                overrides[s.name] = d
            elif s.kind == 'alloc':
                b = 'ccpp_cxx_b_{}'.format(s.name)
                h = 'ccpp_cxx_a_{}'.format(s.name)
                decls.append('{}type({}), target :: {}'.format(i2, s.alloc_box, b))
                decls.append('{}type({}) :: {}'.format(i2, s.alloc_type, h))
                if s.intent == 'out' and s.device and self.directive:
                    from generator.device_emit import SPELLINGS, _directive
                    pre.append('{}if (allocated({})) then'.format(i2, e))
                    pre += _directive(i2 + '  ', SPELLINGS[self.directive].delete(e),
                                      self.directive)
                    pre.append('{}  deallocate({})'.format(i2, e))
                    pre.append('{}end if'.format(i2))
                elif s.intent == 'out':
                    pre.append('{}if (allocated({})) deallocate({})'.format(i2, e, e))
                else:
                    pre.append('{}if (allocated({})) call move_alloc({}, {}%a)'
                               .format(i2, e, e, b))
                pre += [
                    '{}{}%handle = c_loc({})'.format(i2, h, b),
                    '{}{}%allocate = c_funloc({})'.format(i2, h, s.alloc_fn_for(self.directive)),
                    '{}if (allocated({}%a)) then'.format(i2, b),
                    '{}  {}%data = c_loc({}%a)'.format(i2, h, b),
                    '{}  {}%shape = shape({}%a)'.format(i2, h, b),
                    '{}  {}%lbound = lbound({}%a)'.format(i2, h, b),
                    '{}else'.format(i2),
                    '{}  {}%data = c_null_ptr'.format(i2, h),
                    '{}  {}%shape = 0'.format(i2, h),
                    '{}  {}%lbound = 1'.format(i2, h),
                    '{}end if'.format(i2)]
                post.append('{}if (allocated({}%a)) call move_alloc({}%a, {})'
                            .format(i2, b, b, e))
                overrides[s.name] = h
        if not decls:
            return _fit(unwrapped_pre), [], overrides
        before = unwrapped_pre + [
            '{}block'.format(indent),
            '{}! ccpp: conversions for C++ scheme arguments'.format(i2)] + decls + pre
        after = post + ['{}end block'.format(indent)]
        return _fit(before), _fit(after), overrides


def call_wrap(resolved_call, scheme_store, kind_types,
              ddt_index=None, directive: Optional[str] = None
              ) -> Optional[CallWrap]:
    """Conversions a call to a C++ scheme needs, or None."""
    specs = [s for s in _specs(scheme_store, resolved_call.scheme_name,
                               resolved_call.phase, kind_types, ddt_index)
             if s.kind in ('logical', 'ddt', 'alloc')]
    return CallWrap(specs, directive) if specs else None


def attach_call_wraps(resolved_group, scheme_store, kind_types,
                      ddt_index=None, directive: Optional[str] = None) -> None:
    """Set ``cxx_wrap`` on every C++ scheme call of *resolved_group*."""
    from generator.suite_resolver import iter_phase_calls
    for items in resolved_group.phase_calls.values():
        for call in iter_phase_calls(items):
            if is_cxx(scheme_store, call.scheme_name):
                call.cxx_wrap = call_wrap(call, scheme_store, kind_types,
                                          ddt_index, directive)


@dataclass
class MemberArg:
    """One array member reached through a C++ scheme's derived-type
    descriptor, presented to the GPU data-movement planner like an
    ordinary argument (``device_residency`` reads these attributes)."""
    standard_name: str
    scheme_local_name: str      # the derived-type dummy (its memory_space)
    intent: str
    host_entry: object
    base_expr: str
    scheme_dimensions: List[str]
    is_ddt_member: bool = True
    is_optional: bool = False
    active: str = ''
    active_local: str = ''
    source: str = 'host'
    suite_var: object = None
    subscript: str = ''
    needs_transform: bool = False
    is_constituent_arg: bool = False
    constituent_module_name: str = ''

    @property
    def call_expr(self) -> str:
        return self.base_expr

    @property
    def module_name(self):
        return self.host_entry.module_name

    @property
    def root_symbol(self) -> str:
        return self.base_expr.split('%')[0].split('(')[0]


def attach_member_args(suite_res, scheme_store, kind_types, ddt_index,
                       host_dict) -> None:
    """Set ``cxx_member_args`` on every C++ scheme call: one
    :class:`MemberArg` per array member of each derived-type argument, so
    the planner tracks members reached through descriptors (and knows when
    host-side C++ code writes them)."""
    from generator.suite_resolver import iter_phase_calls
    by_path = {e.access_path: e for e in host_dict.values()
               if getattr(e, 'access_path', None)}
    for group in suite_res.groups:
        for items in group.phase_calls.values():
            for call in iter_phase_calls(items):
                if not is_cxx(scheme_store, call.scheme_name):
                    continue
                specs = {s.name: s for s in _specs(
                    scheme_store, call.scheme_name, call.phase, kind_types,
                    ddt_index)}
                members = []
                for arg in call.args:
                    spec = specs.get(arg.scheme_local_name)
                    if spec is None or spec.kind != 'ddt':
                        continue
                    for m in spec.ddt.members:
                        if not m.rank:
                            continue      # scalars: read on the host
                        path = '{}%{}'.format(arg.call_expr, m.name)
                        entry = by_path.get(path)
                        if entry is None:
                            if spec.device:
                                raise CCPPError(
                                    "C++ scheme '{}' phase '{}' argument '{}': "
                                    "device-side derived type member '{}' is "
                                    "not a host variable capgen knows; only "
                                    "derived-type instances in host metadata "
                                    "can be device-side".format(
                                        call.scheme_name, call.phase,
                                        arg.scheme_local_name, path))
                            continue
                        members.append(MemberArg(
                            entry.standard_name, arg.scheme_local_name,
                            arg.intent, entry, path, list(entry.dimensions)))
                call.cxx_member_args = members


def _subroutine_header(indent: str, fname: str, names: List[str]) -> List[str]:
    out = ['{}subroutine {}( &'.format(indent, fname)]
    for i, n in enumerate(names):
        out.append('{}    {}{}'.format(indent, n, ', &' if i < len(names) - 1 else ') &'))
    if not names:
        out = ['{}subroutine {}() &'.format(indent, fname)]
    out.append("{}    bind(C, name='{}')".format(indent, fname))
    return out


########################################################################
# C/C++ headers
########################################################################

def write_headers(scheme_store, scheme_names: List[str], kind_types,
                  output_root: str, logger=None, ddt_index=None) -> List[str]:
    """Write ``<scheme>_ccpp.h`` with the C prototypes of every phase of each
    C++ scheme in *scheme_names*.  Returns the paths written."""
    paths = []
    for scheme in sorted(set(scheme_names)):
        if not is_cxx(scheme_store, scheme):
            continue
        phases = [(p, _specs(scheme_store, scheme, p, kind_types, ddt_index))
                  for p in scheme_store.phases_for(scheme)]
        all_specs = [s for _, specs in phases for s in specs]
        guard = '{}_CCPP_H'.format(scheme.upper())
        out = [
            '/* {}_ccpp.h -- generated by ccpp_capgen, do not edit.'.format(scheme),
            ' *',
            ' * C prototypes the CCPP caps call for the C++ scheme {}.'.format(scheme),
            ' * Arrays are column-major (first dimension fastest) and dense.',
            ' * Character arguments are blank-padded to the host length and are',
            ' * not NUL-terminated.',
            ' * Arguments marked [device] are present on the device when the caps',
            ' * are generated with --directive: use present() (OpenACC) or a',
            ' * tofrom map (OpenMP) to reach them; nothing is copied.',
            ' */',
            '#ifndef {}'.format(guard),
            '#define {}'.format(guard),
            '',
        ]
        if any(s.kind == 'logical' for s in all_specs):
            out += ['#ifndef __cplusplus', '#include <stdbool.h>', '#endif', '']
        seen = set()
        for s in all_specs:
            if s.ddt is not None and s.ddt.type_name not in seen:
                seen.add(s.ddt.type_name)
                out += s.ddt.c_lines()
            if s.kind == 'alloc' and s.alloc_type not in seen:
                seen.add(s.alloc_type)
                out += _alloc_c_lines(s)
        out += ['#ifdef __cplusplus', 'extern "C" {', '#endif', '']
        for phase, specs in phases:
            fname = '{}_{}'.format(scheme, phase)
            if specs:
                out.append('void {}('.format(fname))
                variables = scheme_store.variables_for(scheme, phase) or []
                for i, (s, var) in enumerate(zip(specs, variables)):
                    dev = scheme_store.memory_space_for(scheme, phase, var.local_name) == 'device'
                    tail = ',' if i < len(specs) - 1 else ');'
                    notes = (['device'] if dev and var.dimensions else []) + \
                        ([s.note] if s.note else [])
                    note = '  /* {} */'.format('; '.join(
                        '[device]' if n == 'device' else n for n in notes)) if notes else ''
                    out.append('    {}{}{}'.format(s.cpar, tail, note))
            else:
                out.append('void {}(void);'.format(fname))
            out.append('')
        out += ['#ifdef __cplusplus', '}', '#endif', '', '#endif', '']
        path = os.path.join(output_root, '{}_ccpp.h'.format(scheme))
        with open(path, 'w') as fh:
            fh.write('\n'.join(out))
        if logger is not None:
            logger.info("Wrote %s", path)
        paths.append(path)
    return paths


def group_phase_calls(resolved_group, scheme_store) -> Dict[str, List[str]]:
    """scheme name -> phases called, for the C++ schemes of one group."""
    from generator.suite_resolver import iter_phase_calls
    out: Dict[str, List[str]] = {}
    for phase, items in resolved_group.phase_calls.items():
        for call in iter_phase_calls(items):
            if is_cxx(scheme_store, call.scheme_name):
                phases = out.setdefault(call.scheme_name, [])
                if call.phase not in phases:
                    phases.append(call.phase)
    return out
