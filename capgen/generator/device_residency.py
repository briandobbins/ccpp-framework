#!/usr/bin/env python3
"""GPU data movement: residency planning for generated caps.

Proof of concept; see ``doc/gpu_data_movement_plan.md`` for the design.

Model
-----
* Every array that some *device-side* scheme argument touches is
  **tracked**.  Device-side means the argument's effective
  ``memory_space`` is ``device`` (the argument's own value, else its
  ``[ccpp-arg-table]`` default).  Scalars are never tracked: OpenACC
  passes them into kernels by value.
* A tracked array gets one device copy for the whole run (created when the
  suite is initialised, deleted when it is finalised), unless the host
  declares ``memory_space = device`` on it, in which case the host owns
  the device copy.
* Each tracked array has a **handoff state**: the set of locations
  (``H`` = host, ``D`` = device) holding valid data whenever the host has
  control.  It comes from the host variable's ``host_access`` promise:

  ========================  ============
  variable                  handoff
  ========================  ============
  host, ``read_write``      ``{H}``
  host, ``read``            ``{H, D}``
  host, ``none``            ``{D}``
  host ``memory_space=device``  ``{D}``
  suite interstitial        ``{D}``
  ========================  ============

* Every group phase subroutine starts from the handoff state and restores
  it before returning, so groups are independent of each other.  Inside a
  phase the planner walks the calls in order and emits an update only when
  a call is about to read a copy that is not valid.

The planner only *decides*; the cap writers print what it decides.  When
``--directive`` is not given no plan exists and the writers produce the
same code they always have.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from metadata.parse_tools import CCPPError, check_fortran_intrinsic
from generator.suite_resolver import (
    ResolvedArg, ResolvedCall, ResolvedSubcycle, iter_phase_calls,
)

H = 'H'
D = 'D'

#: Handoff state for each ``host_access`` value.
_HANDOFF_FOR_ACCESS = {
    'read_write': frozenset({H}),
    'read': frozenset({H, D}),
    'none': frozenset({D}),
}

_PHASES_IN_ORDER = ('register', 'init', 'timestep_init', 'run',
                    'timestep_final', 'final')


########################################################################
# Plan data structures
########################################################################

@dataclass(frozen=True)
class DeviceOptions:
    """Command-line options that shape the plan."""
    directive: str = 'acc'
    memory_model: str = 'discrete'
    debug: bool = False
    #: --device-trace: print presence and shape of every device-side array
    #: before each GPU call and of every updated slice.
    trace: bool = False
    #: --device-sync-debug: copy every device-side result back to the host
    #: right after the call that produced it, so host memory mirrors the
    #: device (for host-side debuggers and memory-comparison tools).
    sync_debug: bool = False

    @property
    def unified(self) -> bool:
        return self.memory_model == 'unified'


@dataclass
class TrackedArray:
    """One array that some device-side scheme argument touches."""
    base_expr: str
    #: ``'host'`` (CCPP creates the device copy), ``'host_managed'`` (the
    #: host owns the device copy) or ``'suite'`` (framework interstitial).
    owner: str
    handoff: FrozenSet[str]
    standard_names: Set[str] = field(default_factory=set)
    module_name: Optional[str] = None
    root_symbol: str = ''
    type_: str = ''
    kind: str = ''
    rank: int = 0
    #: DDT parent access paths, outermost first (host-owned arrays only).
    parents: List[str] = field(default_factory=list)
    host_access: str = ''
    #: Host ``active`` condition, translated to Fortran local names ('' if
    #: always active), and the module symbols it references.  Whole-array
    #: create/delete/sync statements are wrapped in it.
    active_local: str = ''
    active_uses: Dict[str, Set[str]] = field(default_factory=dict)
    #: (group, phase, scheme, intent, memory_space) for every access.
    uses: List[Tuple[str, str, str, str, str]] = field(default_factory=list)


@dataclass
class Update:
    """One generated data-movement statement.

    ``kind`` is ``'device'`` (copy host to device), ``'self'`` (copy device
    to host) or ``'wait'`` (unified memory: wait for device work).
    """
    kind: str
    expr: str
    guard: str = ''
    reason: str = ''


@dataclass
class Check:
    """One generated runtime check (``--device-debug``).

    ``kind`` is ``'present'`` (the array must have a device copy),
    ``'coherent'`` (host and device copies of *expr* must agree),
    ``'trace'`` (print presence, shape, bounds and values, never fails) or
    ``'scalars'`` (print the scalar arguments of a device-side call).
    """
    kind: str
    expr: str
    array: Optional[TrackedArray]
    guard: str = ''
    where: str = ''
    #: 'trace': where the plan expects valid data at this point.
    plan_state: str = ''
    #: 'scalars': (name, Fortran expression) of scalar arguments.
    scalars: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class PackArg:
    """A device-side array argument of a C++ scheme passed through a dense
    device scratch array (C++ receives a raw pointer, so a non-contiguous
    column chunk cannot be passed in place)."""
    local_name: str
    base: str
    subscript: str
    array: TrackedArray
    reads: bool
    writes: bool
    #: Runtime condition of an optional argument ('' = always passed).
    #: When false the scratch array stays unallocated, so the argument is
    #: absent (NULL in C++).
    guard: str = ''
    #: Logical array: crosses as 0/1 bytes (see cxx_schemes).
    logical: bool = False


@dataclass
class PhasePlan:
    """Plan for one group phase subroutine (e.g. ``physics_run``)."""
    group_name: str
    phase: str
    #: id(ResolvedCall) -> updates before / after the call.
    pre: Dict[int, List[Update]] = field(default_factory=dict)
    post: Dict[int, List[Update]] = field(default_factory=dict)
    #: id(ResolvedSubcycle) -> updates before the loop.
    loop_entry: Dict[int, List[Update]] = field(default_factory=dict)
    #: Updates restoring the handoff state at the end of the subroutine.
    exit: List[Update] = field(default_factory=list)
    #: Debug checks.
    pre_checks: Dict[int, List[Check]] = field(default_factory=dict)
    #: id(ResolvedCall) -> --device-trace lines, printed right before the
    #: call (after its updates).
    trace_checks: Dict[int, List[Check]] = field(default_factory=dict)
    #: id(ResolvedCall) -> coherence checks after the call (an argument the
    #: call only reads must leave both copies equal: catches a scheme that
    #: writes an ``intent(in)`` argument).
    post_checks: Dict[int, List[Check]] = field(default_factory=dict)
    entry_checks: List[Check] = field(default_factory=list)
    exit_checks: List[Check] = field(default_factory=list)
    #: id(ResolvedCall) -> device-side array arguments of a C++ scheme
    #: call that are packed into dense device scratch arrays.
    cxx_pack: Dict[int, List[PackArg]] = field(default_factory=dict)
    #: id(ResolvedCall) of every call with a device-side argument (used by
    #: --timer-style to wait for device work before stopping its timer).
    device_calls: Set[int] = field(default_factory=set)
    #: Print a trace line before every update (--device-trace).
    trace: bool = False
    #: Unified memory model (no presence query in traces).
    unified: bool = False
    #: Directive language: 'acc' (OpenACC) or 'omp' (OpenMP target).
    directive: str = 'acc'

    def all_updates(self) -> Iterable[Update]:
        for lists in (self.pre, self.post, self.loop_entry):
            for ups in lists.values():
                yield from ups
        yield from self.exit

    @property
    def is_empty(self) -> bool:
        return not (any(True for _ in self.all_updates())
                    or self.entry_checks or self.exit_checks
                    or any(self.pre_checks.values())
                    or any(self.trace_checks.values())
                    or any(self.post_checks.values())
                    # A C++ call whose only device work is its scratch
                    # copies still needs them.
                    or any(self.cxx_pack.values()))


@dataclass
class SuitePlan:
    """The complete residency plan for one suite."""
    suite_name: str
    options: DeviceOptions
    arrays: Dict[str, TrackedArray] = field(default_factory=dict)
    phases: Dict[Tuple[str, str], PhasePlan] = field(default_factory=dict)

    def phase_plan(self, group_name: str, phase: str) -> Optional[PhasePlan]:
        return self.phases.get((group_name, phase))

    def created_host_arrays(self) -> List[TrackedArray]:
        """Host arrays whose device copy CCPP creates and deletes."""
        if self.options.unified:
            return []
        return [a for a in self.arrays.values()
                if a.owner in ('host', 'constituent')]

    def created_suite_arrays(self) -> List[TrackedArray]:
        """Interstitials whose device copy CCPP creates and deletes."""
        if self.options.unified:
            return []
        return [a for a in self.arrays.values() if a.owner == 'suite']

    def scheme_allocated_arrays(self) -> List[TrackedArray]:
        """Interstitials allocated after init starts: by a device-side C++
        scheme (its allocation callback creates the device copy) or by CCPP
        right after the init-phase scheme that sets their size (the group
        cap creates the device copy).  <suite>_final deletes the copy."""
        if self.options.unified:
            return []
        return [a for a in self.arrays.values()
                if a.owner in ('scheme_alloc', 'deferred')]

    def synced_host_arrays(self, direction: str) -> List[TrackedArray]:
        """Host-visible arrays the whole-array sync entry points copy.

        ``'device'``: arrays whose handoff state includes the device
        (``host_access = none`` or ``read``) -- their device copy must
        receive the host's initial values.  ``'host'``: arrays valid only
        on the device (``host_access = none``).  Every other array is
        already valid on the host whenever the host has control, and
        copying device data over it would destroy newer host values.
        """
        out = []
        for a in self.arrays.values():
            if a.owner not in ('host', 'constituent'):
                continue
            if direction == 'device' and D in a.handoff:
                out.append(a)
            elif direction == 'host' and H not in a.handoff:
                out.append(a)
        return out

    def update_count(self) -> int:
        return sum(1 for p in self.phases.values() for _ in p.all_updates())


########################################################################
# Helpers
########################################################################

def _err(suite_name: str, msg: str) -> CCPPError:
    return CCPPError("GPU data movement (suite '{}'): {}".format(
        suite_name, msg))


def _is_device(scheme_store, call: ResolvedCall, arg: ResolvedArg) -> bool:
    return scheme_store.memory_space_for(
        call.scheme_name, call.phase, arg.scheme_local_name) == 'device'


def _is_array(arg: ResolvedArg) -> bool:
    return bool(arg.scheme_dimensions)


def _reads(intent: str) -> bool:
    return intent in ('in', 'inout')


def _writes(intent: str) -> bool:
    return intent in ('out', 'inout')


def _guard(arg: ResolvedArg) -> str:
    """Runtime condition under which an optional argument is passed.

    Non-optional arguments are always passed (a required argument whose
    host ``active`` condition is false fails before the call), so they
    need no guard.
    """
    if arg.is_optional and arg.active_local:
        return arg.active_local
    return ''


#: Constituent state arrays a device-side scheme may reach (members of the
#: generator-owned ``ccpp_model_constituents_obj``), dimensioned
#: (columns, levels, constituents).
_CONST_ARRAY_MEMBERS = ('vars_layer', 'vars_layer_tend')
_CONST_OBJECT = 'ccpp_model_constituents_obj'


def _is_constituent_state(arg: ResolvedArg) -> bool:
    if arg.source != 'constituent' or arg.is_constituent_arg:
        return False
    head, _, member = arg.base_expr.rpartition('%')
    return member in _CONST_ARRAY_MEMBERS and head.startswith(_CONST_OBJECT)


def _split_subscript(sub: str) -> List[str]:
    """``'(a, b(1,2), c)'`` -> ``['a', 'b(1,2)', 'c']`` (top-level commas)."""
    sub = sub.strip()
    if sub.startswith('(') and sub.endswith(')'):
        sub = sub[1:-1]
    parts, depth, cur = [], 0, ''
    for ch in sub:
        if ch == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
            continue
        depth += ch == '('
        depth -= ch == ')'
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _walk_key(arg: ResolvedArg) -> Tuple[str, bool]:
    """Return ``(walk_key, partial)`` for an argument.

    Most arguments are tracked by their exact call expression.  A
    constituent state array is tracked as one unit covering every
    constituent of the slice (its last subscript replaced by ``:``),
    because schemes reach it both one constituent at a time and all at
    once; *partial* is True when this argument covers a single constituent.
    """
    if not _is_constituent_state(arg):
        return arg.call_expr, False
    parts = _split_subscript(arg.subscript)
    if not parts:
        return arg.call_expr, False
    partial = parts[-1] != ':'
    key = '{}({})'.format(arg.base_expr, ', '.join(parts[:-1] + [':']))
    return key, partial


def _ddt_parents(access_path: str) -> List[str]:
    """``'a%b%c'`` -> ``['a', 'a%b']``."""
    parts = access_path.split('%')
    return ['%'.join(parts[:i]) for i in range(1, len(parts))]


def _iter_items_calls(phase_items) -> Iterable[ResolvedCall]:
    return iter_phase_calls(phase_items)


def _call_args(call: ResolvedCall):
    """A call's arguments plus, for a C++ scheme, the array members it
    reaches through derived-type descriptors (cxx_schemes.MemberArg)."""
    return list(call.args) + list(getattr(call, 'cxx_member_args', None) or ())


def _cxx_allocated(suite_res, scheme_store) -> Set[str]:
    """Scheme-allocated interstitials that only device-side C++ schemes
    allocate.  Their device copy is created by the C++ allocation callback
    (cxx_schemes), so they can be tracked; other scheme-allocated
    interstitials cannot."""
    ok: Dict[str, bool] = {}
    for group in suite_res.groups:
        for items in group.phase_calls.values():
            for call in iter_phase_calls(items):
                for arg in call.args:
                    sv = arg.suite_var
                    if arg.source != 'suite' or sv is None or not sv.allocatable:
                        continue
                    if not _writes(arg.intent):
                        ok.setdefault(arg.base_expr, True)
                        continue
                    good = (getattr(call, 'language', 'fortran') == 'c++'
                            and _is_device(scheme_store, call, arg))
                    ok[arg.base_expr] = ok.get(arg.base_expr, True) and good
    return {k for k, v in ok.items() if v}


########################################################################
# Pass 1: find and validate tracked arrays
########################################################################

def _collect_tracked(suite_res, scheme_store, host_dict) -> Dict[str, TrackedArray]:
    sname = suite_res.suite_name
    arrays: Dict[str, TrackedArray] = {}

    # Suite-level <init>/<final> schemes run inside <suite>_init/_final,
    # outside any group; keep them host-side in the PoC.
    for call in (suite_res.suite_init_call, suite_res.suite_final_call):
        if call is None:
            continue
        for arg in call.args:
            if _is_device(scheme_store, call, arg) and _is_array(arg):
                raise _err(sname,
                    "scheme '{}' (suite-level <init>/<final>) declares "
                    "memory_space = device for '{}'; device-side "
                    "arguments are not supported in suite-level "
                    "<init>/<final> schemes".format(
                        call.scheme_name, arg.scheme_local_name))

    cxx_allocated = _cxx_allocated(suite_res, scheme_store)
    for group in suite_res.groups:
        for phase, items in group.phase_calls.items():
            for call in _iter_items_calls(items):
                for arg in _call_args(call):
                    if not _is_device(scheme_store, call, arg):
                        continue
                    if not _is_array(arg):
                        # Scalars go into kernels by value; nothing to move.
                        continue
                    where = "scheme '{}' phase '{}' argument '{}' ({})".format(
                        call.scheme_name, call.phase, arg.scheme_local_name,
                        arg.standard_name)
                    _validate_device_arg(sname, where, call, arg, cxx_allocated)
                    _add_tracked(arrays, arg, host_dict)

    # Second sweep: record every access (host or device side) to a tracked
    # array, so the walk and the datatable see the whole picture.
    for group in suite_res.groups:
        for phase, items in group.phase_calls.items():
            for call in _iter_items_calls(items):
                for arg in _call_args(call):
                    arr = arrays.get(arg.base_expr)
                    if arr is None:
                        continue
                    arr.standard_names.add(arg.standard_name)
                    arr.uses.append((
                        group.group_name, call.phase, call.scheme_name,
                        arg.intent,
                        'device' if _is_device(scheme_store, call, arg)
                        else 'host'))
    return arrays


def _validate_device_arg(sname: str, where: str, call: ResolvedCall,
                         arg: ResolvedArg, cxx_allocated=()) -> None:
    if call.phase == 'register':
        raise _err(sname, "{}: device-side arguments are not supported in "
                          "the register phase (it runs before device "
                          "copies are created)".format(where))
    if arg.source == 'constituent' or arg.is_constituent_arg:
        if not _is_constituent_state(arg):
            raise _err(sname, "{}: only the constituent state arrays "
                              "(constituent values and tendencies) are "
                              "supported on the device; constituent "
                              "properties are not".format(where))
        if not arg.base_expr.startswith(_CONST_OBJECT + '(1)%'):
            raise _err(sname, "{}: multi-instance constituent objects are "
                              "not supported on the device in this proof "
                              "of concept".format(where))
    if arg.source == 'control':
        raise _err(sname, "{}: device-side arrays from the host control "
                          "table are not supported".format(where))
    if arg.needs_transform:
        raise _err(sname, "{}: needs a unit, kind or vertical-flip "
                          "transform, which runs on the host; transforms "
                          "of device-side arguments are not supported in "
                          "this proof of concept.  Make the host and "
                          "scheme metadata agree".format(where))
    type_ = (arg.host_entry.type if arg.host_entry is not None
             else arg.suite_var.type_ if arg.suite_var is not None
             else 'real' if _is_constituent_state(arg) else '')
    if check_fortran_intrinsic(type_.strip(), error=False) is None:
        raise _err(sname, "{}: derived-type arrays ('{}') need a deep "
                          "copy and are not supported as device-side "
                          "arguments in this proof of concept".format(
                              where, type_))
    if arg.source == 'host':
        if '(' in arg.base_expr:
            raise _err(sname, "{}: host variable '{}' is reached through an "
                              "instance or thread subscript; not supported "
                              "in this proof of concept".format(
                                  where, arg.base_expr))
    elif arg.source == 'suite':
        sv = arg.suite_var
        if sv is not None and sv.inst_access != '(1)':
            raise _err(sname, "{}: multi-instance interstitials are not "
                              "supported in this proof of concept".format(
                                  where))
        if sv is not None and sv.allocatable \
                and arg.base_expr not in cxx_allocated:
            raise _err(sname, "{}: interstitial '{}' is allocated by a "
                              "scheme; device-side use of scheme-allocated "
                              "interstitials is supported only when every "
                              "scheme that allocates it is a device-side C++ "
                              "scheme (whose allocation callback creates the "
                              "device copy)".format(where, sv.local_name))


def _active_uses(active: str, host_dict) -> Dict[str, Set[str]]:
    """Module symbols referenced by a host ``active`` expression."""
    uses: Dict[str, Set[str]] = {}
    for tok in re.findall(r'[a-z_][a-z0-9_]*', active.lower()):
        entry = host_dict.get(tok)
        if entry is not None and entry.module_name:
            root = entry.access_path.split('%')[0].split('(')[0]
            uses.setdefault(entry.module_name, set()).add(root)
    return uses


def _add_tracked(arrays: Dict[str, TrackedArray], arg: ResolvedArg,
                 host_dict=None) -> None:
    key = arg.base_expr
    if key in arrays:
        return
    if _is_constituent_state(arg):
        # Framework-owned, but the host reads and writes constituents
        # between physics calls (the dycore advects them), so they keep
        # the default read_write handoff.
        arrays[key] = TrackedArray(
            base_expr=key, owner='constituent', handoff=frozenset({H}),
            module_name=arg.constituent_module_name,
            root_symbol=_CONST_OBJECT, type_='real', kind='kind_phys',
            rank=3, parents=[_CONST_OBJECT], host_access='read_write',
        )
        return
    if arg.source == 'suite':
        sv = arg.suite_var
        arrays[key] = TrackedArray(
            # Scheme-allocated: the device copy is created by the C++
            # allocation callback, not at init.
            base_expr=key,
            owner=('scheme_alloc' if (sv is not None and sv.allocatable)
                   else 'deferred' if (sv is not None and getattr(sv, 'allocated_after', ''))
                   else 'suite'),
            handoff=frozenset({D}),
            module_name=arg.module_name, root_symbol=arg.root_symbol,
            type_=sv.type_ if sv else '', kind=sv.kind if sv else '',
            rank=len(sv.dimensions) if sv else len(arg.scheme_dimensions),
        )
        return
    he = arg.host_entry
    if he.memory_space == 'device':
        owner, handoff = 'host_managed', frozenset({D})
    else:
        owner, handoff = 'host', _HANDOFF_FOR_ACCESS[he.host_access]
    arrays[key] = TrackedArray(
        base_expr=key, owner=owner, handoff=handoff,
        module_name=arg.module_name, root_symbol=arg.root_symbol,
        type_=he.type, kind=he.kind, rank=len(he.dimensions),
        parents=_ddt_parents(key), host_access=he.host_access,
        active_local=arg.active_local if he.active else '',
        active_uses=_active_uses(he.active, host_dict or {}) if he.active else {},
    )


########################################################################
# Pass 2: walk each group phase
########################################################################

class _State:
    """Valid locations per walk key (the argument's call expression).

    A key missing from the map is at its array's handoff state.
    """

    def __init__(self, handoff_of):
        self._handoff_of = handoff_of
        self.valid: Dict[str, FrozenSet[str]] = {}

    def get(self, wk: str) -> FrozenSet[str]:
        return self.valid.get(wk, self._handoff_of(wk))

    def set(self, wk: str, locs: FrozenSet[str]) -> None:
        self.valid[wk] = frozenset(locs)

    def copy(self) -> '_State':
        new = _State(self._handoff_of)
        new.valid = dict(self.valid)
        return new

    def meet(self, other: '_State') -> '_State':
        new = _State(self._handoff_of)
        for wk in set(self.valid) | set(other.valid):
            new.valid[wk] = self.get(wk) & other.get(wk)
        return new

    def same_as(self, other: '_State') -> bool:
        keys = set(self.valid) | set(other.valid)
        return all(self.get(k) == other.get(k) for k in keys)


def _state_label(locs) -> str:
    if locs >= {H, D}:
        return 'valid on host and device'
    if D in locs:
        return 'valid on device'
    if H in locs:
        return 'valid on host'
    return 'no valid copy'


def _scalar_type(arg: ResolvedArg) -> str:
    if arg.host_entry is not None:
        return arg.host_entry.type.strip().lower()
    if arg.suite_var is not None:
        return arg.suite_var.type_.strip().lower()
    return ''


def _to(side: str) -> str:
    return 'device' if side == D else 'self'


class _PhaseWalker:
    """Walks one group phase and fills a :class:`PhasePlan`."""

    def __init__(self, suite_name, plan: PhasePlan, arrays, scheme_store,
                 options: DeviceOptions):
        self.suite_name = suite_name
        self.plan = plan
        self.arrays = arrays
        self.scheme_store = scheme_store
        self.options = options
        #: walk key (call expression) -> tracked array
        self.wk_array: Dict[str, TrackedArray] = {}
        #: walk key -> active guard of the first optional access ('' if any
        #: access is unconditional)
        self.wk_guard: Dict[str, str] = {}
        self.order: List[str] = []

    # -- access extraction ---------------------------------------------
    def accesses(self, call: ResolvedCall):
        """Yield ``(walk_key, side, intent, guard, arg)`` for each argument
        of *call* that touches a tracked array."""
        for arg in _call_args(call):
            arr = self.arrays.get(arg.base_expr)
            if arr is None:
                continue
            wk, partial = _walk_key(arg)
            if wk not in self.wk_array:
                self.wk_array[wk] = arr
                self.order.append(wk)
                self.wk_guard[wk] = _guard(arg)
            elif not _guard(arg):
                self.wk_guard[wk] = ''
            side = D if _is_device(self.scheme_store, call, arg) else H
            intent = arg.intent
            if partial and intent == 'out':
                # Writing one constituent of a unit tracked whole: the rest
                # of the unit must already be valid on this side, so plan
                # it like a read-modify-write.
                intent = 'inout'
            yield wk, side, intent, _guard(arg), arg

    def handoff(self, wk: str) -> FrozenSet[str]:
        return self.wk_array[wk].handoff

    # -- the walk -------------------------------------------------------
    def walk(self, items, state: _State, emit: bool) -> None:
        for item in items:
            if isinstance(item, ResolvedCall):
                self._call(item, state, emit)
            elif isinstance(item, ResolvedSubcycle):
                self._subcycle(item, state, emit)

    def _call(self, call: ResolvedCall, state: _State, emit: bool) -> None:
        pre: List[Update] = []
        post: List[Update] = []
        checks: List[Check] = []
        post_checks: List[Check] = []
        traces: List[Check] = []
        packs: List[PackArg] = []
        where = 'before {}_{}'.format(call.scheme_name, call.phase)
        device_call = False
        for wk, side, intent, guard, arg in list(self.accesses(call)):
            cur = state.get(wk)
            if _reads(intent) and not cur:
                raise _err(self.suite_name, "internal: '{}' read by {} with "
                           "no valid copy".format(wk, call.scheme_name))
            if side == D:
                device_call = True
                logical = self.wk_array[wk].type_.strip().lower() == 'logical'
                if (getattr(call, 'language', 'fortran') == 'c++'
                        and not getattr(arg, 'is_ddt_member', False)
                        and self.wk_array[wk].owner != 'scheme_alloc'
                        and (len(arg.scheme_dimensions) >= 2 or logical)):
                    # Rank 2+: a column chunk is not contiguous.  Logical:
                    # converted to 0/1 bytes on the device.
                    packs.append(PackArg(arg.scheme_local_name, arg.base_expr,
                                         arg.subscript, self.wk_array[wk],
                                         _reads(arg.intent), _writes(arg.intent),
                                         guard=guard, logical=logical))
            if _reads(intent) and side not in cur:
                pre.append(Update(_to(side), wk, guard,
                                  "{} reads it on the {}".format(
                                      call.scheme_name,
                                      'device' if side == D else 'host')))
                if not guard:
                    cur = cur | {side}
                    state.set(wk, cur)
            if _writes(intent):
                if not guard:
                    state.set(wk, frozenset({side}))
                else:
                    # Conditional write: restore every other copy that was
                    # valid, so the state is the same on both paths.
                    for other in sorted(cur - {side}):
                        post.append(Update(_to(other), wk, guard,
                                           "keep copies coherent after a "
                                           "conditional write by {}".format(
                                               call.scheme_name)))
                    state.set(wk, cur)
            debug = self.options.debug and not self.options.unified
            if side == D and debug:
                checks.append(Check('present', self.wk_array[wk].base_expr,
                                    self.wk_array[wk], guard,
                                    '{}_{}'.format(call.scheme_name,
                                                   call.phase)))
            if side == D and self.options.trace:
                # The state the call will see: after this argument's
                # pre-update (or before its write, for write-only access).
                seen = cur | ({side} if (_reads(intent) and not guard) else set())
                traces.append(Check('trace', wk, self.wk_array[wk], guard,
                                    where, plan_state=_state_label(seen)))
            if (side == D and _writes(intent) and self.options.sync_debug
                    and not self.options.unified
                    and H not in state.get(wk)):
                # Mirror the device result on the host at once.  The plan
                # then knows both copies are valid, so later CPU reads need
                # no further transfer.
                post.append(Update('self', wk, guard,
                                   'debug sync: mirror the result of {} on '
                                   'the host'.format(call.scheme_name)))
                if not guard:
                    state.set(wk, state.get(wk) | {H})
            if (debug and not _writes(intent)
                    and state.get(wk) >= {H, D}):
                post_checks.append(Check(
                    'coherent', wk, self.wk_array[wk], guard,
                    'after {}_{} (declared intent(in))'.format(
                        call.scheme_name, call.phase)))
        if emit:
            if pre:
                self.plan.pre[id(call)] = pre
            if post:
                self.plan.post[id(call)] = post
            if checks:
                self.plan.pre_checks[id(call)] = checks
            if post_checks:
                self.plan.post_checks[id(call)] = post_checks
            if self.options.trace and device_call:
                scalars = [(a.scheme_local_name, a.call_expr) for a in call.args
                           if not a.scheme_dimensions and _reads(a.intent)
                           and _scalar_type(a) in ('integer', 'real', 'logical')]
                if scalars:
                    traces.append(Check('scalars', '', None, '', where,
                                        scalars=scalars))
            if traces:
                self.plan.trace_checks[id(call)] = traces
            if packs:
                self.plan.cxx_pack[id(call)] = packs
            if device_call:
                self.plan.device_calls.add(id(call))

    def _first_accesses(self, items) -> Dict[str, Tuple[str, str, str]]:
        """walk key -> (side, intent, guard) of its first access in *items*."""
        first: Dict[str, Tuple[str, str, str]] = {}
        for call in iter_phase_calls(items):
            for wk, side, intent, guard, _arg in list(self.accesses(call)):
                if wk not in first:
                    first[wk] = (side, intent, guard)
        return first

    def _subcycle(self, sc: ResolvedSubcycle, state: _State,
                  emit: bool) -> None:
        """Plan a subcycle loop.

        ``e`` is the state on entry to the loop (the pre-loop state plus
        any copies the planner decides to make before the loop).  The
        state at the top of an iteration is the meet of ``e`` and the
        state at the bottom of the previous iteration, computed to a fixed
        point.  A copy is made once before the loop, instead of inside it,
        when the loop body reads it first and keeps it valid.
        """
        first = self._first_accesses(sc.calls)
        write_first = {wk for wk, (_s, intent, guard) in first.items()
                       if intent == 'out' and not guard}
        try:
            min_iters = 1 if int(sc.loop) >= 1 else 0
        except ValueError:
            min_iters = 0

        e = state.copy()
        entry_updates: List[Update] = []

        def add_before_loop(wk, sides, reason):
            have = e.get(wk)
            if not have:
                raise _err(self.suite_name, "internal: no valid copy of "
                                            "'{}' before a subcycle".format(wk))
            for side in sorted(set(sides) - have):
                entry_updates.append(Update(_to(side), wk, '', reason))
                have = have | {side}
            e.set(wk, have)

        for _ in range(64):
            # Fixed point for the state at the top of an iteration.  If a
            # key read in the body would have no copy that is valid on
            # every iteration, stop and make the loop's copies valid before
            # the loop instead.
            entry = e.copy()
            diverged: Dict[str, FrozenSet[str]] = {}
            for _ in range(64):
                st = entry.copy()
                self.walk(sc.calls, st, emit=False)
                new = e.meet(st)
                for wk in write_first:
                    # Overwritten before any read: the top-of-loop state
                    # of this key does not affect the body.
                    new.set(wk, st.get(wk))
                diverged = {wk: st.get(wk) for wk in first
                            if wk not in write_first and not new.get(wk)}
                if diverged or new.same_as(entry):
                    break
                entry = new
            if diverged:
                for wk, sides in diverged.items():
                    add_before_loop(wk, sides,
                                    'keep the subcycle state consistent')
                continue
            exit_state = entry.copy()
            self.walk(sc.calls, exit_state, emit=False)

            changed = False
            for wk, (side, intent, guard) in first.items():
                if wk in write_first:
                    continue
                top = entry.get(wk)
                if (_reads(intent) and not guard and side not in top
                      and side in exit_state.get(wk)
                      and side not in e.get(wk)):
                    # The body reads it here first and keeps it valid:
                    # copy once before the loop, not every iteration.
                    add_before_loop(wk, {side}, 'first read in the subcycle')
                    changed = True
            if min_iters >= 1:
                after = exit_state
            else:
                after = e.meet(exit_state)
                for wk, locs in list(after.valid.items()):
                    if not locs:
                        # Zero iterations are possible: the copies valid
                        # after the loop must not depend on the count.
                        add_before_loop(wk, exit_state.get(wk),
                                        'the subcycle may run zero times')
                        changed = True
            if not changed:
                break
        else:  # pragma: no cover - the lattice is finite
            raise _err(self.suite_name, "subcycle analysis did not converge")

        self.walk(sc.calls, entry.copy(), emit=emit)
        if emit and entry_updates:
            self.plan.loop_entry[id(sc)] = entry_updates
        state.valid = after.valid

    # -- phase boundaries -----------------------------------------------
    def finish(self, state: _State) -> None:
        for wk in self.order:
            have = state.get(wk)
            need = self.handoff(wk)
            guard = self.wk_guard.get(wk, '')
            for side in sorted(need - have):
                self.plan.exit.append(Update(
                    _to(side), wk, guard,
                    'restore the handoff state ({})'.format(
                        self._handoff_label(wk))))
                have = have | {side}
            if self.options.debug and not self.options.unified:
                if have >= {H, D}:
                    self.plan.exit_checks.append(Check(
                        'coherent', wk, self.wk_array[wk], guard,
                        'exit of {}_{}'.format(self.plan.group_name,
                                               self.plan.phase)))
        if self.options.debug and not self.options.unified:
            for wk in self.order:
                if self.handoff(wk) >= {H, D}:
                    self.plan.entry_checks.append(Check(
                        'coherent', wk, self.wk_array[wk],
                        self.wk_guard.get(wk, ''),
                        'entry of {}_{} (host_access = read)'.format(
                            self.plan.group_name, self.plan.phase)))

    def _handoff_label(self, wk: str) -> str:
        arr = self.wk_array[wk]
        if arr.owner == 'constituent':
            return 'constituent state'
        if arr.owner == 'suite':
            return 'interstitial'
        if arr.owner == 'scheme_alloc':
            return 'scheme-allocated interstitial'
        if arr.owner == 'deferred':
            return 'interstitial allocated after the scheme that sizes it'
        if arr.owner == 'host_managed':
            return 'host memory_space = device'
        return 'host_access = {}'.format(arr.host_access)


def _unify(plan: PhasePlan) -> None:
    """Rewrite a discrete plan for unified memory.

    One memory: copying to the device is never needed, and copying back
    becomes a wait so that asynchronous kernels finish before CPU code
    reads their results.  At most one wait per position.
    """
    def convert(ups: List[Update]) -> List[Update]:
        out: List[Update] = []
        for u in ups:
            if u.kind == 'self' and not any(o.kind == 'wait' for o in out):
                out.append(Update('wait', '', '', 'CPU code reads GPU results'))
        return out

    for table in (plan.pre, plan.post, plan.loop_entry):
        for key in list(table):
            new = convert(table[key])
            if new:
                table[key] = new
            else:
                del table[key]
    plan.exit = convert(plan.exit)


def _guard_scheme_allocated(plan: PhasePlan,
                            arrays: Dict[str, TrackedArray]) -> None:
    """Updates and checks of scheme-allocated interstitials run only while
    the array is allocated (it is allocated, with its device copy, by the
    C++ scheme's allocation callback).  *arrays*: walk key -> array."""
    def guarded(guard: str, arr) -> str:
        if arr is None or arr.owner not in ('scheme_alloc', 'deferred'):
            return guard
        cond = 'allocated({})'.format(arr.base_expr)
        return '({}) .and. {}'.format(guard, cond) if guard else cond

    for table in (plan.pre_checks, plan.trace_checks, plan.post_checks):
        for checks in table.values():
            for c in checks:
                c.guard = guarded(c.guard, c.array)
    for c in plan.entry_checks + plan.exit_checks:
        c.guard = guarded(c.guard, c.array)
    for u in plan.all_updates():
        u.guard = guarded(u.guard, arrays.get(u.expr))


########################################################################
# Public entry point
########################################################################

def plan_suite_residency(suite_res, scheme_store, host_dict,
                         options: DeviceOptions, logger=None) -> SuitePlan:
    """Build the :class:`SuitePlan` for one resolved suite."""
    if options.directive not in ('acc', 'omp'):
        raise CCPPError("Unsupported --directive '{}'".format(options.directive))
    plan = SuitePlan(suite_name=suite_res.suite_name, options=options)
    plan.arrays = _collect_tracked(suite_res, scheme_store, host_dict)

    for group in suite_res.groups:
        for phase in _PHASES_IN_ORDER:
            items = group.phase_calls.get(phase)
            if not items:
                continue
            pplan = PhasePlan(group_name=group.group_name, phase=phase,
                              trace=options.trace, unified=options.unified,
                              directive=options.directive)
            walker = _PhaseWalker(suite_res.suite_name, pplan, plan.arrays,
                                  scheme_store, options)
            # Register every walk key up front so handoff lookups work.
            for call in iter_phase_calls(items):
                list(walker.accesses(call))
            if not walker.order:
                continue
            _check_slices(suite_res.suite_name, group.group_name, phase,
                          walker)
            state = _State(walker.handoff)
            walker.walk(items, state, emit=True)
            walker.finish(state)
            _guard_scheme_allocated(pplan, walker.wk_array)
            if options.unified:
                _unify(pplan)
            if not pplan.is_empty:
                plan.phases[(group.group_name, phase)] = pplan

    if logger is not None:
        n_dev = sum(1 for a in plan.arrays.values())
        logger.info(
            "GPU data movement for suite '%s': %d tracked array(s), "
            "%d update statement(s) in %d group phase(s) [%s memory]",
            suite_res.suite_name, n_dev, plan.update_count(),
            len(plan.phases), options.memory_model)
    return plan


def _check_slices(suite_name, group_name, phase, walker: _PhaseWalker) -> None:
    """Reject one standard name reached through two different slices in
    one phase subroutine: the per-slice state would be ambiguous."""
    by_array: Dict[str, Set[str]] = {}
    for wk, arr in walker.wk_array.items():
        by_array.setdefault(arr.base_expr, set()).add(wk)
    for base, wks in by_array.items():
        stds = walker.wk_array[next(iter(wks))].standard_names
        if len(wks) > 1 and len(stds) <= 1:
            raise _err(suite_name,
                "array '{}' is accessed through different slices ({}) in "
                "group '{}' phase '{}'; not supported in this proof of "
                "concept".format(base, ', '.join(sorted(wks)), group_name,
                                 phase))
