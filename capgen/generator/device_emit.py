#!/usr/bin/env python3
"""GPU data movement: Fortran emission helpers.

Prints what :mod:`generator.device_residency` decided.  Every function here
is only reached when a suite has a residency plan (``--directive`` given);
the existing writers call into this module behind that check so that their
output is unchanged otherwise.  See ``doc/gpu_data_movement_plan.md``.
"""

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from generator.device_residency import Check, SuitePlan, TrackedArray, Update

#: Marker on every generated comment, so the data-movement code is easy to
#: find in generated caps.
TAG = 'ccpp-gpu'

#: Free-form Fortran allows 132 columns; stay well inside it.
_MAX_LINE = 120


########################################################################
# Low-level printers
########################################################################

class _Spelling:
    """How each planned statement is written for one directive language."""

    def __init__(self, sentinel, create, delete, to_device, to_host, wait,
                 present, uses):
        self.sentinel = sentinel
        self.create = create          # fmt(x) -> directive text
        self.delete = delete
        self.to_device = to_device
        self.to_host = to_host
        self.wait = wait
        self.present = present        # fmt(x) -> logical Fortran expression
        self.uses = uses              # USE lines needed by present()


SPELLINGS = {
    'acc': _Spelling(
        '!$acc',
        'enter data create({})'.format,
        'exit data delete({})'.format,
        'update device({})'.format,
        'update self({})'.format,
        'wait',
        'acc_is_present({})'.format,
        ['use openacc, only: acc_is_present']),
    # OpenMP target.  The presence query takes a C address, and c_loc of a
    # host array is only legal through a TARGET dummy argument, so the
    # query goes through a small generated helper (OMP_PRESENT_HELPER).
    'omp': _Spelling(
        '!$omp',
        'target enter data map(alloc: {})'.format,
        'target exit data map(delete: {})'.format,
        'target update to({})'.format,
        'target update from({})'.format,
        'taskwait',
        'ccpp_gpu_is_present({})'.format,
        []),
}

#: Module procedure emitted into a group cap's ``contains`` section when
#: OpenMP debug or trace code queries presence.  Standard Fortran 2018:
#: the assumed-type, assumed-rank TARGET dummy makes ``c_loc`` legal for
#: any host array.
OMP_PRESENT_HELPER = [
    '',
    '  logical function ccpp_gpu_is_present(a)',
    '    ! {}: is the array a mapped on the default OpenMP device?'.format(TAG),
    '    use omp_lib, only: omp_target_is_present, omp_get_default_device',
    '    use, intrinsic :: iso_c_binding, only: c_loc',
    '    type(*), dimension(..), target, intent(in) :: a',
    '    ccpp_gpu_is_present = &',
    '        omp_target_is_present(c_loc(a), omp_get_default_device()) /= 0',
    '  end function ccpp_gpu_is_present',
]


def needs_omp_present_helper(plan: 'SuitePlan', group_name: str) -> bool:
    """True when *group_name*'s cap queries OpenMP presence."""
    if plan is None or plan.options.directive != 'omp' or plan.options.unified:
        return False
    if not (plan.options.debug or plan.options.trace):
        return False
    return any(g == group_name for (g, _p) in plan.phases)


def spelling(directive: str) -> _Spelling:
    return SPELLINGS[directive]


def _directive(indent: str, text: str, directive: str = 'acc') -> List[str]:
    """Emit one directive line (``!$acc`` / ``!$omp``), continued on further
    sentinel lines (``&`` at the end, sentinel at the start) when long.
    Breaks only at spaces, i.e. between clauses or after list commas."""
    sentinel = SPELLINGS[directive].sentinel
    return _fit_words('{}{} '.format(indent, sentinel), text.split(' '),
                      '{}{}   '.format(indent, sentinel))


def _fit_words(first_prefix: str, words: List[str], cont_prefix: str) -> List[str]:
    """Greedy line fill of *words*, each continued line ending in `` &``."""
    out, cur = [], first_prefix
    for w in words:
        sep = '' if cur in (first_prefix, cont_prefix) else ' '
        if len(cur) + len(sep) + len(w) + 2 > _MAX_LINE and cur.strip() and \
                cur not in (first_prefix, cont_prefix):
            out.append(cur + ' &')
            cur = cont_prefix + w
        else:
            cur = cur + sep + w
    out.append(cur)
    return out


def _fit_statement(line: str) -> List[str]:
    """Continue an ordinary Fortran statement after commas when it is too
    long.  Directive and comment lines are returned unchanged (directives
    are fitted by :func:`_directive`)."""
    if len(line) <= _MAX_LINE or line.lstrip().startswith('!'):
        return [line]
    indent = line[:len(line) - len(line.lstrip())]
    words = line.strip().split(', ')
    words = [w + ',' for w in words[:-1]] + words[-1:]
    return _fit_words(indent, words, indent + '    ')


def _fit(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        out.extend(_fit_statement(ln))
    return out


def _trace_write(indent: str, items: List[Tuple[str, str]]) -> List[str]:
    """``write(error_unit, ...)`` of a trace line.

    *items* are ``('s', text)`` for literal text and ``('v', expr)`` for
    values; one item per continuation line so long array expressions
    never overflow the line limit.
    """
    parts = []
    for kind, val in items:
        parts.append("'{}'".format(val.replace("'", "''")) if kind == 's' else val)
    parts = [p + ',' for p in parts[:-1]] + parts[-1:]
    return _wrap(indent, "write(error_unit, '(*(g0,1x))') ", parts, '')


def _strip_subscript(expr: str) -> str:
    """``a%b(lb:ub, f(1))`` -> ``a%b`` (drops a final parenthesised
    subscript, respecting nesting)."""
    expr = expr.strip()
    if not expr.endswith(')'):
        return expr
    depth = 0
    for i in range(len(expr) - 1, -1, -1):
        depth += expr[i] == ')'
        depth -= expr[i] == '('
        if depth == 0:
            return expr[:i]
    return expr


_ABSENT_CONSEQUENCE = {
    'acc': 'the runtime will abort here',
    'omp': 'OpenMP skips this copy and kernels copy the array implicitly',
}


def update_lines(updates: Iterable[Update], indent: str,
                 comment: bool = True, trace: bool = False,
                 directive: str = 'acc') -> List[str]:
    """Fortran lines for a list of planned updates.  With *trace*, each
    update is announced on ``error_unit`` with the slice's shape."""
    lines: List[str] = []
    for u in updates:
        body: List[str] = []
        inner = indent + '  ' if u.guard else indent
        sp = SPELLINGS[directive]
        if trace and u.kind != 'wait':
            what = 'copy to device' if u.kind == 'device' else 'copy to host'
            body += _trace_write(inner, [
                ('s', TAG), ('s', what), ('s', u.expr),
                ('s', 'shape:'), ('v', 'shape({})'.format(u.expr))])
            base = _strip_subscript(u.expr)
            body.append('{}if (.not. {}) then'.format(inner, sp.present(base)))
            body += _trace_write(inner + '  ', [
                ('s', TAG + ' WARNING:'), ('s', base),
                ('s', 'has no device copy at this {};'.format(what)),
                ('s', 'was it reallocated after ccpp_init?'),
                ('s', _ABSENT_CONSEQUENCE[directive])])
            body.append('{}end if'.format(inner))
        if u.kind == 'wait':
            body += _directive(inner, sp.wait, directive)
        elif u.kind == 'device':
            body += _directive(inner, sp.to_device(u.expr), directive)
        else:
            body += _directive(inner, sp.to_host(u.expr), directive)
        if comment and u.reason:
            lines.append('{}! {}: {}'.format(indent, TAG, u.reason))
        if u.guard:
            lines.append('{}if ({}) then'.format(indent, u.guard))
            lines.extend(body)
            lines.append('{}end if'.format(indent))
        else:
            lines.extend(body)
    return _fit(lines)


def wait_lines(indent: str, directive: str = 'acc') -> List[str]:
    """Wait for all device work (``!$acc wait`` / ``!$omp taskwait``)."""
    return _directive(indent, SPELLINGS[directive].wait, directive)


########################################################################
# Debug checks (--device-debug)
########################################################################

def _decl_type(arr: TrackedArray) -> str:
    t = arr.type_.strip().lower()
    if t in ('real', 'integer', 'complex') and arr.kind:
        return '{}(kind={})'.format(t, arr.kind)
    if t == 'logical':
        return 'logical'
    return t


def _guarded(lines: List[str], cond: str, indent: str) -> List[str]:
    """*lines* inside ``if (cond) then`` (indented one level) when *cond*."""
    if not cond:
        return lines
    return (['{}if ({}) then'.format(indent, cond)]
            + ['  ' + ln for ln in lines]
            + ['{}end if'.format(indent)])


def cxx_pack_lines(packs, indent: str, directive: str, unified: bool
                   ) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Dense device scratch arrays for the device-side array arguments of a
    C++ scheme call.

    Returns (lines before the call, lines after it, argument overrides).
    The before-lines open a ``block``, allocate each scratch array with the
    slice's shape, give it a device copy and copy the slice into it on the
    device; the after-lines copy written arguments back on the device,
    delete the scratch arrays and close the block.  The C++ scheme receives
    a dense, column-major, device-resident array of the slice's shape.
    In unified mode the copies are plain loops and no device copies are
    made.
    """
    sp = SPELLINGS[directive]
    i2 = indent + '  '
    before = ['{}block'.format(indent),
              '{}! {}: dense device copies for C++ scheme array arguments'.format(i2, TAG)]
    after: List[str] = []
    overrides: Dict[str, str] = {}
    body_alloc, body_in, body_out, body_free = [], [], [], []
    max_rank = 0
    for pk in packs:
        tmp = 'ccpp_cxx_{}'.format(pk.local_name)
        parts = _split_top(pk.subscript.strip()[1:-1]) if pk.subscript.strip() \
            else [':'] * pk.array.rank
        extents, src = [], []
        loop_dim = 0
        for d, part in enumerate(parts, start=1):
            if ':' in part:
                loop_dim += 1
                lo, _, hi = part.partition(':')
                lo = lo.strip() or 'lbound({}, {})'.format(pk.base, d)
                hi = hi.strip() or 'ubound({}, {})'.format(pk.base, d)
                extents.append('({}) - ({}) + 1'.format(hi, lo))
                src.append('({}) + ccpp_cxx_j{} - 1'.format(lo, loop_dim))
            else:
                src.append(part)
        rank = len(extents)
        max_rank = max(max_rank, rank)
        # Logical arrays cross as 0/1 bytes (see cxx_schemes).
        before.append('{}{}, allocatable :: {}({})'.format(
            i2, 'integer(c_signed_char)' if pk.logical else _decl_type(pk.array),
            tmp, ','.join([':'] * rank)))
        alloc = ['{}allocate({}({}))'.format(i2, tmp, ', '.join(extents))]
        if not unified:
            alloc += _directive(i2, sp.create(tmp), directive)
        # Optional argument: only when present; an unallocated scratch array
        # makes the optional dummy absent (NULL in C++).
        body_alloc += _guarded(alloc, pk.guard, i2)
        idx = ', '.join('ccpp_cxx_j{}'.format(j) for j in range(1, rank + 1))
        src_ref = '{}({})'.format(pk.base, ', '.join(src))

        def nest(stmt):
            out = []
            if not unified:
                if directive == 'omp':
                    out += _directive(i2, 'target teams loop collapse({})'.format(rank), directive)
                else:
                    out += _directive(i2, 'parallel loop collapse({}) default(present)'.format(rank), directive)
            for j in range(rank, 0, -1):
                out.append('{}{}do ccpp_cxx_j{} = 1, size({}, {})'.format(
                    i2, '  ' * (rank - j), j, tmp, j))
            out.append('{}{}{}'.format(i2, '  ' * rank, stmt))
            for j in range(1, rank + 1):
                out.append('{}{}end do'.format(i2, '  ' * (rank - j)))
            return out

        present = 'allocated({})'.format(tmp) if pk.guard else ''
        if pk.logical:
            copy_in = '{}({}) = merge(1_c_signed_char, 0_c_signed_char, {})'.format(
                tmp, idx, src_ref)
            copy_out = '{} = {}({}) /= 0_c_signed_char'.format(src_ref, tmp, idx)
        else:
            copy_in = '{}({}) = {}'.format(tmp, idx, src_ref)
            copy_out = '{} = {}({})'.format(src_ref, tmp, idx)
        if pk.reads:
            body_in += _guarded(nest(copy_in), present, i2)
        if pk.writes:
            body_out += _guarded(nest(copy_out), present, i2)
        free = []
        if not unified:
            free += _directive(i2, sp.delete(tmp), directive)
        free.append('{}deallocate({})'.format(i2, tmp))
        body_free += _guarded(free, present, i2)
        overrides[pk.local_name] = tmp
    if max_rank:
        before.append('{}integer :: {}'.format(
            i2, ', '.join('ccpp_cxx_j{}'.format(j) for j in range(1, max_rank + 1))))
    before += body_alloc + body_in
    after = body_out + body_free + ['{}end block'.format(indent)]
    return _fit(before), _fit(after), overrides


def check_kinds(plan: SuitePlan, group_name: str) -> List[str]:
    """Kind parameters needed by the coherence and trace checks in *group_name*."""
    kinds: Set[str] = set()
    for (gname, _phase), pp in plan.phases.items():
        if gname != group_name:
            continue
        checks = list(pp.entry_checks) + list(pp.exit_checks)
        for lst in (list(pp.pre_checks.values()) + list(pp.post_checks.values())
                    + list(pp.trace_checks.values())):
            checks += lst
        for packs in pp.cxx_pack.values():
            for pk in packs:
                if pk.array.kind and pk.array.type_.strip().lower() in (
                        'real', 'integer', 'complex'):
                    kinds.add(pk.array.kind)
        for c in checks:
            if c.kind in ('coherent', 'trace') and c.array.kind and \
                    c.array.type_.strip().lower() in ('real', 'integer', 'complex'):
                kinds.add(c.array.kind)
    return sorted(kinds)


def _wrap(indent: str, head: str, items: List[str], tail: str) -> List[str]:
    """``head item1 item2 ... tail`` on continuation lines when long."""
    one = '{}{}{}{}'.format(indent, head, ' '.join(items), tail)
    if len(one) <= _MAX_LINE:
        return [one]
    out = ['{}{}&'.format(indent, head)]
    for i, it in enumerate(items):
        end = ' &' if i < len(items) - 1 else tail
        out.append('{}    {}{}'.format(indent, it, end))
    return out


def _slice_elements(expr: str, base: str, rank: int) -> Tuple[str, str]:
    """First and last element references of the slice *expr* of *base*.

    ``x(lb:ub, 1:n)`` -> (``x(lb, 1)``, ``x(ub, n)``); a ``:`` dimension
    uses the array's own bounds; a whole-array *expr* uses them for every
    dimension.
    """
    sub = expr[len(base):].strip()
    parts = _split_top(sub[1:-1]) if sub.startswith('(') else [':'] * rank
    first, last = [], []
    for d, part in enumerate(parts, start=1):
        if part == ':':
            first.append('lbound({}, {})'.format(base, d))
            last.append('ubound({}, {})'.format(base, d))
        elif ':' in part:
            lo, _, hi = part.partition(':')
            first.append(lo.strip() or 'lbound({}, {})'.format(base, d))
            last.append(hi.strip() or 'ubound({}, {})'.format(base, d))
        else:
            first.append(part)
            last.append(part)
    return ('{}({})'.format(base, ', '.join(first)),
            '{}({})'.format(base, ', '.join(last)))


def _one_element_section(element: str, base: str) -> str:
    """``x(i, j)`` -> ``x(i:i, j:j)``: a contiguous one-element array
    section, which presence queries accept (they take arrays)."""
    inner = element[len(base) + 1:-1]
    return '{}({})'.format(base, ', '.join('{0}:{0}'.format(p)
                                         for p in _split_top(inner)))


def _split_top(text: str) -> List[str]:
    parts, depth, cur = [], 0, ''
    for ch in text:
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


def _trace_check_lines(c: Check, inner: str, unified: bool,
                       directive: str) -> List[str]:
    """One --device-trace record for an array a device-side call uses.

    Prints where the plan expects valid data, whether the whole array and
    the slice's first and last elements are present on the device, the
    slice shape, the array bounds, the values of the variables in the slice
    bounds, and the slice's first element read on the host and on the
    device.  Warns when the slice is not fully present: the next device
    access to it would fail.  The device read runs only when the data is
    present, so tracing never triggers an implicit copy.
    """
    arr = c.array
    base = arr.base_expr
    sp = SPELLINGS[directive]
    first, last = _slice_elements(c.expr, base, arr.rank)
    names = _unique(re.findall(r'[A-Za-z_][A-Za-z0-9_%]*', c.expr[len(base):]))
    items = [('s', TAG), ('s', c.where), ('s', c.expr), ('s', 'plan:'),
             ('s', c.plan_state)]
    i2 = inner + '  '
    lines = ['{}block'.format(inner)]
    if not unified:
        lines.append('{}logical :: ccpp_gpu_p, ccpp_gpu_p1, ccpp_gpu_p2'.format(i2))
    lines += ['{}{} :: ccpp_gpu_hv, ccpp_gpu_dv'.format(i2, _decl_type(arr)),
              '{}ccpp_gpu_hv = {}'.format(i2, first),
              '{}ccpp_gpu_dv = ccpp_gpu_hv'.format(i2)]
    if not unified:
        lines += ['{}ccpp_gpu_p = {}'.format(i2, sp.present(base)),
                  '{}ccpp_gpu_p1 = {}'.format(i2, sp.present(_one_element_section(first, base))),
                  '{}ccpp_gpu_p2 = {}'.format(i2, sp.present(_one_element_section(last, base))),
                  '{}if (ccpp_gpu_p1) then'.format(i2)]
        if directive == 'omp':
            lines += _directive(i2 + '  ', 'target map(from: ccpp_gpu_dv)', directive)
            lines.append('{}  ccpp_gpu_dv = {}'.format(i2, first))
            lines += _directive(i2 + '  ', 'end target', directive)
        else:
            lines += _directive(i2 + '  ', 'serial present({}) copyout(ccpp_gpu_dv)'.format(base), directive)
            lines.append('{}  ccpp_gpu_dv = {}'.format(i2, first))
            lines += _directive(i2 + '  ', 'end serial', directive)
        lines.append('{}end if'.format(i2))
        items += [('s', 'present (array, first, last):'), ('v', 'ccpp_gpu_p'),
                  ('v', 'ccpp_gpu_p1'), ('v', 'ccpp_gpu_p2')]
    items += [('s', 'shape:'), ('v', 'shape({})'.format(c.expr)),
              ('s', 'array lbound:'), ('v', 'lbound({})'.format(base)),
              ('s', 'array ubound:'), ('v', 'ubound({})'.format(base))]
    for name in names:
        items += [('s', '{}='.format(name)), ('v', name)]
    if unified:
        items += [('s', 'first element:'), ('v', 'ccpp_gpu_hv')]
    else:
        items += [('s', 'first element host / device:'), ('v', 'ccpp_gpu_hv'),
                  ('v', 'ccpp_gpu_dv')]
    lines += _trace_write(i2, items)
    if not unified:
        lines.append('{}if (.not. (ccpp_gpu_p .and. ccpp_gpu_p1 .and. ccpp_gpu_p2)) then'.format(i2))
        consequence = ('the next device access to it will fail'
                       if directive == 'acc' else
                       'OpenMP will copy it implicitly at every kernel launch')
        lines += _trace_write(i2 + '  ', [
            ('s', TAG + ' WARNING:'), ('s', c.expr),
            ('s', 'is not fully present on the device ' + c.where + ';'),
            ('s', consequence)])
        lines.append('{}end if'.format(i2))
    lines.append('{}end block'.format(inner))
    return lines


def check_lines(checks: Iterable[Check], indent: str, errmsg: str,
                errflg: str, label: str, unified: bool = False,
                directive: str = 'acc') -> List[str]:
    """Runtime presence and coherence checks.

    *Presence*: the array must have a device copy before a device-side
    call.  *Coherence*: wherever the plan says host and device copies are
    both valid, copy the device slice back into a scratch array and
    compare it with the host slice.
    """
    lines: List[str] = []
    for c in checks:
        inner = indent + '  ' if c.guard else indent
        body: List[str] = []
        if c.kind == 'trace':
            body += _trace_check_lines(c, inner, unified, directive)
        elif c.kind == 'scalars':
            items = [('s', TAG), ('s', c.where), ('s', 'scalar arguments:')]
            for name, expr in c.scalars:
                items += [('s', '{}='.format(name)), ('v', expr)]
            body += _trace_write(inner, items)
        elif c.kind == 'present':
            body += [
                '{}if (.not. {}) then'.format(inner, SPELLINGS[directive].present(c.expr)),
                "{}  {} = '{}: {} has no device copy' // &".format(
                    inner, errmsg, label, c.expr),
                "{}      ' before {}'".format(inner, c.where),
                '{}  {} = 1'.format(inner, errflg),
                '{}  return'.format(inner),
                '{}end if'.format(inner),
            ]
        else:
            arr = c.array
            rank = arr.rank
            dims = '({})'.format(','.join([':'] * rank)) if rank else ''
            t = arr.type_.strip().lower()
            e = c.expr
            if t == 'logical':
                differ = ['any({} .neqv.'.format(e), 'ccpp_gpu_chk)']
            elif t in ('real', 'complex'):
                # NaN-safe: two NaNs count as equal.
                differ = ['any({} /= ccpp_gpu_chk'.format(e),
                          '.and. ({e} == {e}'.format(e=e),
                          '.or. ccpp_gpu_chk == ccpp_gpu_chk))']
            else:
                differ = ['any({} /='.format(e), 'ccpp_gpu_chk)']
            msg = ["'{}:' //".format(label),
                   "' host and device copies of {} differ' //".format(e),
                   "' ({})'".format(c.where)]
            body += ['{}block'.format(inner),
                     '{}  {}, allocatable :: ccpp_gpu_chk{}'.format(
                         inner, _decl_type(arr), dims),
                     '{}  allocate(ccpp_gpu_chk, source={})'.format(inner, e)]
            body += _directive(inner + '  ', SPELLINGS[directive].to_host(e), directive)
            body += _wrap(inner + '  ', 'if (', differ, ') then')
            body += _wrap(inner + '    ', "write({}, '(a)') ".format(errmsg),
                          msg, '')
            body += ['{}    {} = 1'.format(inner, errflg),
                     '{}  end if'.format(inner),
                     '{}  {} = ccpp_gpu_chk'.format(inner, e),
                     '{}end block'.format(inner),
                     '{}if ({} /= 0) return'.format(inner, errflg)]
        if c.guard:
            lines.append('{}if ({}) then'.format(indent, c.guard))
            lines.extend(body)
            lines.append('{}end if'.format(indent))
        else:
            lines.extend(body)
    return _fit(lines)


########################################################################
# Suite cap: device copies and sync routines
########################################################################

def _unique(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _uses_for(arrays: Iterable[TrackedArray]) -> Dict[str, Set[str]]:
    uses: Dict[str, Set[str]] = {}
    for a in arrays:
        if a.module_name:
            uses.setdefault(a.module_name, set()).add(a.root_symbol)
        for mod, syms in a.active_uses.items():
            uses.setdefault(mod, set()).update(syms)
    return uses


def _whole_array(indent: str, a: TrackedArray, text: str,
                 directive: str = 'acc') -> List[str]:
    """A whole-array directive, wrapped in the host ``active`` condition
    when the array has one (it may not be allocated otherwise)."""
    if not a.active_local:
        return _directive(indent, text, directive)
    return (['{}if ({}) then'.format(indent, a.active_local)]
            + _directive(indent + '  ', text, directive)
            + ['{}end if'.format(indent)])


def suite_init_uses(plan: SuitePlan) -> Dict[str, Set[str]]:
    uses = _uses_for(plan.created_host_arrays() + plan.created_suite_arrays()
                     + plan.scheme_allocated_arrays())
    if plan.options.trace and plan.options.directive == 'acc' \
            and not plan.options.unified:
        uses.setdefault('openacc', set()).add('acc_is_present')
    return uses


def suite_needs_omp_present_helper(plan: 'SuitePlan') -> bool:
    """True when the suite cap's trace lines query OpenMP presence."""
    return (plan is not None and plan.options.directive == 'omp'
            and plan.options.trace and not plan.options.unified
            and bool(plan.created_host_arrays() + plan.created_suite_arrays()))


def suite_init_lines(plan: SuitePlan, indent: str) -> List[str]:
    """Create one device copy per tracked array (``<suite>_init``)."""
    hosts = plan.created_host_arrays()
    suites = plan.created_suite_arrays()
    if not hosts and not suites:
        return []
    d = plan.options.directive
    sp = SPELLINGS[d]
    lines = ['', '{}! {}: create device copies (kept until {}_final)'.format(
        indent, TAG, plan.suite_name)]
    parents = _unique(p for a in hosts for p in a.parents)
    for p in parents:
        lines += _directive(indent, sp.create(p), d)
    for a in hosts:
        lines += _whole_array(indent, a, sp.create(a.base_expr), d)
    if suites:
        lines += _directive(indent, sp.create('ccpp_suite_data'), d)
        for a in suites:
            lines += _directive(indent, sp.create(a.base_expr), d)
    if plan.options.trace:
        for a in hosts + suites:
            items = [('s', TAG), ('s', 'created in {}_init:'.format(plan.suite_name)),
                     ('s', a.base_expr), ('s', 'present:'), ('v', sp.present(a.base_expr)),
                     ('s', 'lbound:'), ('v', 'lbound({})'.format(a.base_expr)),
                     ('s', 'ubound:'), ('v', 'ubound({})'.format(a.base_expr))]
            if a.active_local:
                lines.append('{}if ({}) then'.format(indent, a.active_local))
                lines += _trace_write(indent + '  ', items)
                lines.append('{}end if'.format(indent))
            else:
                lines += _trace_write(indent, items)
    return lines


def suite_final_lines(plan: SuitePlan, indent: str) -> List[str]:
    """Delete the device copies made by :func:`suite_init_lines`."""
    hosts = plan.created_host_arrays()
    suites = plan.created_suite_arrays()
    allocs = plan.scheme_allocated_arrays()
    if not hosts and not suites and not allocs:
        return []
    d = plan.options.directive
    sp = SPELLINGS[d]
    lines = ['{}! {}: delete device copies'.format(indent, TAG)]
    for a in allocs:
        # Created by a C++ scheme's allocation callback, if allocated.
        lines.append('{}if (allocated({})) then'.format(indent, a.base_expr))
        lines += _directive(indent + '  ', sp.delete(a.base_expr), d)
        lines.append('{}end if'.format(indent))
    for a in suites:
        lines += _directive(indent, sp.delete(a.base_expr), d)
    if suites:
        lines += _directive(indent, sp.delete('ccpp_suite_data'), d)
    for a in hosts:
        lines += _whole_array(indent, a, sp.delete(a.base_expr), d)
    for p in reversed(_unique(p for a in hosts for p in a.parents)):
        lines += _directive(indent, sp.delete(p), d)
    return lines


SYNC_DIRECTIONS = ('device', 'host')


def suite_sync_sub_name(suite_name: str, direction: str) -> str:
    return '{}_device_update_{}'.format(suite_name, direction)


def suite_sync_subroutines(plan: SuitePlan, errmsg: str, errflg: str,
                           indent: str, indent2: str) -> List[str]:
    """``<suite>_device_update_device`` / ``_host``.  The host calls them
    (through the host cap) after filling data the GPU needs, and before
    reading data that lives on the device.  ``_device`` copies the arrays
    whose handoff state includes the device (``host_access = none`` or
    ``read``); ``_host`` copies only the ``host_access = none`` arrays.
    Copying anything else would overwrite newer host values."""
    lines: List[str] = []
    for direction in SYNC_DIRECTIONS:
        arrays = plan.synced_host_arrays(direction)
        uses = _uses_for(arrays)
        name = suite_sync_sub_name(plan.suite_name, direction)
        lines += ['', '{}subroutine {}({}, {})'.format(indent, name, errmsg, errflg)]
        for mod in sorted(uses):
            lines.append('{}use {}, only: {}'.format(
                indent2, mod, ', '.join(sorted(uses[mod]))))
        lines += [
            '',
            '{}character(len=*), intent(out) :: {}'.format(indent2, errmsg),
            '{}integer, intent(out) :: {}'.format(indent2, errflg),
            '',
            "{}{} = ''".format(indent2, errmsg),
            '{}{} = 0'.format(indent2, errflg),
        ]
        d = plan.options.directive
        sp = SPELLINGS[d]
        if plan.options.unified:
            if direction == 'host':
                lines += _directive(indent2, sp.wait, d)
            else:
                lines.append('{}! {}: unified memory, nothing to copy'.format(
                    indent2, TAG))
        else:
            fmt = sp.to_device if direction == 'device' else sp.to_host
            for a in arrays:
                lines += _whole_array(indent2, a, fmt(a.base_expr), d)
        lines.append('{}end subroutine {}'.format(indent, name))
    return lines


########################################################################
# Host cap: dispatching sync entry points
########################################################################

def host_sync_sub_name(direction: str) -> str:
    return 'ccpp_physics_device_update_{}'.format(direction)


def host_sync_subroutines(planned_suites: List[str], all_suites: List[str],
                          suite_name_local: str, errmsg: str, errflg: str,
                          indent: str) -> List[str]:
    """``ccpp_physics_device_update_device`` / ``_host`` in the host cap."""
    i2, i3 = indent * 2, indent * 3
    lines: List[str] = []
    for direction in SYNC_DIRECTIONS:
        name = host_sync_sub_name(direction)
        lines += [
            '',
            '{}subroutine {}({}, {}, {})'.format(indent, name, suite_name_local,
                                                 errmsg, errflg),
            '',
            '{}! {}: copy to the {} the CCPP-managed arrays whose valid copy'.format(
                i2, TAG, direction),
            '{}! must be there while the host has control (see host_access).'.format(i2),
            '{}character(len=*), intent(in) :: {}'.format(i2, suite_name_local),
            '{}character(len=*), intent(out) :: {}'.format(i2, errmsg),
            '{}integer, intent(out) :: {}'.format(i2, errflg),
            '',
            "{}{} = ''".format(i2, errmsg),
            '{}{} = 0'.format(i2, errflg),
            '{}select case(trim({}))'.format(i2, suite_name_local),
        ]
        for s in all_suites:
            lines.append("{}case('{}')".format(i2, s))
            if s in planned_suites:
                lines.append('{}call {}({}, {})'.format(
                    i3, suite_sync_sub_name(s, direction), errmsg, errflg))
            else:
                lines.append('{}continue'.format(i3))
        lines += [
            '{}case default'.format(i2),
            '{}{} = 1'.format(i3, errflg),
            "{}{} = '{}: unknown suite: ' // trim({})".format(
                i3, errmsg, name, suite_name_local),
            '{}end select'.format(i2),
            '',
            '{}end subroutine {}'.format(indent, name),
        ]
    return lines
