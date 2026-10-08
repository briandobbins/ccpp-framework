#!/usr/bin/env python3
"""Generated timers around scheme calls and data movement (``--timer-style``).

With ``--timer-style`` every scheme call in a group cap is timed, and so is
the generated host<->device data movement around it, in the host model's own
timing library:

* ``<scheme>_<phase>``      -- the call itself (compute).  For a call with a
  device-side argument the timer stops only after the device has finished
  (``!$acc wait`` / ``!$omp taskwait``), so asynchronous kernels are counted.
* ``<scheme>_<phase>_h2d``  -- copies to the device made for this call.
* ``<scheme>_<phase>_d2h``  -- copies back to the host made for this call
  (after a GPU call, or before a CPU call that reads device results).
* ``<group>_<phase>_loop_h2d`` / ``_loop_d2h`` -- copies before a subcycle.
* ``<group>_<phase>_exit_h2d`` / ``_exit_d2h`` -- copies at group exit.

Unit and kind transforms and ``--device-debug`` checks are outside the timers.
Without ``--directive`` only the compute timers exist, so a CPU build and a GPU
build of the same suite report the same compute timer names.

Styles:

``cesm``
    GPTL through CIME's ``perf_mod``: ``t_startf(name)`` / ``t_stopf(name)``.
    Thread-safe.  Names up to 63 characters.
``fms``
    FMS clocks (UFS): ``mpp_clock_id(name, grain=CLOCK_ROUTINE)`` once, then
    ``mpp_clock_begin(id)`` / ``mpp_clock_end(id)``.  FMS clocks are not
    thread-safe, so only OpenMP thread 0 times (UFS runs physics blocks in
    threads; thread 0's time is representative, not a sum).  Names up to 32
    characters.  Clocks finer than FMS's ``clock_grain`` namelist setting are
    not reported; set ``clock_grain = 'ROUTINE'`` to see them.

Over-long names are shortened deterministically: the suffix (``_h2d``,
``_d2h``) is kept, the start is truncated, and a 4-digit hash keeps distinct
names distinct.
"""

import hashlib
from typing import Dict, List

VALID_TIMER_STYLES = ('cesm', 'fms')

_NAME_LIMIT = {'cesm': 63, 'fms': 32}
_SUFFIXES = ('_h2d', '_d2h')

_INDENT = '  '


def _shorten(name: str, limit: int) -> str:
    if len(name) <= limit:
        return name
    suffix = next((s for s in _SUFFIXES if name.endswith(s)), '')
    stem = name[:len(name) - len(suffix)] if suffix else name
    tag = hashlib.md5(name.encode()).hexdigest()[:4]
    keep = limit - len(suffix) - len(tag) - 1
    return '{}~{}{}'.format(stem[:keep], tag, suffix)


class TimerRegistry:
    """The timers one group cap module uses, in order of first use."""

    def __init__(self, style: str):
        if style not in VALID_TIMER_STYLES:
            raise ValueError('unknown timer style {!r}'.format(style))
        self.style = style
        self.names: List[str] = []
        self._index: Dict[str, int] = {}

    def index(self, name: str) -> int:
        name = _shorten(name, _NAME_LIMIT[self.style])
        if name not in self._index:
            self.names.append(name)
            self._index[name] = len(self.names)
        return self._index[name]

    def start(self, name: str, indent: str) -> str:
        return '{}call ccpp_timer_start({})  ! {}'.format(
            indent, self.index(name), _shorten(name, _NAME_LIMIT[self.style]))

    def stop(self, name: str, indent: str) -> str:
        return '{}call ccpp_timer_stop({})'.format(indent, self.index(name))

    # -- module pieces ---------------------------------------------------
    def use_lines(self) -> List[str]:
        if not self.names:
            return []
        if self.style == 'cesm':
            return ['{}use perf_mod, only: t_startf, t_stopf'.format(_INDENT)]
        return ['{}use mpp_mod, only: mpp_clock_id, mpp_clock_begin, '
                'mpp_clock_end, CLOCK_ROUTINE'.format(_INDENT)]

    def declaration_lines(self) -> List[str]:
        if not self.names:
            return []
        n = len(self.names)
        width = max(len(x) for x in self.names)
        out = ['',
               '{}! Timers (--timer-style {})'.format(_INDENT, self.style),
               '{}integer, parameter :: ccpp_ntimers = {}'.format(_INDENT, n),
               '{0}character(len={1}), parameter :: ccpp_timer_names(ccpp_ntimers) = '
               '[character(len={1}) :: &'.format(_INDENT, width)]
        for i, name in enumerate(self.names):
            sep = ', &' if i < n - 1 else ' ]'
            out.append("{}  '{}'{}".format(_INDENT, name, sep))
        if self.style == 'fms':
            out.append('{}integer, save :: ccpp_timer_ids(ccpp_ntimers) = 0'
                       .format(_INDENT))
            out.append('{}logical, save :: ccpp_timer_known(ccpp_ntimers) = .false.'
                       .format(_INDENT))
        return out

    def helper_lines(self) -> List[str]:
        if not self.names:
            return []
        i1, i2 = _INDENT, _INDENT * 2
        if self.style == 'cesm':
            return [
                '{}subroutine ccpp_timer_start(i)'.format(i1),
                '{}integer, intent(in) :: i'.format(i2),
                '{}call t_startf(trim(ccpp_timer_names(i)))'.format(i2),
                '{}end subroutine ccpp_timer_start'.format(i1),
                '',
                '{}subroutine ccpp_timer_stop(i)'.format(i1),
                '{}integer, intent(in) :: i'.format(i2),
                '{}call t_stopf(trim(ccpp_timer_names(i)))'.format(i2),
                '{}end subroutine ccpp_timer_stop'.format(i1),
                '',
            ]
        return [
            '{}subroutine ccpp_timer_start(i)'.format(i1),
            '!$  use omp_lib, only: omp_get_thread_num',
            '{}integer, intent(in) :: i'.format(i2),
            '!$  if (omp_get_thread_num() /= 0) return',
            '{}if (.not. ccpp_timer_known(i)) then'.format(i2),
            '{}  ccpp_timer_ids(i) = mpp_clock_id(trim(ccpp_timer_names(i)), '
            'grain=CLOCK_ROUTINE)'.format(i2),
            '{}  ccpp_timer_known(i) = .true.'.format(i2),
            '{}end if'.format(i2),
            '{}call mpp_clock_begin(ccpp_timer_ids(i))'.format(i2),
            '{}end subroutine ccpp_timer_start'.format(i1),
            '',
            '{}subroutine ccpp_timer_stop(i)'.format(i1),
            '!$  use omp_lib, only: omp_get_thread_num',
            '{}integer, intent(in) :: i'.format(i2),
            '!$  if (omp_get_thread_num() /= 0) return',
            '{}call mpp_clock_end(ccpp_timer_ids(i))'.format(i2),
            '{}end subroutine ccpp_timer_stop'.format(i1),
            '',
        ]


def split_by_direction(updates):
    """Group planned updates into maximal runs of one direction, keeping
    their order: yields ``('h2d' | 'd2h', [updates])``.  A unified-memory
    ``wait`` counts as ``d2h`` (it is where the host waits for device
    results)."""
    run, kind = [], None
    for u in updates:
        k = 'h2d' if u.kind == 'device' else 'd2h'
        if kind is not None and k != kind:
            yield kind, run
            run = []
        kind = k
        run.append(u)
    if run:
        yield kind, run
