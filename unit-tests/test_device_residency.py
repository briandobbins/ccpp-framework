"""Unit tests for GPU data movement (doc/gpu_data_movement_plan.md).

Covers the metadata attributes (``memory_space``, ``host_access``), the
residency planner (``generator/device_residency.py``) and, via the
generated Fortran, the cap emitters.
"""

import os
import shutil
import tempfile
import textwrap
import unittest

from ccpp_capgen import capgen
from metadata.metadata_table import parse_metadata_file
from metadata.parse_tools import CCPPError

_CONTROL = """\
[ccpp-table-properties]
  name = main
  type = control
[ccpp-arg-table]
  name = main
  type = control
[ suite_name ]
  standard_name = suite_name
  units = none
  dimensions = ()
  type = character
  kind = len=256
[ group_name ]
  standard_name = group_name
  units = none
  dimensions = ()
  type = character
  kind = len=256
[ lb ]
  standard_name = horizontal_loop_begin
  units = index
  dimensions = ()
  type = integer
[ ub ]
  standard_name = horizontal_loop_end
  units = index
  dimensions = ()
  type = integer
[ nphys_threads ]
  standard_name = number_of_physics_threads
  units = count
  dimensions = ()
  type = integer
[ errmsg ]
  standard_name = ccpp_error_message
  units = none
  dimensions = ()
  type = character
  kind = len=512
[ errflg ]
  standard_name = ccpp_error_code
  units = 1
  dimensions = ()
  type = integer
"""

# name -> (standard_name, units, dims, type, kind)
_VARS = {
    'ncol': ('horizontal_dimension', 'count', '()', 'integer', ''),
    'nlev': ('vertical_layer_dimension', 'count', '()', 'integer', ''),
    'flag': ('flag_for_extra_forcing', 'flag', '()', 'logical', ''),
    'nsub': ('number_of_subcycles', 'count', '()', 'integer', ''),
    'a': ('field_a', 'K', '(horizontal_dimension, vertical_layer_dimension)', 'real', 'kind_phys'),
    'b': ('field_b', 'K', '(horizontal_dimension, vertical_layer_dimension)', 'real', 'kind_phys'),
    'c': ('field_c', 'K', '(horizontal_dimension, vertical_layer_dimension)', 'real', 'kind_phys'),
    'w': ('scratch_field', 'K', '(horizontal_dimension, vertical_layer_dimension)', 'real', 'kind_phys'),
    'opt': ('optional_field', 'K', '(horizontal_dimension, vertical_layer_dimension)', 'real', 'kind_phys'),
    'mask': ('cloud_mask', 'flag', '(horizontal_dimension, vertical_layer_dimension)', 'logical', ''),
    'nbin': ('number_of_bins', 'count', '()', 'integer', ''),
    'binned': ('binned_field', 'K', '(horizontal_dimension, number_of_bins)', 'real', 'kind_phys'),
    's': ('scalar_field', 'K', '()', 'real', 'kind_phys'),
    'errmsg': ('ccpp_error_message', 'none', '()', 'character', 'len=*'),
    'errflg': ('ccpp_error_code', '1', '()', 'integer', ''),
}


def _var(name, extra=(), intent=None, units=None):
    std, u, dims, typ, kind = _VARS[name]
    lines = ['[{}]'.format(name), '  standard_name = {}'.format(std),
             '  units = {}'.format(units or u), '  dimensions = {}'.format(dims),
             '  type = {}'.format(typ)]
    if kind:
        lines.append('  kind = {}'.format(kind))
    if intent:
        lines.append('  intent = {}'.format(intent))
    lines.extend('  {} = {}'.format(k, v) for k, v in extra)
    return '\n'.join(lines)


def _host_meta(host_attrs=None, with_flag=False, with_nsub=False):
    """Host table with ncol/nlev and arrays a, b, c, opt (+ flag/nsub)."""
    host_attrs = host_attrs or {}
    out = ['[ccpp-table-properties]', '  name = hmod', '  type = host',
           '[ccpp-arg-table]', '  name = hmod', '  type = host']
    names = ['ncol', 'nlev', 'a', 'b', 'c']
    if with_flag:
        names += ['flag', 'opt']
    if with_nsub:
        names.append('nsub')
    for n in names:
        extra = list(host_attrs.get(n, {}).items())
        if n == 'opt':
            extra.append(('active', '(flag_for_extra_forcing)'))
        out.append(_var(n, extra))
    return '\n'.join(out) + '\n'


def _scheme_meta(name, phases, header_space=None):
    """phases: {phase: [(var, intent, {extra}), ...]}"""
    out = ['[ccpp-table-properties]', '  name = {}'.format(name),
           '  type = scheme', '']
    for phase, args in phases.items():
        out += ['[ccpp-arg-table]', '  name = {}_{}'.format(name, phase),
                '  type = scheme']
        if header_space:
            out.append('  memory_space = {}'.format(header_space))
        for spec in args:
            var, intent = spec[0], spec[1]
            extra = list(spec[2].items()) if len(spec) > 2 else []
            units = dict(extra).pop('units', None) if extra else None
            extra = [(k, v) for k, v in extra if k != 'units']
            out.append(_var(var, extra, intent, units))
        out += [_var('errmsg', (), 'out'), _var('errflg', (), 'out'), '']
    return '\n'.join(out)


def _suite_xml(groups):
    """groups: [(group_name, [item, ...])]; item is a scheme name or
    ('subcycle', loop, [scheme, ...])."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<suite name="tsuite" version="2.0">']
    for gname, items in groups:
        lines.append('  <group name="{}">'.format(gname))
        for it in items:
            if isinstance(it, tuple):
                lines.append('    <subcycle loop="{}">'.format(it[1]))
                lines += ['      <scheme>{}</scheme>'.format(s) for s in it[2]]
                lines.append('    </subcycle>')
            else:
                lines.append('    <scheme>{}</scheme>'.format(it))
        lines.append('  </group>')
    lines.append('</suite>')
    return '\n'.join(lines) + '\n'


class _Fixture:
    """Writes metadata to a temp dir and runs capgen on it."""

    def __init__(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, 'out')
        self.schemes = []

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, fname, text):
        path = os.path.join(self.dir, fname)
        with open(path, 'w') as fh:
            fh.write(text)
        return path

    def scheme(self, name, phases, header_space=None):
        self.schemes.append(self.write(name + '.meta',
                                       _scheme_meta(name, phases, header_space)))
        self.write(name + '.F90', '! placeholder\n')

    def run(self, groups, host_attrs=None, with_flag=False, with_nsub=False,
            directive='acc', memory_model='discrete', debug=False,
            trace=False, sync_debug=False, timer_style=None):
        host = self.write('hmod.meta', _host_meta(host_attrs, with_flag, with_nsub))
        ctrl = self.write('main.meta', _CONTROL)
        suite = self.write('suite.xml', _suite_xml(groups))
        _, results = capgen('thost', [host, ctrl], self.schemes, [suite],
                            self.out, {}, return_state=True,
                            directive=directive, memory_model=memory_model,
                            device_debug=debug, device_trace=trace,
                            device_sync_debug=sync_debug,
                            timer_style=timer_style)
        self.res = results[0]
        return self.res.device_plan

    def text(self, fname):
        with open(os.path.join(self.out, fname)) as fh:
            return fh.read()


def _gpu(*vars_, phase='run'):
    return {phase: [(v, i) for v, i in vars_]}


def _ups(pplan, call_index_by_scheme, which='pre'):
    """Updates attached to the call of *scheme* as (kind, expr) pairs."""
    table = getattr(pplan, which)
    return [(u.kind, u.expr) for u in table.get(call_index_by_scheme, [])]


def _calls(res, group, phase='run'):
    from generator.suite_resolver import iter_phase_calls
    g = next(g for g in res.groups if g.group_name == group)
    return {c.scheme_name: id(c) for c in iter_phase_calls(g.phase_calls[phase])}


A = 'a(lb:ub, 1:nlev)'
B = 'b(lb:ub, 1:nlev)'
C = 'c(lb:ub, 1:nlev)'


class _Base(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture()

    def tearDown(self):
        self.fx.cleanup()


########################################################################
# Metadata
########################################################################

class TestMetadataAttributes(unittest.TestCase):

    def _parse(self, text):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, 'x.meta')
            with open(p, 'w') as fh:
                fh.write(textwrap.dedent(text))
            return parse_metadata_file(p)
        finally:
            shutil.rmtree(d)

    def test_scheme_arg_memory_space(self):
        t = self._parse(_scheme_meta('s', {'run': [('a', 'in', {'memory_space': 'Device'})]}))
        sec = t[0].sections()[0]
        self.assertEqual(sec.variables[0].memory_space, 'device')
        self.assertEqual(sec.effective_memory_space(sec.variables[0]), 'device')

    def test_header_default_and_override(self):
        t = self._parse(_scheme_meta('s', {'run': [
            ('a', 'in'), ('b', 'in', {'memory_space': 'host'})]}, header_space='device'))
        sec = t[0].sections()[0]
        self.assertEqual(sec.memory_space, 'device')
        eff = {v.local_name: sec.effective_memory_space(v) for v in sec.variables}
        self.assertEqual(eff['a'], 'device')
        self.assertEqual(eff['b'], 'host')
        self.assertEqual(eff['errflg'], 'device')

    def test_invalid_memory_space(self):
        with self.assertRaisesRegex(CCPPError, 'memory_space'):
            self._parse(_scheme_meta('s', {'run': [('a', 'in', {'memory_space': 'gpu'})]}))

    def test_invalid_header_memory_space(self):
        with self.assertRaisesRegex(CCPPError, 'memory_space'):
            self._parse(_scheme_meta('s', {'run': [('a', 'in')]}, header_space='fast'))

    def test_host_access_values(self):
        t = self._parse(_host_meta({'a': {'host_access': 'none'},
                                    'b': {'host_access': 'read'}}))
        by = {v.local_name: v for v in t[0].sections()[0].variables}
        self.assertEqual(by['a'].host_access, 'none')
        self.assertEqual(by['b'].host_access, 'read')
        self.assertEqual(by['c'].host_access, '')

    def test_invalid_host_access(self):
        with self.assertRaisesRegex(CCPPError, 'host_access'):
            self._parse(_host_meta({'a': {'host_access': 'sometimes'}}))

    def test_host_access_rejected_on_scheme(self):
        with self.assertRaisesRegex(CCPPError, 'host_access'):
            self._parse(_scheme_meta('s', {'run': [('a', 'in', {'host_access': 'none'})]}))

    def test_memory_space_rejected_on_control(self):
        bad = _CONTROL.replace('  kind = len=256\n', '  kind = len=256\n  memory_space = device\n', 1)
        with self.assertRaisesRegex(CCPPError, 'memory_space'):
            self._parse(bad)

    def test_header_memory_space_rejected_on_host_table(self):
        bad = _host_meta().replace('  type = host\n[ncol]', '  type = host\n  memory_space = device\n[ncol]', 1)
        with self.assertRaisesRegex(CCPPError, 'memory_space'):
            self._parse(bad)


class TestHostDictInheritance(_Base):

    def test_ddt_member_inherits_host_access(self):
        from metadata.variable_resolver import build_flat_host_dict
        d = self.fx.dir
        ddt = """\
[ccpp-table-properties]
  name = st_t
  type = ddt
[ccpp-arg-table]
  name = st_t
  type = ddt
{a}
{b}
""".format(a=_var('a'), b=_var('b', [('host_access', 'read')]))
        host = """\
[ccpp-table-properties]
  name = hmod
  type = host
[ccpp-arg-table]
  name = hmod
  type = host
[st]
  standard_name = state_container
  units = DDT
  dimensions = ()
  type = st_t
  host_access = none
"""
        tables = (parse_metadata_file(self.fx.write('ddt.meta', ddt))
                  + parse_metadata_file(self.fx.write('h.meta', host)))
        hd = build_flat_host_dict([t for t in tables if t.table_type == 'host'], [],
                                  [t for t in tables if t.table_type == 'ddt'])
        self.assertEqual(hd['field_a'].host_access, 'none')   # inherited
        self.assertEqual(hd['field_b'].host_access, 'read')   # own value wins
        self.assertEqual(hd['field_a'].access_path, 'st%a')


########################################################################
# Planner
########################################################################

class TestFlagOff(_Base):

    def test_no_plan_without_directive(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g'])], directive=None)
        self.assertIsNone(plan)
        self.assertNotIn('$acc', self.fx.text('ccpp_tsuite_g1_cap.F90'))


class TestStraightLine(_Base):

    def test_gpu_read_write_of_read_write_var(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g'])])
        pp = plan.phase_plan('g1', 'run')
        call = _calls(self.fx.res, 'g1')['g']
        self.assertEqual(_ups(pp, call), [('device', A)])
        self.assertEqual([(u.kind, u.expr) for u in pp.exit], [('self', A)])
        self.assertEqual(plan.arrays['a'].owner, 'host')
        self.assertEqual(plan.arrays['a'].handoff, frozenset({'H'}))

    def test_gpu_write_only_needs_no_copy_in(self):
        self.fx.scheme('g', _gpu(('a', 'out')), header_space='device')
        plan = self.fx.run([('g1', ['g'])])
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual(_ups(pp, _calls(self.fx.res, 'g1')['g']), [])
        self.assertEqual([(u.kind, u.expr) for u in pp.exit], [('self', A)])

    def test_gpu_to_gpu_stays_on_device(self):
        self.fx.scheme('g1s', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('g2s', _gpu(('a', 'in'), ('b', 'out')), header_space='device')
        plan = self.fx.run([('g1', ['g1s', 'g2s'])])
        pp = plan.phase_plan('g1', 'run')
        calls = _calls(self.fx.res, 'g1')
        self.assertEqual(_ups(pp, calls['g1s']), [('device', A)])
        self.assertEqual(_ups(pp, calls['g2s']), [])
        self.assertEqual(sorted((u.kind, u.expr) for u in pp.exit),
                         [('self', A), ('self', B)])

    def test_cpu_reads_gpu_result(self):
        self.fx.scheme('g', _gpu(('a', 'out')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'in')))
        plan = self.fx.run([('g1', ['g', 'c1'])])
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual(_ups(pp, _calls(self.fx.res, 'g1')['c1']), [('self', A)])
        self.assertEqual(pp.exit, [])          # host copy already valid

    def test_untracked_arrays_and_scalars_ignored(self):
        self.fx.scheme('g', {'run': [('a', 'inout'), ('ncol', 'in')]}, header_space='device')
        self.fx.scheme('c1', _gpu(('b', 'inout')))
        plan = self.fx.run([('g1', ['g', 'c1'])])
        self.assertEqual(sorted(plan.arrays), ['a'])

    def test_per_argument_override(self):
        self.fx.scheme('g', {'run': [('a', 'inout'), ('b', 'in', {'memory_space': 'host'})]},
                       header_space='device')
        plan = self.fx.run([('g1', ['g'])])
        self.assertEqual(sorted(plan.arrays), ['a'])


class TestHandoffStates(_Base):

    def test_none_has_no_boundary_traffic(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g'])], host_attrs={'a': {'host_access': 'none'}})
        self.assertIsNone(plan.phase_plan('g1', 'run'))
        self.assertEqual(plan.arrays['a'].handoff, frozenset({'D'}))

    def test_none_cpu_reader_and_writer(self):
        self.fx.scheme('c1', _gpu(('a', 'inout')))
        self.fx.scheme('g', _gpu(('a', 'in')), header_space='device')
        plan = self.fx.run([('g1', ['c1', 'g'])], host_attrs={'a': {'host_access': 'none'}})
        pp = plan.phase_plan('g1', 'run')
        calls = _calls(self.fx.res, 'g1')
        self.assertEqual(_ups(pp, calls['c1']), [('self', A)])
        self.assertEqual(_ups(pp, calls['g']), [('device', A)])
        self.assertEqual(pp.exit, [])

    def test_read_keeps_both_copies(self):
        self.fx.scheme('g', _gpu(('a', 'in')), header_space='device')
        plan = self.fx.run([('g1', ['g'])], host_attrs={'a': {'host_access': 'read'}})
        self.assertIsNone(plan.phase_plan('g1', 'run'))

    def test_read_restores_both_after_gpu_write(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g'])], host_attrs={'a': {'host_access': 'read'}})
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual(_ups(pp, _calls(self.fx.res, 'g1')['g']), [])
        self.assertEqual([(u.kind, u.expr) for u in pp.exit], [('self', A)])

    def test_read_restores_both_after_cpu_write(self):
        self.fx.scheme('g', _gpu(('a', 'in')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'inout')))
        plan = self.fx.run([('g1', ['g', 'c1'])], host_attrs={'a': {'host_access': 'read'}})
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual([(u.kind, u.expr) for u in pp.exit], [('device', A)])

    def test_host_managed_device_copy(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g'])], host_attrs={'a': {'memory_space': 'device'}})
        self.assertEqual(plan.arrays['a'].owner, 'host_managed')
        self.assertEqual(plan.created_host_arrays(), [])
        self.assertIsNone(plan.phase_plan('g1', 'run'))

    def test_groups_are_independent(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('h', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g']), ('g2', ['h'])])
        for grp, sch in (('g1', 'g'), ('g2', 'h')):
            pp = plan.phase_plan(grp, 'run')
            self.assertEqual(_ups(pp, _calls(self.fx.res, grp)[sch]), [('device', A)])
            self.assertEqual([(u.kind, u.expr) for u in pp.exit], [('self', A)])

    def test_interstitial_stays_on_device(self):
        self.fx.scheme('p', _gpu(('w', 'out')), header_space='device')
        self.fx.scheme('q', _gpu(('w', 'in'), ('a', 'out')), header_space='device')
        plan = self.fx.run([('g1', ['p', 'q'])])
        w = plan.arrays['ccpp_suite_data(1)%w']
        self.assertEqual(w.owner, 'suite')
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual([(u.kind, u.expr) for u in pp.exit], [('self', A)])


class TestSubcycles(_Base):

    def test_alternating_gpu_cpu_in_loop(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'inout')))
        plan = self.fx.run([('g1', [('subcycle', 2, ['g', 'c1'])])])
        pp = plan.phase_plan('g1', 'run')
        calls = _calls(self.fx.res, 'g1')
        self.assertEqual(_ups(pp, calls['g']), [('device', A)])
        self.assertEqual(_ups(pp, calls['c1']), [('self', A)])
        self.assertEqual(pp.loop_entry, {})
        self.assertEqual(pp.exit, [])

    def test_gpu_only_loop_hoists_copy(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', [('subcycle', 3, ['g'])])])
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual(_ups(pp, _calls(self.fx.res, 'g1')['g']), [])
        (entry,) = pp.loop_entry.values()
        self.assertEqual([(u.kind, u.expr) for u in entry], [('device', A)])
        self.assertEqual([(u.kind, u.expr) for u in pp.exit], [('self', A)])

    def test_cpu_written_scratch_read_after_loop(self):
        self.fx.scheme('c1', _gpu(('w', 'out')))
        self.fx.scheme('g', _gpu(('w', 'in'), ('a', 'out')), header_space='device')
        plan = self.fx.run([('g1', [('subcycle', 2, ['c1']), 'g'])])
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual(pp.loop_entry, {})
        self.assertEqual(_ups(pp, _calls(self.fx.res, 'g1')['g']),
                         [('device', 'ccpp_suite_data(1)%w(lb:ub, 1:nlev)')])

    def test_zero_iteration_loop_keeps_state_consistent(self):
        # Loop count is a variable: the loop may run zero times.  A GPU
        # write inside must not leave "valid on device" claimed if the loop
        # did not run, nor "valid on host" if it did.
        self.fx.scheme('g', _gpu(('a', 'out')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'in')))
        plan = self.fx.run([('g1', [('subcycle', 'number_of_subcycles', ['g']), 'c1'])],
                           with_nsub=True)
        pp = plan.phase_plan('g1', 'run')
        # Host copy is valid before the loop; after a GPU write it is not.
        # Making both valid before the loop keeps the post-loop state
        # independent of the iteration count; then the CPU reader needs
        # the device result copied back.
        (entry,) = pp.loop_entry.values()
        self.assertEqual([(u.kind, u.expr) for u in entry], [('device', A)])
        self.assertEqual(_ups(pp, _calls(self.fx.res, 'g1')['c1']), [('self', A)])


class TestOptionalArguments(_Base):

    def test_guarded_update(self):
        self.fx.scheme('g', {'run': [('opt', 'in', {'optional': 'True'}), ('a', 'out')]},
                       header_space='device')
        plan = self.fx.run([('g1', ['g'])], with_flag=True)
        pp = plan.phase_plan('g1', 'run')
        (u,) = pp.pre[_calls(self.fx.res, 'g1')['g']]
        self.assertEqual((u.kind, u.expr, u.guard),
                         ('device', 'opt(lb:ub, 1:nlev)', '(flag)'))

    def test_guarded_write_restores_state(self):
        self.fx.scheme('g', {'run': [('opt', 'inout', {'optional': 'True'})]},
                       header_space='device')
        plan = self.fx.run([('g1', ['g'])], with_flag=True)
        pp = plan.phase_plan('g1', 'run')
        call = _calls(self.fx.res, 'g1')['g']
        self.assertEqual([(u.kind, u.guard) for u in pp.pre[call]], [('device', '(flag)')])
        self.assertEqual([(u.kind, u.guard) for u in pp.post[call]], [('self', '(flag)')])
        self.assertEqual(pp.exit, [])


class TestUnifiedMemory(_Base):

    def test_updates_become_waits(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'in')))
        self.fx.scheme('h', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g', 'c1', 'h'])], memory_model='unified')
        pp = plan.phase_plan('g1', 'run')
        calls = _calls(self.fx.res, 'g1')
        self.assertEqual(pp.pre.get(calls['g']), None)
        self.assertEqual([u.kind for u in pp.pre[calls['c1']]], ['wait'])
        self.assertEqual(pp.pre.get(calls['h']), None)
        self.assertEqual([u.kind for u in pp.exit], ['wait'])
        self.assertEqual(plan.created_host_arrays(), [])


class TestRestrictions(_Base):

    def test_transform_on_device_argument(self):
        self.fx.scheme('g', {'run': [('a', 'inout', {'units': 'C'})]}, header_space='device')
        with self.assertRaisesRegex(CCPPError, 'transform'):
            self.fx.run([('g1', ['g'])])

    def test_register_phase(self):
        self.fx.scheme('g', {'register': [('a', 'in')], 'run': [('b', 'in')]},
                       header_space='device')
        with self.assertRaisesRegex(CCPPError, 'register'):
            self.fx.run([('g1', ['g'])])

    def test_transform_on_host_argument_is_fine(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('c1', {'run': [('a', 'inout', {'units': 'C'})]})
        plan = self.fx.run([('g1', ['g', 'c1'])])
        pp = plan.phase_plan('g1', 'run')
        self.assertEqual(_ups(pp, _calls(self.fx.res, 'g1')['c1']), [('self', A)])


########################################################################
# Emission (generated Fortran and datatable)
########################################################################

class TestEmission(_Base):

    def _two_scheme(self, **kw):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'in'), ('b', 'out')))
        return self.fx.run([('g1', ['g', 'c1'])], **kw)

    def test_group_cap_updates_in_order(self):
        self._two_scheme()
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        i_dev = text.index('!$acc update device(a(lb:ub, 1:nlev))')
        i_g = text.index('call g_run(')
        i_self = text.index('!$acc update self(a(lb:ub, 1:nlev))')
        i_c = text.index('call c1_run(')
        self.assertLess(i_dev, i_g)
        self.assertLess(i_g, i_self)
        self.assertLess(i_self, i_c)

    def test_suite_cap_creates_and_deletes(self):
        self._two_scheme()
        text = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertIn('!$acc enter data create(a)', text)
        self.assertIn('!$acc exit data delete(a)', text)
        self.assertIn('use hmod, only: a', text)
        self.assertIn('subroutine tsuite_device_update_device(errmsg, errflg)', text)
        # a is host_access = read_write: valid on the host whenever the host
        # has control, so the sync routines must not copy it.
        self.assertNotIn('!$acc update self(a)', text)

    def test_host_cap_sync_entry_points(self):
        self._two_scheme()
        text = self.fx.text('thost_ccpp_cap.F90')
        self.assertIn('public :: ccpp_physics_device_update_device', text)
        self.assertIn('public :: ccpp_physics_device_update_host', text)
        self.assertIn('call tsuite_device_update_host(errmsg, errflg)', text)

    def test_datatable_section(self):
        self._two_scheme()
        text = self.fx.text('datatable.xml')
        self.assertIn('<device_residency directive="acc" memory_model="discrete">', text)
        self.assertIn('<array name="a" standard_names="field_a" owner="host" handoff="host"', text)

    def test_flag_off_has_no_gpu_code(self):
        self._two_scheme(directive=None)
        for f in ('ccpp_tsuite_g1_cap.F90', 'ccpp_tsuite_cap.F90',
                  'thost_ccpp_cap.F90', 'datatable.xml'):
            text = self.fx.text(f)
            self.assertNotIn('$acc', text)
            self.assertNotIn('device_update', text)
            self.assertNotIn('device_residency', text)

    def test_unified_emits_waits_only(self):
        self._two_scheme(memory_model='unified')
        group = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('!$acc wait', group)
        self.assertNotIn('!$acc update', group)
        suite = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertNotIn('enter data', suite)

    def test_debug_checks(self):
        self._two_scheme(debug=True)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('use openacc, only: acc_is_present', text)
        self.assertIn('if (.not. acc_is_present(a)) then', text)
        # c1 only reads a while both copies are valid: checked after it.
        self.assertIn('ccpp_gpu_chk', text)
        self.assertIn('use ccpp_kinds, only: kind_phys', text)
        self.assertTrue(all(len(l) <= 132 for l in text.splitlines()))

    def test_optional_array_guarded_in_suite_cap(self):
        self.fx.scheme('g', {'run': [('opt', 'in', {'optional': 'True'})]},
                       header_space='device')
        self.fx.run([('g1', ['g'])], with_flag=True)
        text = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertIn('if ((flag)) then\n      !$acc enter data create(opt)', text)
        self.assertIn('use hmod, only: flag, opt', text)


class TestTraceAndSyncDebug(_Base):

    def test_sync_debug_mirrors_device_writes(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'in')))
        plan = self.fx.run([('g1', ['g', 'c1'])], sync_debug=True)
        pp = plan.phase_plan('g1', 'run')
        calls = _calls(self.fx.res, 'g1')
        self.assertEqual(_ups(pp, calls['g'], 'post'), [('self', A)])
        # The host copy is already valid, so the CPU reader needs nothing.
        self.assertEqual(_ups(pp, calls['c1']), [])
        self.assertEqual(pp.exit, [])

    def test_sync_debug_off_by_default(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        plan = self.fx.run([('g1', ['g'])])
        self.assertEqual(plan.phase_plan('g1', 'run').post, {})

    def test_trace_lines(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.run([('g1', ['g'])], trace=True)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn("'before g_run',", text)
        self.assertIn('ccpp_gpu_p = acc_is_present(a)', text)
        self.assertIn('shape(a(lb:ub, 1:nlev)),', text)
        self.assertIn("'ub=',", text)
        self.assertIn("'copy to device',", text)
        self.assertIn('use openacc, only: acc_is_present', text)
        self.assertTrue(all(len(l) <= 132 for l in text.splitlines()))

    def test_trace_unified_has_no_presence_query(self):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.run([('g1', ['g'])], trace=True, memory_model='unified')
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn("'before g_run',", text)
        self.assertNotIn('acc_is_present', text)


class TestOpenMPBackend(_Base):

    def _run(self, **kw):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'in')))
        return self.fx.run([('g1', ['g', 'c1'])], directive='omp', **kw)

    def test_same_plan_as_openacc(self):
        omp = self._run()
        pp = omp.phase_plan('g1', 'run')
        calls = _calls(self.fx.res, 'g1')
        self.assertEqual(_ups(pp, calls['g']), [('device', A)])
        self.assertEqual(_ups(pp, calls['c1']), [('self', A)])

    def test_openmp_spelling(self):
        self._run()
        group = self.fx.text('ccpp_tsuite_g1_cap.F90')
        suite = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertIn('!$omp target update to(a(lb:ub, 1:nlev))', group)
        self.assertIn('!$omp target update from(a(lb:ub, 1:nlev))', group)
        self.assertIn('!$omp target enter data map(alloc: a)', suite)
        self.assertIn('!$omp target exit data map(delete: a)', suite)
        for text in (group, suite, self.fx.text('thost_ccpp_cap.F90')):
            self.assertNotIn('!$acc', text)

    def test_openmp_debug_uses_portable_presence_helper(self):
        self._run(debug=True, trace=True)
        group = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('if (.not. ccpp_gpu_is_present(a)) then', group)
        self.assertIn('logical function ccpp_gpu_is_present(a)', group)
        self.assertIn('type(*), dimension(..), target, intent(in) :: a', group)
        self.assertNotIn('acc_is_present', group)
        self.assertNotIn('use openacc', group)

    def test_openmp_helper_only_when_needed(self):
        self._run()
        self.assertNotIn('ccpp_gpu_is_present', self.fx.text('ccpp_tsuite_g1_cap.F90'))

    def test_openmp_unified_uses_taskwait(self):
        self._run(memory_model='unified')
        group = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('!$omp taskwait', group)
        self.assertNotIn('target update', group)


########################################################################
# Constituents (end-to-end-tests/gpu_constituents metadata)
########################################################################

_CONST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                          'end-to-end-tests', 'gpu_constituents')
_CONST_V = 'ccpp_model_constituents_obj(1)%vars_layer'
_CONST_T = 'ccpp_model_constituents_obj(1)%vars_layer_tend'


class TestConstituents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        d = _CONST_DIR
        schemes = [os.path.join(d, n + '.meta') for n in
                   ('const_reg', 'gpu_liq', 'cpu_ice', 'gpu_apply', 'cpu_check_const')]
        _, results = capgen('test_host',
                            [os.path.join(d, 'const_data.meta'), os.path.join(d, 'main.meta')],
                            schemes, [os.path.join(d, 'suite_gpu_constituents.xml')],
                            cls.out, {}, return_state=True, directive='acc')
        cls.res = results[0]
        cls.plan = cls.res.device_plan

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_tracked_as_whole_arrays(self):
        a = self.plan.arrays
        self.assertEqual(a[_CONST_V].owner, 'constituent')
        self.assertEqual(a[_CONST_T].owner, 'constituent')
        self.assertEqual(a[_CONST_V].parents, ['ccpp_model_constituents_obj'])
        self.assertEqual(a[_CONST_V].handoff, frozenset({'H'}))
        # Both single-constituent standard names roll up into one unit.
        self.assertIn('cloud_liquid_dry_mixing_ratio', a[_CONST_V].standard_names)
        self.assertIn('cloud_ice_dry_mixing_ratio', a[_CONST_V].standard_names)

    def test_single_constituent_write_brings_whole_unit_up_to_date(self):
        pp = self.plan.phase_plan('phys', 'run')
        call = _calls(self.res, 'phys')['gpu_liq']
        unit_v = _CONST_V + '(lb:ub, 1:pver, :)'
        unit_t = _CONST_T + '(lb:ub, 1:pver, :)'
        ups = _ups(pp, call)
        self.assertIn(('device', unit_v), ups)
        # liq_tend is intent(out) for ONE constituent: the rest of the
        # tendency unit must be valid on the device before the write.
        self.assertIn(('device', unit_t), ups)

    def test_overlapping_all_constituent_access(self):
        pp = self.plan.phase_plan('phys', 'run')
        calls = _calls(self.res, 'phys')
        self.assertEqual(sorted(_ups(pp, calls['cpu_ice'])),
                         sorted([('self', _CONST_V + '(lb:ub, 1:pver, :)'),
                                 ('self', _CONST_T + '(lb:ub, 1:pver, :)')]))
        self.assertEqual(sorted(_ups(pp, calls['gpu_apply'])),
                         sorted([('device', _CONST_V + '(lb:ub, 1:pver, :)'),
                                 ('device', _CONST_T + '(lb:ub, 1:pver, :)')]))

    def test_device_copies_created_with_parent(self):
        with open(os.path.join(self.out, 'ccpp_gpu_const_suite_cap.F90')) as fh:
            text = fh.read()
        i_parent = text.index('!$acc enter data create(ccpp_model_constituents_obj)')
        i_member = text.index('!$acc enter data create({})'.format(_CONST_V))
        self.assertLess(i_parent, i_member)
        self.assertIn('use ccpp_host_constituents, only: ccpp_model_constituents_obj', text)

    def test_constituents_not_in_sync_routines(self):
        # read_write handoff: valid on the host whenever the host has
        # control, so neither sync routine may touch them.
        self.assertEqual(self.plan.synced_host_arrays('device'), [])
        self.assertEqual(self.plan.synced_host_arrays('host'), [])


class TestSyncRoutineSelection(_Base):

    def test_only_arrays_that_need_it(self):
        self.fx.scheme('g', _gpu(('a', 'inout'), ('b', 'inout'), ('c', 'inout')),
                       header_space='device')
        plan = self.fx.run([('g1', ['g'])], host_attrs={
            'a': {'host_access': 'none'}, 'b': {'host_access': 'read'}})
        names = lambda d: sorted(x.base_expr for x in plan.synced_host_arrays(d))
        self.assertEqual(names('device'), ['a', 'b'])
        self.assertEqual(names('host'), ['a'])
        text = self.fx.text('ccpp_tsuite_cap.F90')
        host_sync = text[text.index('subroutine tsuite_device_update_host'):]
        self.assertIn('!$acc update self(a)', host_sync)
        self.assertNotIn('update self(c)', host_sync)
        self.assertNotIn('update self(b)', host_sync)


class TestCrashTrace(_Base):

    def _run(self, directive='acc'):
        self.fx.scheme('g', {'run': [('a', 'inout'), ('ncol', 'in')]}, header_space='device')
        self.fx.run([('g1', ['g'])], trace=True, directive=directive)
        return self.fx.text('ccpp_tsuite_g1_cap.F90')

    def test_slice_presence_and_device_read(self):
        text = self._run()
        self.assertIn('ccpp_gpu_p1 = acc_is_present(a(lb:lb, 1:1))', text)
        self.assertIn('ccpp_gpu_p2 = acc_is_present(a(ub:ub, nlev:nlev))', text)
        self.assertIn('!$acc serial present(a) copyout(ccpp_gpu_dv)', text)
        self.assertIn("'plan:',", text)
        self.assertIn("'valid on host and device',", text)
        self.assertIn('is not fully present on the device', text)

    def test_scalar_arguments_printed(self):
        text = self._run()
        self.assertIn("'scalar arguments:',", text)
        self.assertIn("'ncol=',", text)

    def test_update_warns_when_absent(self):
        text = self._run()
        i_check = text.index("has no device copy at this copy to device;")
        i_update = text.index('!$acc update device(a(lb:ub, 1:nlev))')
        self.assertLess(i_check, i_update)
        self.assertIn("'the runtime will abort here'", text)

    def test_openmp_trace(self):
        text = self._run('omp')
        self.assertIn('!$omp target map(from: ccpp_gpu_dv)', text)
        self.assertIn('ccpp_gpu_is_present(a(lb:lb, 1:1))', text)
        self.assertIn("'OpenMP skips this copy and kernels copy the array implicitly'", text)
        suite = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertIn('logical function ccpp_gpu_is_present(a)', suite)

    def test_lines_fit(self):
        for d in ('acc', 'omp'):
            self.fx.cleanup()
            self.fx = _Fixture()
            text = self._run(d)
            self.assertTrue(all(len(l) <= 132 for l in text.splitlines()), d)

    def test_output_scalars_not_printed(self):
        text = self._run()
        line = next(l for l in text.splitlines() if "'scalar arguments:'" in l)
        self.assertNotIn('errflg', line)


########################################################################
# Fortran host calling C++ schemes (language = c++)
########################################################################

def _cxx(meta_text):
    return meta_text.replace('  type = scheme\n\n', '  type = scheme\n  language = c++\n\n', 1)


class TestCxxSchemes(_Base):

    def _scheme(self, name, phases, header_space=None):
        path = self.fx.write(name + '.meta', _cxx(_scheme_meta(name, phases, header_space)))
        self.fx.schemes.append(path)
        self.fx.write(name + '.cpp', '// placeholder\n')

    def test_interface_block_instead_of_use(self):
        self._scheme('cs', {'run': [('ncol', 'in'), ('s', 'out'), ('a', 'inout'), ('b', 'in')]})
        self.fx.run([('g1', ['cs'])], directive=None)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertNotIn('use cs,', text)
        self.assertIn("bind(C, name='cs_run')", text)
        self.assertIn('integer(c_int), value, intent(in) :: ncol', text)
        self.assertIn('real(c_double), intent(out) :: s', text)
        self.assertIn('real(c_double), intent(inout) :: a(*)', text)
        self.assertIn('real(c_double), intent(in) :: b(*)', text)
        self.assertIn('character(kind=c_char), intent(out) :: errmsg(*)', text)
        # The call site is unchanged: keyword arguments.
        self.assertIn('call cs_run( &', text)

    def test_c_header(self):
        self._scheme('cs', {'run': [('ncol', 'in'), ('a', 'inout'), ('b', 'in')]})
        self.fx.run([('g1', ['cs'])], directive=None)
        hdr = self.fx.text('cs_ccpp.h')
        self.assertIn('extern "C" {', hdr)
        self.assertIn('void cs_run(', hdr)
        self.assertIn('    int ncol,', hdr)
        self.assertIn('    double* a,', hdr)
        self.assertIn('    const double* b,', hdr)
        self.assertIn('    int* errflg);', hdr)

    def test_real32_kind_maps_to_float(self):
        self._scheme('cs', {'run': [('a', 'inout')]})
        host = self.fx.write('hmod.meta', _host_meta())
        ctrl = self.fx.write('main.meta', _CONTROL)
        suite = self.fx.write('suite.xml', _suite_xml([('g1', ['cs'])]))
        capgen('thost', [host, ctrl], self.fx.schemes, [suite], self.fx.out,
               {'kind_phys': ('iso_fortran_env', 'REAL32')})
        self.assertIn('real(c_float), intent(inout) :: a(*)',
                      self.fx.text('ccpp_tsuite_g1_cap.F90'))
        self.assertIn('float* a', self.fx.text('cs_ccpp.h'))

    def test_logical_scalar_and_array(self):
        self._scheme('cs', {'run': [('flag', 'in'), ('mask', 'out')]})
        self.fx.run([('g1', ['cs'])], with_flag=True, directive=None)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('use, intrinsic :: iso_c_binding, only: c_signed_char', text)
        self.assertIn('integer(c_signed_char), value, intent(in) :: flag', text)
        self.assertIn('integer(c_signed_char), intent(out) :: mask(*)', text)
        # Converted through c_bool temporaries in a block around the call.
        self.assertIn('ccpp_cxx_l_flag = merge(1_c_signed_char, 0_c_signed_char, flag)', text)
        self.assertIn('flag=ccpp_cxx_l_flag', text)
        self.assertIn('allocate(ccpp_cxx_l_mask(size(ccpp_suite_data(1)%mask(lb:ub, 1:nlev), 1)', text)
        self.assertRegex(text, r'mask\(lb:ub, 1:nlev\) = ccpp_cxx_l_mask /= 0_c_signed_char')
        self.assertNotIn('ccpp_cxx_l_mask = merge(', text)   # out: no copy in
        header = self.fx.text('cs_ccpp.h')
        self.assertIn('bool flag,', header)
        self.assertIn('bool* mask,', header)
        self.assertIn('#include <stdbool.h>', header)

    def test_optional_arguments(self):
        self._scheme('cs', {'run': [('opt', 'in', {'optional': 'True'})]})
        self.fx.run([('g1', ['cs'])], with_flag=True, directive=None)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('real(c_double), intent(in), optional :: opt(*)', text)
        self.assertIn('opt=opt_p%ptr', text)      # disassociated -> absent
        header = self.fx.text('cs_ccpp.h')
        self.assertIn('const double* opt,  /* optional: NULL when absent */', header)

    def test_optional_scalar_is_by_reference(self):
        self._scheme('cs', {'run': [('flag', 'in', {'optional': 'True'})]})
        self.fx.run([('g1', ['cs'])], with_flag=True, directive=None)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('integer(c_signed_char), intent(in), optional :: flag', text)
        self.assertIn('const bool* flag,', self.fx.text('cs_ccpp.h'))

    def test_allocatable_argument(self):
        self._scheme('cs', {'run': [('w', 'out', {'allocatable': 'True'})]})
        self.fx.run([('g1', ['cs'])], directive=None)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        # An allocation handle, not an allocatable dummy (NVHPC, see
        # cxx_schemes): the scheme calls back into Fortran to allocate.
        self.assertIn('type, bind(C) :: ccpp_alloc_double_2_t', text)
        self.assertIn('type(ccpp_alloc_double_2_t), intent(in) :: w', text)
        self.assertRegex(text, r'if \(allocated\((\S+%w)\)\) deallocate\(\1\)')
        self.assertIn('ccpp_cxx_a_w%allocate = c_funloc(ccpp_cxx_alloc_double_2)', text)
        self.assertRegex(text, r'if \(allocated\(ccpp_cxx_b_w%a\)\) call move_alloc\(ccpp_cxx_b_w%a, \S+%w\)')
        self.assertIn("bind(C, name='ccpp_tsuite_g1_cap_alloc_double_2')", text)
        self.assertIn('allocate(box%a(lower(1):upper(1), lower(2):upper(2)))', text)
        header = self.fx.text('cs_ccpp.h')
        self.assertIn('double *(*allocate)(void *handle, const int *lower, const int *upper);', header)
        self.assertIn('ccpp_alloc_double_2_t* w,', header)
        self.assertNotIn('ISO_Fortran_binding', header)

    def _ddt_run(self, intent='inout', device=False, member_extra=''):
        host = textwrap.dedent("""\
            [ccpp-table-properties]
              name = st_t
              type = ddt
            [ccpp-arg-table]
              name = st_t
              type = ddt
            [ ps ]
              standard_name = surface_air_pressure
              units = Pa
              dimensions = (horizontal_dimension)
              type = real
              kind = kind_phys
            {member_extra}[ nstep ]
              standard_name = step_count
              units = count
              dimensions = ()
              type = integer
            [ lmask ]
              standard_name = land_mask
              units = flag
              dimensions = (horizontal_dimension)
              type = logical
            """).replace('{member_extra}', member_extra) + _host_meta().replace(
            '[ncol]', '[st]\n  standard_name = model_state\n  units = DDT\n'
            '  dimensions = ()\n  type = st_t\n[ncol]', 1)
        self.fx.write('cs.meta', _cxx(textwrap.dedent("""\
            [ccpp-table-properties]
              name = cs
              type = scheme

            [ccpp-arg-table]
              name = cs_run
              type = scheme
            {dev}[ st ]
              standard_name = model_state
              units = DDT
              dimensions = ()
              type = st_t
              intent = {intent}
            """.format(intent=intent, dev='  memory_space = device\n' if device else ''))
            + _var('errmsg', (), 'out') + '\n'
            + _var('errflg', (), 'out') + '\n'))
        self.fx.write('cs.cpp', '// placeholder\n')
        h = self.fx.write('hmod.meta', host)
        ctrl = self.fx.write('main.meta', _CONTROL)
        suite = self.fx.write('suite.xml', _suite_xml([('g1', ['cs'])]))
        _, results = capgen('thost', [h, ctrl], [os.path.join(self.fx.dir, 'cs.meta')],
                            [suite], self.fx.out, {}, return_state=True,
                            directive='acc' if device else None)
        return results[0].device_plan

    def test_derived_type_descriptor(self):
        self._ddt_run()
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        # One bind(C) descriptor, one argument (not one per member).
        self.assertIn('type, bind(C) :: ccpp_st_t_t', text)
        self.assertIn('type(c_ptr) :: ps', text)
        self.assertIn('integer(c_int) :: ps_shape(1)', text)
        self.assertIn('type(c_ptr) :: nstep', text)
        self.assertIn('! (member lmask not accessible from C++: logical)', text)
        self.assertIn('import :: ccpp_st_t_t', text)
        self.assertIn('type(ccpp_st_t_t), intent(in) :: st', text)
        self.assertIn('ccpp_cxx_d_st%ps = ccpp_cxx_loc(st%ps)', text)
        self.assertIn('ccpp_cxx_d_st%ps_shape = shape(st%ps)', text)
        self.assertIn('ccpp_cxx_d_st%ps_lbound = lbound(st%ps)', text)
        self.assertIn('st=ccpp_cxx_d_st', text)
        self.assertIn('function ccpp_cxx_loc(a) result(p)', text)
        header = self.fx.text('cs_ccpp.h')
        self.assertIn('typedef struct {', header)
        self.assertIn('    double *ps;', header)
        self.assertIn('    int ps_shape[1];', header)
        self.assertIn('    int *nstep;', header)
        self.assertIn('} ccpp_st_t_t;', header)
        self.assertIn('ccpp_st_t_t* st,', header)

    def test_derived_type_intent_in_is_const(self):
        self._ddt_run(intent='in')
        self.assertIn('const ccpp_st_t_t* st,', self.fx.text('cs_ccpp.h'))

    def test_device_allocatable(self):
        # A device-side C++ scheme allocates w; a host Fortran scheme reads it.
        self._scheme('cs', {'run': [('w', 'out', {'allocatable': 'True'})]},
                     header_space='device')
        self.fx.scheme('rd', {'run': [('w', 'in', {'allocatable': 'True'})]})
        plan = self.fx.run([('g1', ['cs', 'rd'])])
        arr = plan.arrays['ccpp_suite_data(1)%w']
        self.assertEqual(arr.owner, 'scheme_alloc')
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        # Device allocation callback: deletes the old device copy, creates the new.
        self.assertIn('function ccpp_cxx_alloc_acc_double_2(handle, lower, upper)', text)
        fn = text[text.index('function ccpp_cxx_alloc_acc_double_2'):]
        self.assertLess(fn.index('!$acc exit data delete(box%a)'), fn.index('deallocate(box%a)'))
        self.assertLess(fn.index('allocate(box%a(lower(1)'), fn.index('!$acc enter data create(box%a)'))
        self.assertIn('ccpp_cxx_a_w%allocate = c_funloc(ccpp_cxx_alloc_acc_double_2)', text)
        # intent(out): old device copy deleted before the cap deallocates.
        self.assertRegex(text, r'if \(allocated\(ccpp_suite_data\(1\)%w\)\) then\s+'
                               r'!\$acc exit data delete\(ccpp_suite_data\(1\)%w\)\s+'
                               r'deallocate\(ccpp_suite_data\(1\)%w\)')
        # Host reader: copy back only while allocated.
        self.assertRegex(text, r'if \(allocated\(ccpp_suite_data\(1\)%w\)\) then\s+'
                               r'!\$acc update self\(ccpp_suite_data\(1\)%w\)')
        # Not created at init; deleted at final if still allocated.
        suite = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertNotIn('enter data create(ccpp_suite_data(1)%w)', suite)
        self.assertRegex(suite, r'if \(allocated\(ccpp_suite_data\(1\)%w\)\) then\s+'
                                r'!\$acc exit data delete\(ccpp_suite_data\(1\)%w\)')

    def test_device_allocatable_needs_cxx_allocator(self):
        # A Fortran scheme allocating it: still rejected for device use.
        self.fx.scheme('fa', {'run': [('w', 'out', {'allocatable': 'True'})]})
        self._scheme('cs', {'run': [('w', 'in', {'allocatable': 'True'})]},
                     header_space='device')
        with self.assertRaisesRegex(CCPPError, 'device-side C\\+\\+ scheme'):
            self.fx.run([('g1', ['fa', 'cs'])])

    def test_device_derived_type(self):
        plan = self._ddt_run(device=True)
        # The array member is tracked like an ordinary host array.
        self.assertIn('st%ps', plan.arrays)
        self.assertEqual(plan.arrays['st%ps'].parents, ['st'])
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        i = text.index('call cs_run(')
        self.assertIn('!$acc update device(st%ps)', text[:i])
        self.assertIn('!$acc update self(st%ps)', text[i:])
        suite = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertIn('!$acc enter data create(st)', suite)
        self.assertIn('!$acc enter data create(st%ps)', suite)
        header = self.fx.text('cs_ccpp.h')
        self.assertIn('array members present on the device', header)

    def test_device_derived_type_allocatable_member(self):
        with self.assertRaisesRegex(CCPPError, 'members declared allocatable'):
            self._ddt_run(device=True, member_extra='  allocatable = True\n')

    def test_device_logical_array_packed_and_converted(self):
        self._scheme('cs', {'run': [('flag', 'inout'), ('mask', 'out')]},
                     header_space='device')
        plan = self.fx.run([('g1', ['cs'])], with_flag=True)
        pp = plan.phase_plan('g1', 'run')
        call = _calls(self.fx.res, 'g1')['cs']
        packs = {p.local_name: p for p in pp.cxx_pack[call]}
        self.assertTrue(packs['mask'].logical)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('integer(c_signed_char), allocatable :: ccpp_cxx_mask(:,:)', text)
        self.assertIn('!$acc enter data create(ccpp_cxx_mask)', text)
        self.assertRegex(text, r'%mask\(\(lb\) \+ ccpp_cxx_j1 - 1, \(1\) \+ ccpp_cxx_j2 - 1\) = '
                               r'ccpp_cxx_mask\(ccpp_cxx_j1, &\s+ccpp_cxx_j2\) /= 0_c_signed_char')
        self.assertIn('mask=ccpp_cxx_mask', text)
        # Not converted again on the host.
        self.assertNotIn('ccpp_cxx_l_mask', text)
        # Logical scalar out/inout on a device-side scheme: host conversion.
        self.assertIn('ccpp_cxx_l_flag = merge(1_c_signed_char, 0_c_signed_char, flag)', text)
        self.assertIn('flag = ccpp_cxx_l_flag /= 0_c_signed_char', text)

    def test_device_optional_array_guarded(self):
        self._scheme('cs', {'run': [('opt', 'in', {'optional': 'True'})]},
                     header_space='device')
        self.fx.run([('g1', ['cs'])], with_flag=True)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        i = text.index('allocate(ccpp_cxx_opt(')
        self.assertIn('if ((flag)) then', text[i - 200:i])
        self.assertIn('if (allocated(ccpp_cxx_opt)) then', text)
        self.assertIn('opt=ccpp_cxx_opt', text)
        # The device copy of the host array is updated only when present.
        self.assertRegex(text, r'if \(\(flag\)\) then\s+!\$acc update device\(opt\(lb:ub, 1:nlev\)\)')

    def test_still_unsupported(self):
        self._scheme('cs', {'run': [('mask', 'out', {'allocatable': 'True'})]})
        with self.assertRaisesRegex(CCPPError, 'allocatable logical'):
            self.fx.run([('g1', ['cs'])], with_flag=True, directive=None)

    def test_language_only_on_scheme_tables(self):
        bad = _host_meta().replace('  type = host\n[ccpp-arg-table]',
                                   '  type = host\n  language = c++\n[ccpp-arg-table]', 1)
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, 'h.meta')
            with open(p, 'w') as fh:
                fh.write(bad)
            with self.assertRaisesRegex(CCPPError, 'only supported on scheme tables'):
                parse_metadata_file(p)
        finally:
            shutil.rmtree(d)

    def test_device_arrays_are_packed(self):
        self._scheme('cs', {'run': [('ncol', 'in'), ('a', 'inout'), ('b', 'in')]},
                     header_space='device')
        plan = self.fx.run([('g1', ['cs'])])
        pp = plan.phase_plan('g1', 'run')
        call = _calls(self.fx.res, 'g1')['cs']
        packs = {p.local_name: (p.reads, p.writes) for p in pp.cxx_pack[call]}
        self.assertEqual(packs, {'a': (True, True), 'b': (True, False)})
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('!$acc enter data create(ccpp_cxx_a)', text)
        self.assertIn('a=ccpp_cxx_a', text)
        self.assertIn('b=ccpp_cxx_b', text)
        # b is intent(in): copied in, never copied back.
        self.assertEqual(text.count('= ccpp_cxx_b('), 0)
        self.assertEqual(text.count('= ccpp_cxx_a('), 1)
        self.assertIn('/* [device] */', self.fx.text('cs_ccpp.h'))

    def test_flag_off_unchanged_for_fortran_schemes(self):
        self.fx.scheme('g', _gpu(('a', 'inout')))
        self.fx.run([('g1', ['g'])], directive=None)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('use g, only: g_run', text)
        self.assertNotIn('interface', text)


########################################################################
# Timers (--timer-style)
########################################################################

class TestTimers(_Base):

    def _two_scheme(self, **kw):
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.scheme('c1', _gpu(('a', 'in'), ('b', 'out')))
        self.fx.run([('g1', ['g', 'c1'])], **kw)
        return self.fx.text('ccpp_tsuite_g1_cap.F90')

    def _between(self, text, first, last):
        return text[text.index(first):text.index(last)]

    def test_cesm_compute_and_transfer_timers(self):
        text = self._two_scheme(timer_style='cesm')
        self.assertIn('use perf_mod, only: t_startf, t_stopf', text)
        self.assertIn('call t_startf(trim(ccpp_timer_names(i)))', text)
        # h2d timer around the copy to the device, then the compute timer
        # (with a wait) around the GPU call, then d2h before the CPU reader.
        i_h2d = text.index('call ccpp_timer_start(1)  ! g_run_h2d')
        i_dev = text.index('!$acc update device(a(lb:ub, 1:nlev))')
        i_g0 = text.index('call ccpp_timer_start(2)  ! g_run\n')
        i_call = text.index('call g_run(')
        i_wait = text.index('!$acc wait')
        i_g1 = text.index('call ccpp_timer_stop(2)')
        i_d2h = text.index('call ccpp_timer_start(3)  ! c1_run_d2h')
        i_self = text.index('!$acc update self(a(lb:ub, 1:nlev))')
        self.assertTrue(i_h2d < i_dev < i_g0 < i_call < i_wait < i_g1
                        < i_d2h < i_self)
        # The CPU call gets a compute timer but no wait.
        cpu = self._between(text, 'call ccpp_timer_start(4)  ! c1_run',
                            'call ccpp_timer_stop(4)')
        self.assertIn('call c1_run(', cpu)
        self.assertNotIn('wait', cpu)

    def test_timer_stops_before_error_return(self):
        text = self._two_scheme(timer_style='cesm')
        call = text[text.index('call g_run('):]
        self.assertLess(call.index('call ccpp_timer_stop(2)'),
                        call.index('if (errflg /= 0) return'))

    def test_fms_style(self):
        text = self._two_scheme(timer_style='fms')
        self.assertIn('use mpp_mod, only: mpp_clock_id, mpp_clock_begin, '
                      'mpp_clock_end, CLOCK_ROUTINE', text)
        self.assertIn('mpp_clock_id(trim(ccpp_timer_names(i)), '
                      'grain=CLOCK_ROUTINE)', text)
        self.assertIn('!$  if (omp_get_thread_num() /= 0) return', text)
        self.assertIn('integer, save :: ccpp_timer_ids(ccpp_ntimers) = 0', text)

    def test_cpu_build_has_compute_timers_only(self):
        text = self._two_scheme(timer_style='cesm', directive=None)
        self.assertIn("'g_run'", text)
        self.assertIn("'c1_run'", text)
        self.assertNotIn('_h2d', text)
        self.assertNotIn('_d2h', text)
        self.assertNotIn('!$acc', text)

    def test_no_timers_by_default(self):
        text = self._two_scheme()
        self.assertNotIn('ccpp_timer', text)
        self.assertNotIn('perf_mod', text)

    def test_exit_timers(self):
        # A GPU write of a read_write host variable is copied back at group
        # exit, under <group>_<phase>_exit_d2h.
        self.fx.scheme('g', _gpu(('a', 'inout')), header_space='device')
        self.fx.run([('g1', ['g'])], timer_style='cesm')
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        self.assertIn('! g1_run_exit_d2h', text)

    def test_long_names_shortened_for_fms(self):
        from generator.timers import TimerRegistry
        reg = TimerRegistry('fms')
        long = 'dycore_energy_consistency_adjust_run'
        a = reg.names[reg.index(long) - 1]
        b = reg.names[reg.index(long + '_h2d') - 1]
        c = reg.names[reg.index(long + '_d2h') - 1]
        self.assertTrue(all(len(x) <= 32 for x in (a, b, c)))
        self.assertTrue(b.endswith('_h2d') and c.endswith('_d2h'))
        self.assertEqual(len({a, b, c}), 3)
        self.assertEqual(reg.index(long), 1)   # stable
        cesm = TimerRegistry('cesm')
        self.assertEqual(cesm.names[cesm.index(long) - 1], long)


########################################################################
# Suite variables sized by an init-phase scheme
########################################################################

class TestAllocatedAfterSizingScheme(_Base):

    def _run(self, **kw):
        # 'sz' sets the size in init; 'pr' produces the array in run; 'rd'
        # reads it on the host.
        self.fx.scheme('sz', {'init': [('nbin', 'out')]})
        self.fx.scheme('pr', {'run': [('nbin', 'in'), ('binned', 'out')]},
                       header_space=kw.pop('header_space', None))
        self.fx.scheme('rd', {'run': [('binned', 'in')]})
        return self.fx.run([('g1', ['sz', 'pr', 'rd'])], **kw)

    def test_allocated_right_after_sizing_scheme(self):
        self._run(directive=None)
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        i_call = text.index('call sz_init(')
        i_alloc = text.index('allocate(ccpp_suite_data(1)%binned(')
        self.assertLess(i_call, i_alloc)
        self.assertIn('if (.not. allocated(ccpp_suite_data(1)%binned)) then', text)
        self.assertIn('allocate(ccpp_suite_data(1)%binned(ncol, ccpp_suite_data(1)%nbin))', text)
        data = self.fx.text('ccpp_tsuite_data.F90')
        init_fields = data[data.index('subroutine suite_data_init_fields'):
                           data.index('end subroutine suite_data_init_fields')]
        self.assertNotIn('%binned(', init_fields)        # not at init start
        self.assertIn('if (allocated(ccpp_suite_data(i)%binned)) deallocate', data)

    def test_strict_allocatable_rejects(self):
        self.fx.scheme('sz', {'init': [('nbin', 'out')]})
        self.fx.scheme('pr', {'run': [('nbin', 'in'), ('binned', 'out')]})
        host = self.fx.write('hmod.meta', _host_meta())
        ctrl = self.fx.write('main.meta', _CONTROL)
        suite = self.fx.write('suite.xml', _suite_xml([('g1', ['sz', 'pr'])]))
        with self.assertRaisesRegex(CCPPError, 'strict-allocatable'):
            capgen('thost', [host, ctrl], self.fx.schemes, [suite],
                   self.fx.out, {}, return_state=True, strict_allocatable=True)

    def test_device_copy_created_after_allocation(self):
        plan = self._run(header_space='device')
        arr = plan.arrays['ccpp_suite_data(1)%binned']
        self.assertEqual(arr.owner, 'deferred')
        text = self.fx.text('ccpp_tsuite_g1_cap.F90')
        i_alloc = text.index('allocate(ccpp_suite_data(1)%binned(')
        i_create = text.index('!$acc enter data create(ccpp_suite_data(1)%binned)')
        self.assertLess(i_alloc, i_create)
        # Host reader: copy back only while allocated; not created at init;
        # deleted at final if allocated.
        self.assertRegex(text, r'if \(allocated\(ccpp_suite_data\(1\)%binned\)\) then\s+'
                               r'!\$acc update self\(ccpp_suite_data\(1\)%binned\(lb:ub, :\)\)'
                               r'|if \(allocated\(ccpp_suite_data\(1\)%binned\)\) then\s+'
                               r'!\$acc update self\(ccpp_suite_data\(1\)%binned')
        suite = self.fx.text('ccpp_tsuite_cap.F90')
        self.assertNotIn('enter data create(ccpp_suite_data(1)%binned)', suite)
        self.assertRegex(suite, r'if \(allocated\(ccpp_suite_data\(1\)%binned\)\) then\s+'
                                r'!\$acc exit data delete\(ccpp_suite_data\(1\)%binned\)')


if __name__ == '__main__':
    unittest.main()
