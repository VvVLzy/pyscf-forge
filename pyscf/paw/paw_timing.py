'''Wall-clock timers for PAW: one J/K/XC line per Fock build, and a summary of setup and
the SCF printed when the PAW object is garbage collected.'''
import contextlib
import time


class PAWTimingMixin:
    '''timings[label] holds the seconds of every call and timing_starts[label] when each
    began; a J, K or XC build records each piece once per build, summed over spins.'''

    _report_depth = 0  # timing_block(report=True) blocks currently open

    def _init_timing(self):
        self.timings = {}        # label -> seconds of each call
        self.timing_starts = {}  # label -> perf_counter() at the start of each call
        self.scf_runs = []       # (start, end, cycles) of each timed mf.kernel
        self._fock_pending = {}  # phase -> seconds, for the next J/K/XC line
        self._fock_builds = 0    # J/K/XC lines printed so far

    def _record(self, label, seconds, start):
        self.timings.setdefault(label, []).append(seconds)
        self.timing_starts.setdefault(label, []).append(start)

    @contextlib.contextmanager
    def timing_block(self, label, report=False):
        '''Time the enclosed block under `label`; report=True also prints "label ..." on
        entry and "label ... done (t s)" on exit, indented by the open reported blocks.'''
        pad = '  ' * self._report_depth
        if report:
            print(f'{pad}{label} ...', flush=True)
            self._report_depth += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record(label, elapsed, t0)
            if report:
                self._report_depth -= 1
        if report:
            print(f'{pad}{label} ... done ({elapsed:.4f} s)', flush=True)

    @contextlib.contextmanager
    def build_timer(self, phase):
        '''Time one J, K or XC build, each piece under `with lap(piece):`; records
        '<phase> Total' and '<phase> <piece>', and holds the total for the J/K/XC line.'''
        pieces = {}

        @contextlib.contextmanager
        def lap(piece):
            t0 = time.perf_counter()
            try:
                yield
            finally:
                pieces[piece] = pieces.get(piece, 0.0) + time.perf_counter() - t0

        t0 = time.perf_counter()
        yield lap
        total = time.perf_counter() - t0
        self._record(f'{phase} Total', total, t0)
        for piece, seconds in pieces.items():
            self._record(f'{phase} {piece}', seconds, t0)
        self._fock_pending[phase] = self._fock_pending.get(phase, 0.0) + total

    def _print_fock_build(self):
        '''Print the J, K and XC times since the last call as one line ('---' for a phase
        not built), then reset them.'''
        if not self._fock_pending:
            return
        self._fock_builds += 1
        columns = []
        for phase in ('J', 'K', 'XC'):
            seconds = self._fock_pending.get(phase)
            columns.append(f'{phase} ' + (f'{seconds:9.4f} s' if seconds is not None
                                          else f"{'---':>9}  "))
        print(f"[J/K/XC TIME] build {self._fock_builds:3d}: {' | '.join(columns)}",
              flush=True)
        self._fock_pending.clear()

    def time_scf(self, mf):
        '''Record each mf.kernel wall time as 'SCF Total' via pyscf's post_kernel hook.'''
        mf.post_kernel = self._record_scf
        mf._keys = mf._keys | {'post_kernel'}  # an intended override: no sanity warning

    def _record_scf(self, envs):
        mf = envs['mf']
        type(mf).post_kernel(mf, envs)
        if 'cput0' in envs:  # the SCF kernel's own start clock, (cpu, wall)
            start, end = envs['cput0'][1], time.perf_counter()
            self._record('SCF Total', end - start, start)
            self.scf_runs.append((start, end, mf.cycles))

    # Report hierarchy: (label, [children]); a parent's residual prints as [unaccounted].
    _TREE_SETUP = ('PAW Setup', [
        ('Molecule and Box', []),
        ('Basis Decontraction', []),
        ('Uniform Grid', []),
        ('Coulomb Kernel', []),
        ('Neighbor List', []),
        ('Alpha0', []),
        ('ISDF Grid', []),
        ('Compensating Charge', []),
        ('Local Functions', []),
        ('AO Evaluation', []),
        ('Nuclear-Electron Split', []),
        ('Multigrid Init', []),
    ])
    _TREE_BUILDS = [
        ('J Total', [
            ('J Smooth PW', []),
            ('J Smooth PW (from XC)', []),
            ('J Sharp Local', []),
            ('J Smooth Local', []),
        ]),
        ('K Total', [
            ('K Smooth ISDF', []),
            ('K Sharp Local', []),
            ('K Smooth Local', []),
        ]),
        ('XC Total', [
            ('XC Smooth Uniform', []),
            ('XC Sharp Atomic', []),
            ('XC Smooth Atomic', []),
        ]),
    ]
    _TREE_SCF = ('SCF Total', _TREE_BUILDS)

    def _split_builds_by_scf(self):
        '''The J, K and XC timers as two {label: seconds} views: calls that started inside
        a timed mf.kernel, and the rest (e.g. a get_jk after the SCF).'''
        def in_scf(start):
            return any(scf_start <= start < scf_end
                       for scf_start, scf_end, _ in self.scf_runs)

        inside, outside = {}, {}
        labels = [label for root, children in self._TREE_BUILDS
                  for label in (root, *(child for child, _ in children))]
        for label in labels:
            starts = self.timing_starts.get(label, ())
            for seconds, start in zip(self.timings.get(label, ()), starts):
                view = inside if in_scf(start) else outside
                view.setdefault(label, []).append(seconds)
        return inside, outside

    _REPORT_W = 110  # table width
    _LABEL_W = 34    # label column width

    def print_timing_summary(self):
        '''Print every timer (calls, total, mean, first call, mean of the rest), then the
        breakdown: setup, J + K + XC + other inside the SCF, and J/K/XC outside it.'''
        timings, W, LW = self.timings, self._REPORT_W, self._LABEL_W
        rule = f"  {'-' * (W - 2)}"
        inside, outside = self._split_builds_by_scf()
        if 'SCF Total' in timings:
            inside['SCF Total'] = timings['SCF Total']

        def seconds(view, label):
            return sum(view.get(label, ()))

        def row(depth, label, total, calls, parent_total, first=None, rest=None):
            name = ('  ' * depth + '-> ' if depth > 0 else '') + label
            pct = total / parent_total * 100.0 if parent_total > 0 else 0.0
            first = f'{first:9.4f}s' if first is not None else ' ' * 10
            rest = f'{rest:9.4f}s' if rest is not None else ' ' * 10
            print(f"  {name:<{LW}} | {calls:5d} | {total:10.4f}s | "
                  f"{total / calls:9.4f}s | {first} | {rest} | ({pct:6.2f}%)")

        def emit(view, node, depth, parent_total):
            label, children = node
            calls = view[label]
            total = sum(calls)
            rest = (total - calls[0]) / (len(calls) - 1) if len(calls) > 1 else None
            row(depth, label, total, len(calls), parent_total, calls[0], rest)
            recorded = [child for child in children if child[0] in view]
            for child in recorded:
                emit(view, child, depth + 1, total)
            if recorded:
                resid = total - sum(seconds(view, child) for child, _ in recorded)
                row(depth + 1, label + ' [unaccounted]', resid, len(calls), total)

        sections = [
            ('SETUP', timings, [self._TREE_SETUP, ('XC Setup', [])]),
            ('SCF', inside, [self._TREE_SCF]),
            ('J/K/XC OUTSIDE THE SCF', outside, self._TREE_BUILDS),
        ]

        print('\n' + '=' * W)
        print('PAW TIMING SUMMARY'.center(W))
        print('=' * W)
        print(f"\n  {'Section':<{LW}} | {'Calls':>5} | {'Total':>11} | "
              f"{'Avg/Call':>10} | {'First':>10} | {'Avg Rest':>10} | {'% Parent':>9}")
        print(rule)
        for title, view, roots in sections:
            roots = [root for root in roots if root[0] in view]
            if not roots:
                continue
            print(f'\n  [{title}]')
            for root in roots:
                emit(view, root, 0, seconds(view, root[0]))
            print(rule)
        if 'J Smooth PW (from XC)' in timings:
            print('  J Smooth PW (from XC): reused from the XC pass; the cost is in '
                  'XC Smooth Uniform')

        phases = ('J', 'K', 'XC')
        setup = seconds(timings, 'PAW Setup') + seconds(timings, 'XC Setup')
        components = [('Setup', setup)]
        if self.scf_runs:
            builds = [(phase, seconds(inside, f'{phase} Total')) for phase in phases]
            other = seconds(timings, 'SCF Total') - sum(t for _, t in builds)
            components += builds + [('SCF other (diag, DIIS, hcore)', other)]
            outside_total = sum(seconds(outside, f'{phase} Total') for phase in phases)
            if outside_total:
                components.append(('J/K/XC outside the SCF', outside_total))
        else:
            components += [(phase, seconds(outside, f'{phase} Total'))
                           for phase in phases]
        grand = sum(t for _, t in components)
        print('\n  [BREAKDOWN]  wall-clock seconds; the SCF is J + K + XC + SCF other')
        print(rule)
        for name, t in components:
            pct = t / grand * 100.0 if grand > 0 else 0.0
            print(f'  {name:<{LW}} | {t:10.4f}s | ({pct:6.2f}%)')
        print(rule)
        print(f"  {'TOTAL':<{LW}} | {grand:10.4f}s |")
        if self.scf_runs:
            cycles = ', '.join(str(n) for _, _, n in self.scf_runs)
            print(f'  SCF cycles per kernel: {cycles}')
        print('=' * W + '\n', flush=True)

    def __del__(self):
        '''Print the timing summary on garbage collection, unless a driver already printed
        it and cleared timings.'''
        try:
            if self.timings:
                self.print_timing_summary()
        except Exception:
            pass
