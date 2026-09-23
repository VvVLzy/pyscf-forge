'''The ISDF exchange path: everything in PAW that faces the external Gausslets repo.

Kept in its own module so the rest of pyscf/paw stays import-clean with respect to
Gausslets. The import of ISDFGrid happens INSIDE build_isdf_grid, so the dependency
is only needed by a run that actually asks for K.

WHO OWNS THE GEOMETRY
---------------------
PAW does, and that is a reversal. Historically ISDFGrid sized the box, TRANSLATED the
molecule into it, and PAW was then built from grid.shifted_mol() / grid.shifted_cell()
(see Examples/isdf_paw_molecule.py in the Gausslets repo). That forced alpha0 to come
off the grid (`alpha0=grid.maxgto`), because the grid had to exist before PAW did.

Here PAW runs prepareMolForPAW first, determines alpha0 from its own basis and
accuracy targets, and only then builds the grid on the geometry it already has. Two
things make that work:

  * Passing `L` EXPLICITLY suppresses ISDFGrid._auto_box, and with it the translation
    (isdfgridv4.py: "Passing L explicitly keeps the historical behaviour exactly: no
    translation, and the caller is responsible for placing the molecule inside the
    box"). prepareMolForPAW already centres the molecule in [0, L), and ISDF's grid
    origin is hard-wired at zero, so the two conventions agree.
  * ISDF still rounds L up per axis to a whole number of block_size*d cells, so its
    box is >= PAW's and gains a little vacuum at the upper faces. Nothing moves; the
    atoms keep their coordinates and stay inside. Each side then runs its own Poisson
    solve on its own box, which is fine because they are separate solves.

Nothing in ISDFGrid checks that the geometry it was handed agrees with PAW's, and
box_basis_screening CLIPS an out-of-box atom onto the boundary cells rather than
raising. assert_geometry_handoff below is therefore not a nicety.

TWO GRIDS, ONE DENSITY
----------------------
The J build, get_nuc and vxc want the compensating-charge shape functions on the
uniform FFT grid; the exchange build wants them on the ISDF pivots only, because
pawexchange.py reads gridIdx[a] as offsets into the cell-major concatenated pivot
array. So PAWdata comes in two tagged flavours -- see paw_helper.PAWDataTuple -- and
this module is the 'pivot' side of that.

They remain two different POINT SETS, but they are no longer two different
RESOLUTIONS. ISDF's grid spacing `d` is derived from PAW's FFT mesh by
paw_grid_spacing() and is not a knob: ke_cutoff is the single control over grid
density for the whole program, and passing `d` in isdf_args is an error. Every OTHER
ISDFGrid knob is forwarded verbatim through isdf_args.

That is a correctness requirement, not tidiness. calculate_alpha0_dict picks alpha0
as the sharpest exponent PAW's mesh can represent, so PAW's grid resolves its own
compensating charge by construction and alpha0 CLIMBS WITH ke_cutoff. When d was
independent it stayed put while alpha0 moved, and past the crossover the pivots could
no longer resolve the shape functions -- gOnR on the pivots went quietly wrong.
Measured on CH4/cc-pVDZ, 14 bohr, isdf_tol=1e-7, error in E_K:

    ke_cutoff   alpha0   d_isdf   d_paw   d/d_paw     dE_K
          400    36.19    0.182   0.109     1.68   +20.95 mHa
          400    36.19    0.111   0.109     1.02    +0.01 mHa

The error tracked d/d_paw, not ke_cutoff, and went bad exactly as that ratio crossed
1. Tying the two together removes the ratio from the problem.
'''

import numpy

from pyscf import gto
from pyscf.lib import logger

from .paw_helper import require_grid_kind
from .paw_jk import _dm_channel


class _MfShim:
    '''The two attributes ISDFGrid actually reads off a mean-field object.

    It takes `mf` only for `mf.mol` and, as a fallback for an AO count it also gets
    from dm.shape, `mf.mo_coeff`. It never runs the SCF and never touches the
    geometry through it (isdfgridv4._auto_box: "self.mf is deliberately NOT
    touched"). Handing it a real RHF object here would build an SCF we never use, so
    a shim it is -- and `mo_coeff = None` routes the AO count to mol.nao, which is
    the contracted count the density matrix and localIdx both index.
    '''
    __slots__ = ('mol', 'mo_coeff')

    def __init__(self, mol):
        self.mol = mol
        self.mo_coeff = None


def isdf_mol_from_cell(cell):
    '''A plain gto.Mole, in BOHR, on exactly the coordinates PAW is using.

    Three reasons this is not just `cell`:

      * Units. ISDFGrid reads `mol.unit` to set its internal conversion factor, and
        interprets `L` and `d` in that unit. PAW's cell comes out of
        prepareMolForPAW as unit='A'. Handing over a Bohr Mole makes the factor 1
        and takes the conversion out of the picture entirely.
      * Evaluator. ISDFGrid switches to pbc_eval_gto for a pbc Cell. For a molecular
        run we want the open-boundary evaluator, matching the truncated
        (non-periodic) Coulomb kernel the grid actually uses.
      * Basis. The CONTRACTED basis, so mol.nao equals cell.nao -- which is what the
        density matrix, localIdx and the returned K are all indexed by. pmol (the
        decontracted one) would silently be a different size.

    Atom order is preserved, which matters: every per-atom list in PAWdata is
    indexed by position, not by symbol.
    '''
    atoms = [(cell.atom_symbol(i), tuple(cell.atom_coord(i)))    # atom_coord is Bohr
             for i in range(cell.natm)]
    return gto.M(atom=atoms, basis=cell.basis, unit='Bohr',
                 spin=None, charge=cell.charge,
                 verbose=0, dump_input=False)


def assert_geometry_handoff(grid, cell, log_obj=None):
    '''Check that the grid really is on PAW's geometry, before any SCF runs.

    Every one of these failures is otherwise silent: a wrong-by-a-translation grid
    puts each atom-local correction on the wrong nucleus and still converges.
    '''
    if getattr(grid, 'box_auto', False):
        raise ValueError(
            'ISDFGrid sized its own box and translated the molecule (box_auto=True). '
            'PAW owns the geometry here, so L must be passed explicitly.')

    paw_coords = cell.atom_coords()                  # Bohr
    grid_coords = numpy.asarray(grid.shifted_coords())

    if grid_coords.shape != paw_coords.shape:
        raise ValueError(f'atom count mismatch: grid has {grid_coords.shape[0]}, '
                         f'PAW cell has {paw_coords.shape[0]}')

    dev = numpy.abs(grid_coords - paw_coords).max()
    if dev > 1e-12:
        worst = int(numpy.argmax(numpy.abs(grid_coords - paw_coords).max(axis=1)))
        raise ValueError(
            f'grid and PAW disagree about the geometry by {dev:.3e} Bohr '
            f'(worst: atom {worst}). Every atom-local PAW correction would land on '
            f'the wrong nucleus, silently.')

    # Element ordering, not just positions: the per-atom lists in PAWdata are
    # positional, so a permutation with coincidentally-equal coordinates still breaks.
    paw_sym = [cell.atom_pure_symbol(i) for i in range(cell.natm)]
    grid_sym = [grid.mol.atom_pure_symbol(i) for i in range(grid.mol.natm)]
    if paw_sym != grid_sym:
        raise ValueError(f'atom ORDER differs: PAW {paw_sym} vs grid {grid_sym}')

    L_grid = numpy.asarray(grid.L, dtype=float)
    L_paw = numpy.asarray(cell.lattice_vectors().diagonal(), dtype=float)
    if numpy.any(L_grid < L_paw - 1e-10):
        raise ValueError(f'grid box {L_grid} is smaller than PAW\'s {L_paw}; '
                         'rounding to whole cells should only ever grow it.')

    # The grid spans [0, L) with the origin at zero and CLIPS anything outside onto
    # the boundary cells instead of complaining, so this is the check that matters.
    if numpy.any(paw_coords < 0.0) or numpy.any(paw_coords >= L_grid):
        bad = numpy.where((paw_coords < 0.0) | (paw_coords >= L_grid))[0]
        raise ValueError(
            f'atoms {sorted(set(bad.tolist()))} lie outside the grid box [0, {L_grid}). '
            'box_basis_screening would clip them onto the boundary cells without error.')

    # Deliberately NOT checked here: whether an augmentation sphere pokes out of the
    # box. The check above already guarantees grid.L >= PAW's L, and PAW's own Rgrid
    # spans only [0, L_paw), so PAW's box is always the binding one -- a sphere that
    # fits for the J build fits here with room to spare. Adding the test would need
    # Rs, which does not exist until mergeCompensatingCharge has run, i.e. after this.
    if log_obj is not None:
        logger.info(log_obj, 'ISDF box     : %s Bohr (PAW box %s, rounded up to cells)',
                    numpy.array2string(L_grid, precision=3),
                    numpy.array2string(L_paw, precision=3))
        logger.info(log_obj, 'geometry handoff verified: max deviation %.2e Bohr', dev)


def validate_k_pawdata(PAWdata_K, n_pivots):
    '''Shape and index-space checks on the pivot PAWdata, before the first K build.

    pawexchange.py checks that its pivot-row count matches gOnR's, but nothing
    checks the OTHER axis (the compensating-charge channel count q_a) or that
    gridIdx actually indexes the pivot array. Left alone, the first shows up as an
    opaque GEMM error deep in the exchange and the second as a wrong answer.
    '''
    if getattr(PAWdata_K, 'grid_kind', None) != 'pivot':
        raise ValueError("the exchange build needs a 'pivot'-tagged PAWdata; got "
                         f"{getattr(PAWdata_K, 'grid_kind', 'untagged')!r}")

    M_PQLarr, gridIdx, gOnR = PAWdata_K[4], PAWdata_K[7], PAWdata_K[8]
    for a, (gi, g, m) in enumerate(zip(gridIdx, gOnR, M_PQLarr)):
        gi = numpy.asarray(gi)
        expected = (len(gi), m.shape[2])
        if tuple(g.shape) != expected:
            raise ValueError(f'atom {a}: gOnR has shape {tuple(g.shape)}, expected '
                             f'{expected} = (pivots in sphere, q from M_PQLarr)')
        if gi.size and (gi.min() < 0 or gi.max() >= n_pivots):
            raise ValueError(f'atom {a}: gridIdx runs [{gi.min()}, {gi.max()}] but '
                             f'there are only {n_pivots} pivots -- this gridIdx is '
                             'indexing the uniform grid, not the pivots.')


# The three ISDFGrid arguments PAW determines itself, and why. Everything else in
# ISDFGrid.__init__ is the caller's to set through isdf_args.
PAW_OWNED = {
    'L': 'L comes from the cell PAW built',
    'maxgto': "maxgto is PAW's alpha0, per element",
    'd': "d is PAW's own grid spacing, set by ke_cutoff -- one grid density for J "
         "and K. Change ke_cutoff to change it",
}


def _check_isdf_args(grid_cls, isdf_args):
    """Validate isdf_args against ISDFGrid.__init__ and return it as a plain dict.

    ISDFGrid takes a couple of dozen accuracy and algorithm knobs and they are all
    meant to stay reachable, so this is a pass-through, not a whitelist -- the
    signature is read at call time rather than copied here, and a knob added to
    ISDFGrid works through PAW with no change on this side.

    Two things it does catch. A PAW-owned argument is rejected with the reason,
    because silently honouring `d` would break the single grid density and silently
    honouring `L` or `maxgto` would put the grid on a different geometry or a
    different basis split than the augmentation. And a misspelling is reported
    against the real signature: ISDFGrid has no **kwargs, so it would raise
    TypeError on its own, but by then the message is about a constructor the caller
    did not call and does not list the alternatives.
    """
    opts = dict(isdf_args or {})
    for name, why in PAW_OWNED.items():
        if name in opts:
            raise ValueError(f'isdf_args may not set {name!r}: PAW owns it ({why}).')

    import inspect
    params = inspect.signature(grid_cls.__init__).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        allowed = {n for n in params if n not in ('self', 'mf')} - set(PAW_OWNED)
        unknown = sorted(set(opts) - allowed)
        if unknown:
            raise ValueError(
                '%s is not an %s argument; available: %s'
                % (', '.join(repr(u) for u in unknown), grid_cls.__name__,
                   ', '.join(sorted(allowed))))
    return opts


def paw_grid_spacing(cell, mesh):
    """PAW's own uniform grid spacing in Bohr -- the one grid density.

    The MINIMUM over axes, not the geometric mean. For the cubic boxes PAW builds
    they are the same number, but on a non-cubic one the minimum is the only choice
    that leaves ISDF at least as fine as PAW on every axis; ISDF's d is a single
    scalar and its cells must be cubic, so it cannot follow a per-axis mesh.

    Taken from the mesh PAW actually uses rather than from the analytic
    pi/sqrt(2*ke_cutoff), because cutoff_to_mesh rounds up to a convenient FFT size
    and the real grid is a little finer than the formula (0.2979 vs 0.3142 bohr at
    ke_cutoff=50). The grid PAW runs on is the one ISDF has to match.
    """
    L = numpy.asarray(cell.lattice_vectors().diagonal(), dtype=float)
    return float(numpy.min(L / numpy.asarray(mesh, dtype=float)))


def _seed_box_for_cells(grid_cls, L, d, isdf_args):
    """The L to REQUEST so ISDFGrid's own rounding lands on cubic cells.

    ISDFGrid indexes its far-field eps table by integer cell displacement and so
    refuses a non-cubic cell lattice (_resolve_fmm_table). Getting one is fiddlier
    than it looks, because _round_L_to_cells only rewrites an axis that needs it:

        nx = ceil(L_i / d)
        if nx % block_size != 0:              # <-- only then
            L_i = ceil(nx/block_size) * block_size * d

    Two ways that bites, both of which this function exists to avoid:

    * An axis whose ceil(L_i/d) is ALREADY a multiple of block_size keeps its
      requested L_i, so its spacing is L_i/nx rather than d, while a neighbouring
      axis that was rewritten sits at exactly d. Cells differ between axes by a
      fraction of a percent and the cubic check fails. Invisible with a scalar L --
      every axis then takes the same branch -- which is why it only appeared once
      PAW started handing over a genuinely per-axis box.
    * Requesting exactly n*block_size*d does NOT avoid that: (n*bs*d)/d evaluates a
      hair ABOVE n*bs in floating point, ceil bumps it to n*bs+1, and the axis is
      rounded up by a whole extra block. Measured: a 17.56 bohr request came back
      23.41.

    So rather than trying to hit the boundary, aim at the MIDDLE of the target cell
    block. ceil(L_i/d) is then comfortably not a multiple of block_size, every axis
    takes the rewrite branch, and the final L_i is ISDFGrid's own new_nx * d -- the
    same product generate_grid later divides by, so the spacing is exactly d on all
    three axes with no floating-point coincidence required.

    The seed is below the true box, but that is harmless: the rewrite rounds it back
    UP to n*block_size*d >= the box PAW placed the molecule in, and
    assert_geometry_handoff checks containment afterwards regardless.
    """
    import inspect
    params = inspect.signature(grid_cls.__init__).parameters
    Nmax = isdf_args.get('Nmax', params['Nmax'].default)
    block_size = int(numpy.floor(float(Nmax) ** (1. / 3.)))
    if block_size <= 1:
        return numpy.asarray(L, dtype=float)    # every nx is a multiple; nothing to do
    pitch = block_size * d
    n_cells = numpy.maximum(1, numpy.ceil(numpy.asarray(L, dtype=float) / pitch - 1e-12))
    return (n_cells - 0.5) * pitch, n_cells * pitch


def build_isdf_grid(cell, alpha0, gmolSph, mesh, isdf_args=None, log_obj=None):
    '''Build an ISDFGrid on PAW's geometry and return (grid, pivot coordinates).

    `alpha0` is passed straight through as ISDFGrid's `maxgto`, dict and all: the
    grid's smooth/sharp split and PAW's augmentation have to agree shell by shell
    about which primitives each owns, so collapsing a per-element dict to one number
    here would be exactly the disagreement both docstrings warn about.

    `gmolSph` goes to setup_isdf as `extrafuns`, which is what makes the pivot
    selection see the compensating charge and not just the AO products. It must be
    the SAME object mergeCompensatingCharge later builds gOnR from.

    `mesh` is PAW's uniform FFT mesh, and it SETS the ISDF grid spacing -- see
    paw_grid_spacing and the module docstring.

    `isdf_args` is forwarded verbatim to ISDFGrid, so every accuracy and algorithm
    knob it has stays reachable (isdf_tol, Nmax, ao_thresh, screen_error,
    matrix_free, the fmm_* family, ...). The exceptions are the three PAW determines
    itself -- see PAW_OWNED -- which are rejected rather than overridden.
    '''
    # Deliberately inside the function: this is the only hard dependency on the
    # external Gausslets repo in all of pyscf/paw, and a J-only run must not need it.
    from Gausslets.isdfgridv4 import ISDFGrid

    opts = _check_isdf_args(ISDFGrid, isdf_args)

    mol = isdf_mol_from_cell(cell)
    L_paw = numpy.asarray(cell.lattice_vectors().diagonal(), dtype=float)   # Bohr
    d = paw_grid_spacing(cell, mesh)
    L_seed, L_want = _seed_box_for_cells(ISDFGrid, L_paw, d, opts)

    grid = ISDFGrid(_MfShim(mol), L=L_seed, maxgto=alpha0, d=d, **opts)
    assert_geometry_handoff(grid, cell, log_obj=log_obj)

    # ISDFGrid should have rounded the seed back up to the box we actually wanted,
    # with cells cubic to the last bit. Checked here because the alternative is a
    # RuntimeError from inside _resolve_fmm_table that names neither PAW nor the
    # box it objects to, several hundred lines into setup.
    if not numpy.allclose(grid.L, L_want, rtol=0, atol=1e-9):
        raise RuntimeError(f'ISDFGrid rounded the box to {grid.L}, expected '
                           f'{L_want}; _seed_box_for_cells is wrong')
    n_pts = numpy.round(numpy.asarray(grid.L, dtype=float) / grid.d)
    sides = numpy.asarray(grid.L, dtype=float) / n_pts
    if sides.max() - sides.min() > 1e-12 * sides.mean():
        raise RuntimeError(f'ISDF cells are not cubic: per-axis spacing {sides}')

    # ISDFGrid rounds L up to a whole number of block_size*d cells, so its box grows
    # when d is coarse. Harmless -- the atoms do not move and the assertion above
    # checked they are still inside -- but worth showing, because a coarse ke_cutoff
    # can inflate it noticeably (14 -> 17.9 bohr at ke_cutoff=50, block_size=20).
    if log_obj is not None:
        logger.info(log_obj,
                    'ISDF spacing : %.4f Bohr, from PAW mesh %s (one grid density; '
                    'change it with ke_cutoff)', d, numpy.asarray(mesh).tolist())
        logger.info(log_obj, 'ISDF box     : %s Bohr, rounded up from PAW\'s %s',
                    numpy.array2string(numpy.asarray(grid.L), precision=3),
                    numpy.array2string(L_paw, precision=3))

    grid.setup_isdf(gmol=gmolSph)
    pivots = grid.get_sparse_grid()
    if pivots is None:
        raise RuntimeError('setup_isdf produced no pivots')

    if log_obj is not None:
        logger.info(log_obj, 'ISDF pivots  : %10d (of %d uniform grid points)',
                    len(pivots), grid.isdf_res.get('total_grid_points', -1))
        # (cell, AO) pairs where an AO has SHARP support but no smooth support there.
        # Nonzero means some shell has all its primitives above alpha0, so it is zeroed
        # out of smooth_mol entirely -- the exchange then carries a wider AO set per
        # cell for the screen. Correct either way, but it costs a little, and until
        # 2026-09-22 it raised outright, so it is worth saying out loud rather than
        # leaving buried in ISDFGrid's own stdout.
        n_out = int(grid.isdf_res.get('n_sharp_outside_smooth', 0) or 0)
        if n_out:
            logger.info(log_obj,
                        'sharp cover  : %d (cell, AO) entries lie outside the smooth '
                        'cover -- some shell is entirely above alpha0, so the exchange '
                        'screens on a wider AO set in those cells', n_out)
    return grid, numpy.asarray(pivots)


def getkSmoothISDF(cell, dm, gaussgrid, PAWdata_K, Periodic=False, spin=0,
                   dm_screen=None):
    """The smooth (grid) part of K, plus the grid-mediated augmentation coupling.

    This is the whole ISDF exchange engine: compute_paw_exchange runs terms 1, 2/3
    and 4 -- the smooth-smooth pivot exchange and both compensating-charge couplings
    -- and returns a symmetrised (nao, nao) matrix. The purely atom-local remainder
    is paw_jk.getkSharpLocal + getkSmoothLocal.

    Port of pawisdf's PAWutils.getkSmoothISDF, with the restricted-only
    `dm[0,0]/2 ... *2` replaced by _dm_channel so one spin channel of a UKS density
    matrix works too.

    `dm_screen` is the incremental path. None means screen on the matrix being built
    from (the full build). Pass it and `dm` is a density DIFFERENCE consumed by the
    arithmetic, while `dm_screen` is used only by Screen 1 and Screen 2 -- the screens
    estimate an ENERGY, quadratic in the density, where K is linear, so a difference
    in both slots would be judged by O(dD^2) and prune pairs that still matter.

    Both go through _dm_channel: it halves a restricted dm, and halving only one
    would move the screen's operating point by a factor of two.
    """
    if Periodic:
        raise NotImplementedError(
            'ISDF exchange is gamma-point/molecular only: the grid uses a truncated, '
            'non-periodic Coulomb kernel.')
    # compute_paw_exchange indexes gridIdx/gOnR as offsets into the pivot array, so a
    # 'uniform' tuple here would evaluate the shape functions at uniform-grid points
    # and index them as pivots -- in range, wrong points, no error. validate_k_pawdata
    # covers the tuple PAW built at setup; this covers every call site thereafter.
    require_grid_kind(PAWdata_K, 'pivot', 'getkSmoothISDF')
    dm_k, scale = _dm_channel(dm, spin)
    if dm_screen is None:
        K = gaussgrid.compute_paw_exchange(dm_k, PAWdata_K)
    else:
        dm_s, _ = _dm_channel(dm_screen, spin)
        K = gaussgrid.compute_paw_exchange(dm_k, PAWdata_K, dm_screen=dm_s)
    return K * scale
