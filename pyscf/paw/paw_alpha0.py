'''PAW's alpha0 per basis label: the least predicted worst-atom error over the gaps between its exponents,
    E_a = C_GRID E_miss + q_a^2 aliasing + C_SAME P_same + C_UNLIKE P_unlike + C_FIT M0      (Hartree).
Derivations and calibration: Gausslets, Tuning/alpha0_rules/proposal/alpha0_final.pdf.
'''
import functools

import numpy
import scipy.integrate
import scipy.optimize
import scipy.special
from scipy.spatial import cKDTree

from pyscf import gto, scf
from pyscf.dft import LebedevGrid
from pyscf.gto.eval_gto import eval_gto
from pyscf.lib import logger

# Hartree per unit of each term, from one joint log10 fit to 1526 W4-11 PAW errors (2026-10-30).
C_GRID, C_SAME, C_UNLIKE, C_FIT = 1.7842369311117017, 3.2654181227294314, 16.359812632154743, 0.37812976204764326
HALF_BOND = 32.4             # Bohr^2: an unlike neighbour at distance d is given alpha0 = HALF_BOND / d^2
POPULATION_FLOOR = 0.01      # electrons per AO, so AOs empty in the atom (polarization, virtuals) still count
MAX_SPHERE = 5.0             # Bohr: the largest augmentation sphere allowed
NUCLEUS_LEAK = 1e-3          # largest share of a compensating charge allowed beyond the nearest nucleus
NEIGHBOUR_CUTOFF = 8.0       # Bohr: partners in the overlap and fit terms
BELOW_EXPONENT = 1 - 1e-6    # a gap's top sits just under its exponent; PAW's code paths disagree at equality
INTERIOR_POINTS, FINE_POINTS = 8, 24

# Fit term: Gauss-Legendre out to exp(-FIT_EDGE) of the softest sharp primitive, times a Lebedev sphere.
FIT_RADIAL, FIT_ANGULAR, FIT_EDGE = 40, 302, 30.0
# Missed density: radial and Lebedev grid for the spherical average, then G points beyond the mesh cutoff.
MISS_RMAX, MISS_RADIAL, MISS_ANGULAR, MISS_G_POINTS, MISS_G_SPAN = 12.0, 400, 26, 1500, 60.0

# A unit Gaussian sampled on the mesh has self-energy error sqrt(alpha0) F(ke / alpha0), worst over its
# position in a mesh cell; F tabulated at ke / alpha0 = 2, 2.25, ..., 20.
ALIASING_X = numpy.arange(2.0, 20.001, 0.25)
ALIASING_LOG_F = numpy.log([
    0.2091438, 0.1368116, 0.09139476, 0.06204592, 0.04266466, 0.02964793, 0.02078641, 0.01468562, 0.01044534,
    0.007473789, 0.005376194, 0.003885924, 0.00282098, 0.002055974, 0.001503811, 0.001103545, 0.0008122417,
    0.0005994724, 0.0004435508, 0.0003289419, 0.0002444652, 0.0001820392, 0.0001357995, 0.0001014749, 7.594402e-05,
    5.691872e-05, 4.271695e-05, 3.209887e-05, 2.414839e-05, 1.818709e-05, 1.371152e-05, 1.034733e-05, 7.815681e-06,
    5.90852e-06, 4.470368e-06, 3.384868e-06, 2.564817e-06, 1.944783e-06, 1.475608e-06, 1.120321e-06, 8.510842e-07,
    6.469179e-07, 4.919958e-07, 3.743681e-07, 2.850045e-07, 2.170757e-07, 1.654126e-07, 1.261002e-07, 9.617134e-08,
    7.337548e-08, 5.600478e-08, 4.276236e-08, 3.266292e-08, 2.49574e-08, 1.907609e-08, 1.458545e-08, 1.115544e-08,
    8.534629e-09, 6.531453e-09, 4.999861e-09, 3.82847e-09, 2.932296e-09, 2.246479e-09, 1.721493e-09, 1.319513e-09,
    1.011635e-09, 7.757709e-10, 5.950305e-10, 4.564981e-10, 3.502917e-10, 2.688491e-10, 2.063831e-10, 1.584613e-10])


def choose_alpha0(mol, ke_cutoff, periodic=False, fixed=None, orbital_cutoff=1e-8):
    '''({label: alpha0}, expected error per atom in Ha). fixed: one alpha0 for every label, or
    {label: alpha0} for some; those are kept and only their expected error is evaluated.'''
    model = _ErrorModel(mol, ke_cutoff, periodic)
    if fixed is not None and not isinstance(fixed, dict):
        fixed = dict.fromkeys(model.types, fixed)
    fixed = {label: a for label, a in (fixed or {}).items() if a is not None}
    sphere_floor = _sphere_floor(mol, orbital_cutoff)
    alpha0, expected_error = {}, numpy.zeros(mol.natm)
    for label in model.types:
        atoms = model.atoms_of[label]
        if label in fixed:
            alpha0[label] = fixed[label]
            expected_error[atoms] = model.errors_at(label, fixed[label])
        else:
            lowest = max(sphere_floor, model.nucleus_floor(label))
            alpha0[label], expected_error[atoms] = model.least_error(label, lowest)
    return alpha0, expected_error


def log_expected_error(mol, alpha0, expected_error):
    logger.note(mol, 'Expected PAW error: %.2e Ha   (alpha0 %s)', expected_error.sum(),
                ', '.join('%s %.4g' % item for item in alpha0.items()))
    for atom, error in enumerate(expected_error):
        logger.note(mol, '    atom %3d %-4s %.2e Ha', atom, mol.atom_symbol(atom), error)


def aliasing_error(alpha0, ke_cutoff):
    '''Self-energy error of a unit-charge Gaussian of exponent alpha0 sampled on the ke_cutoff mesh.'''
    x = ke_cutoff / alpha0
    if x <= ALIASING_X[-1]:
        log_f = numpy.interp(x, ALIASING_X, ALIASING_LOG_F)
    else:   # continue the table's last decay
        slope = (ALIASING_LOG_F[-1] - ALIASING_LOG_F[-2]) / (ALIASING_X[-1] - ALIASING_X[-2])
        log_f = ALIASING_LOG_F[-1] + slope * (x - ALIASING_X[-1])
    return numpy.sqrt(alpha0) * numpy.exp(log_f)


def _sphere_floor(mol, orbital_cutoff):
    '''alpha0 whose PAW sphere is MAX_SPHERE: get_gaussian_radius's N r^l exp(-alpha0 r^2) = orbital_cutoff,
    at the basis' largest l, solved for alpha0.'''
    l = mol._bas[:, gto.ANG_OF].max()
    shape = numpy.sqrt(2.0**l / scipy.special.factorial2(2 * l + 1) / numpy.sqrt(numpy.pi))
    excess = lambda a: (numpy.log(shape * (2 * a)**(l / 2 + 0.75)) + l * numpy.log(MAX_SPHERE)
                        - a * MAX_SPHERE**2 - numpy.log(orbital_cutoff))
    return scipy.optimize.brentq(excess, 1e-3, 1e4, xtol=1e-10)


def _candidate_gaps(exponents, lowest):
    '''Per gap between consecutive exponents (the last one above them all): geometric interior points
    and the gap's top, ascending; nothing below lowest.'''
    edges = [0.0, *exponents, numpy.inf]
    gaps = []
    for below, above in zip(edges[:-1], edges[1:]):
        top = above * BELOW_EXPONENT if numpy.isfinite(above) else 50 * exponents[-1]
        bottom = max(below, lowest)
        if top > bottom:
            gaps.append(numpy.unique([*numpy.geomspace(bottom, top, INTERIOR_POINTS + 2)[1:-1], top]))
    return gaps


def _neighbour_pairs(mol, periodic):
    '''(i, j, shift, distance) for every atom j, or its periodic image at j + shift, within
    NEIGHBOUR_CUTOFF of atom i.'''
    coords = mol.atom_coords()
    shifts = mol.get_lattice_Ls(rcut=NEIGHBOUR_CUTOFF) if periodic else numpy.zeros((1, 3))
    images = (shifts[:, None] + coords).reshape(-1, 3)
    found = cKDTree(coords).sparse_distance_matrix(cKDTree(images), NEIGHBOUR_CUTOFF, output_type='ndarray')
    found = found[found['v'] > 0]   # an atom is not its own neighbour
    found = found[numpy.argsort(found['i'], kind='stable')]
    shift_index, j = numpy.divmod(found['j'], mol.natm)
    return found['i'], j, shifts[shift_index], found['v']


@functools.lru_cache(maxsize=None)
def _miss_radial_grid():
    '''Gauss-Legendre points on [0, MISS_RMAX], their r^2 weights, and the Lebedev sphere.'''
    x, w = numpy.polynomial.legendre.leggauss(MISS_RADIAL)
    r = 0.5 * MISS_RMAX * (x + 1)
    return r, 0.5 * MISS_RMAX * w * r**2, LebedevGrid.MakeAngularGrid(MISS_ANGULAR)


@functools.lru_cache(maxsize=8)
def _missed_transform(ke_cutoff):
    '''G points beyond sqrt(2 ke), and the spherical Fourier transform to them, 4 pi j0(G r) r^2 dr.'''
    r, radial_weight, _ = _miss_radial_grid()
    g_max = numpy.sqrt(2 * ke_cutoff)
    g = numpy.linspace(g_max, g_max + MISS_G_SPAN, MISS_G_POINTS)
    return g, 4 * numpy.pi * numpy.sinc(numpy.outer(g, r) / numpy.pi) * radial_weight


class _AtomType:
    '''One basis label alone at the origin: its primitives, contraction and atomic-guess density.
    The error terms that depend only on the label and the gap are cached here.'''

    def __init__(self, mol, label, charge):
        atom = gto.M(atom=[(label, (0.0, 0.0, 0.0))], basis={label: mol._basis[label]}, unit='B',
                     spin=charge % 2, verbose=0)
        self.prim, self.ctr = atom.decontract_basis(aggregate=True)   # contracted AOs = prim AOs @ ctr
        self.prim_exponent = numpy.repeat(self.prim._env[self.prim._bas[:, gto.PTR_EXP]],
                                          2 * self.prim._bas[:, gto.ANG_OF] + 1)
        self.exponents = numpy.unique(self.prim_exponent)
        self.dm = scf.hf.init_guess_by_atom(atom)
        self.population = self.dm.diagonal()
        self.fit_weight = numpy.sqrt(numpy.maximum(self.population, POPULATION_FLOOR))
        self._sharp_charge, self._missed_energy, self._radial_density, self._fit_operator = {}, {}, {}, {}

    def gap_of(self, alpha0):
        '''Index of the gap holding alpha0 = the number of exponents below it.'''
        return int(numpy.searchsorted(self.exponents, alpha0))

    def is_sharp(self, gap):
        '''Per primitive AO: at or above the gap, so in the one-centre (sharp) part.'''
        if gap == len(self.exponents):
            return numpy.zeros(len(self.prim_exponent), bool)
        return self.prim_exponent >= self.exponents[gap]

    def sharp_charge(self, gap):
        '''q = Tr[D (S - S~)], the charge the compensating Gaussian carries.'''
        if gap not in self._sharp_charge:
            s_prim = self.prim.intor('int1e_ovlp')
            smooth_ctr = self.ctr * ~self.is_sharp(gap)[:, None]
            s_lost = self.ctr.T @ s_prim @ self.ctr - smooth_ctr.T @ s_prim @ smooth_ctr
            self._sharp_charge[gap] = float(numpy.sum(self.dm * s_lost))
        return self._sharp_charge[gap]

    def missed_energy(self, gap, ke_cutoff):
        '''E_miss = (1/pi) int_{G > sqrt(2 ke)} |rho~(G)|^2 dG, rho~ the smooth atomic density with
        every AO population raised to POPULATION_FLOOR.'''
        key = (gap, ke_cutoff)
        if key not in self._missed_energy:
            g, fourier = _missed_transform(ke_cutoff)
            density_g = fourier @ self._smooth_radial_density(gap)
            self._missed_energy[key] = float(scipy.integrate.trapezoid(density_g**2, g) / numpy.pi)
        return self._missed_energy[key]

    def _smooth_radial_density(self, gap):
        '''Spherical average of the floored smooth atomic density on the missed-energy radial grid.'''
        if gap not in self._radial_density:
            smooth = ~self.is_sharp(gap)
            smooth_ao = self._miss_prim_ao[:, smooth] @ self.ctr[smooth]
            floored_dm = self.dm + numpy.diag(numpy.maximum(POPULATION_FLOOR - self.population, 0))
            density = numpy.sum(smooth_ao @ floored_dm * smooth_ao, axis=1)
            self._radial_density[gap] = density.reshape(MISS_RADIAL, -1) @ _miss_radial_grid()[2][:, 3]
        return self._radial_density[gap]

    @functools.cached_property
    def _miss_prim_ao(self):
        r, _, sphere = _miss_radial_grid()
        return self.prim.eval_gto('GTOval_sph', (r[:, None, None] * sphere[None, :, :3]).reshape(-1, 3))

    def fit_operator(self, gap):
        '''(grid points, F): F @ (neighbour AOs on the points) = int phi_sharp (phi - P phi), with P the fit
        onto this atom's smooth primitives, least squares weighted by its sharp density.'''
        if gap not in self._fit_operator:
            r_max = numpy.sqrt(FIT_EDGE / self.exponents[gap])
            x, w = numpy.polynomial.legendre.leggauss(FIT_RADIAL)
            r = 0.5 * r_max * (x + 1)
            sphere = LebedevGrid.MakeAngularGrid(FIT_ANGULAR)
            points = (r[:, None, None] * sphere[None, :, :3]).reshape(-1, 3)
            weights = (0.5 * r_max * w * r**2)[:, None] * 4 * numpy.pi * sphere[None, :, 3]
            prim_ao = self.prim.eval_gto('GTOval_sph', points)
            sharp = self.is_sharp(gap)
            sharp_ao = prim_ao[:, sharp] @ self.ctr[sharp]
            weighted_sharp_ao = sharp_ao * weights.reshape(-1, 1)
            fit_basis = prim_ao[:, ~sharp]
            if not fit_basis.shape[1]:   # nothing smooth to fit with
                self._fit_operator[gap] = (points, weighted_sharp_ao.T)
                return self._fit_operator[gap]
            # least squares weighted by the sharp density, by SVD (rcond 1e-10) of the weighted fit basis
            root_weight = numpy.sqrt((weighted_sharp_ao * sharp_ao) @ self.population)
            u, s, vt = numpy.linalg.svd(fit_basis * root_weight[:, None], full_matrices=False)
            keep = s > 1e-10 * s[0]
            coefficients = (vt[keep].T / s[keep]) @ u[:, keep].T * root_weight   # per point value
            self._fit_operator[gap] = (points, weighted_sharp_ao.T - (weighted_sharp_ao.T @ fit_basis) @ coefficients)
        return self._fit_operator[gap]


class _ErrorModel:
    '''Per-atom error terms of a molecule or cell at trial alpha0 values, one basis label at a time.'''

    def __init__(self, mol, ke_cutoff, periodic):
        self.mol, self.ke_cutoff = mol, ke_cutoff
        self.coords = mol.atom_coords()
        labels = [mol.atom_symbol(i) for i in range(mol.natm)]
        self.labels = numpy.array(labels)
        self.atoms_of = {label: numpy.flatnonzero(self.labels == label) for label in dict.fromkeys(labels)}
        self.types = {label: _AtomType(mol, label, mol.atom_charge(atoms[0])) for label, atoms in self.atoms_of.items()}
        self.shell_range = mol.aoslice_by_atom()[:, :2]
        self.pair_i, self.pair_j, self.pair_shift, self.pair_distance = _neighbour_pairs(mol, periodic)
        self.pairs_start = numpy.searchsorted(self.pair_i, numpy.arange(mol.natm + 1))
        self._pairs = {}

    def nucleus_floor(self, label):
        '''Least alpha0 keeping all but NUCLEUS_LEAK of each compensating charge inside the nearest neighbour.'''
        distance = self.pair_distance[numpy.isin(self.pair_i, self.atoms_of[label])]
        if not len(distance):
            return 0.0
        x = scipy.optimize.brentq(lambda v: scipy.special.gammaincc(1.5, v) - NUCLEUS_LEAK, 1e-6, 500.0)
        return x / distance.min()**2

    def base_errors(self, label, alpha0):
        '''Per atom of the label: grid + aliasing + penetration overlap (everything but the fit term).'''
        atom_type, atoms = self.types[label], self.atoms_of[label]
        gap = atom_type.gap_of(alpha0)
        q = atom_type.sharp_charge(gap)
        grid = (C_GRID * atom_type.missed_energy(gap, self.ke_cutoff)
                + q**2 * aliasing_error(alpha0, self.ke_cutoff))
        local_atom, d, same, q_unlike = self._pairs_of(label)
        # same label: the neighbour shares alpha0; otherwise it gets the half-bond estimate
        beta = numpy.where(same, alpha0, HALF_BOND / d**2)
        q_partner = numpy.where(same, q, q_unlike)
        mu = alpha0 * beta / (alpha0 + beta)
        erfc = scipy.special.erfc
        penetration = q * q_partner * (2 * erfc(numpy.sqrt(mu) * d) - erfc(numpy.sqrt(beta) * d)
                                       - erfc(numpy.sqrt(alpha0) * d)) / d
        overlap = numpy.bincount(local_atom, weights=penetration * numpy.where(same, C_SAME, C_UNLIKE),
                                 minlength=len(atoms))
        return grid + overlap

    def _pairs_of(self, label):
        '''(index among the label's atoms, distance, same label?, unlike partner's sharp charge) per pair;
        an unlike partner's charge is at its half-bond alpha0, so it never changes.'''
        if label not in self._pairs:
            mine = numpy.isin(self.pair_i, self.atoms_of[label])
            d, partner = self.pair_distance[mine], self.labels[self.pair_j[mine]]
            same = partner == label
            q_unlike = numpy.array([0.0 if s else self.types[p].sharp_charge(self.types[p].gap_of(HALF_BOND / x**2))
                                    for s, p, x in zip(same, partner, d)])
            local_atom = numpy.searchsorted(self.atoms_of[label], self.pair_i[mine])
            self._pairs[label] = (local_atom, d, same, q_unlike)
        return self._pairs[label]

    def fit_charge(self, atom, gap):
        '''M0 = sum_{mu on atom, nu on neighbours} sqrt(D_mumu D_nunu) |int phi_mu^sharp (phi_nu - P phi_nu)|.'''
        atom_type = self.types[self.labels[atom]]
        if gap == len(atom_type.exponents):
            return 0.0
        points, fit_operator = atom_type.fit_operator(gap)
        neighbours = slice(self.pairs_start[atom], self.pairs_start[atom + 1])
        charge = 0.0
        for j, shift in zip(self.pair_j[neighbours], self.pair_shift[neighbours]):
            neighbour_ao = eval_gto(self.mol, 'GTOval_sph', points + self.coords[atom] - shift,
                                    shls_slice=tuple(self.shell_range[j]))
            charge += atom_type.fit_weight @ numpy.abs(fit_operator @ neighbour_ao) @ self.types[self.labels[j]].fit_weight
        return charge

    def errors_at(self, label, alpha0):
        gap = self.types[label].gap_of(alpha0)
        fit = [C_FIT * self.fit_charge(atom, gap) for atom in self.atoms_of[label]]
        return self.base_errors(label, alpha0) + fit

    def least_error(self, label, lowest):
        '''(alpha0, per-atom errors) minimising the worst atom over every gap's candidates, then over a fine
        scan of the winning gap. The fit term is constant in a gap, so it is skipped where it cannot win.'''
        atom_type, atoms = self.types[label], self.atoms_of[label]
        gaps = _candidate_gaps(atom_type.exponents, lowest)
        base = [numpy.array([self.base_errors(label, a) for a in candidates]) for candidates in gaps]
        lower_bound = [b.max(1).min() for b in base]
        best_error, best_gap, best_fit = numpy.inf, None, None
        for g in numpy.argsort(lower_bound):
            if lower_bound[g] >= best_error:
                break
            fit = self._fit_errors_below(atoms, atom_type.gap_of(gaps[g][-1]), base[g], best_error)
            if fit is not None and (base[g] + fit).max(1).min() < best_error:
                best_error, best_gap, best_fit = (base[g] + fit).max(1).min(), g, fit
        candidates = numpy.append(numpy.geomspace(gaps[best_gap][0], gaps[best_gap][-1], FINE_POINTS), gaps[best_gap])
        totals = numpy.array([self.base_errors(label, a) for a in candidates]) + best_fit
        best = numpy.argmin(totals.max(1))
        return float(candidates[best]), totals[best]

    def _fit_errors_below(self, atoms, gap, base, best_error):
        '''C_FIT M0 per atom, or None as soon as one atom alone puts the gap at or above best_error.'''
        fit = numpy.zeros(len(atoms))
        for k in numpy.argsort(-base.min(0)):
            fit[k] = C_FIT * self.fit_charge(atoms[k], gap)
            if (base[:, k] + fit[k]).min() >= best_error:
                return None
        return fit
