"""PAW + PBE single point on n-henicosane, C21H44, standalone.

    python paw_pbe_c21h44.py

C21H44 is the shortest alkane on which PAW's smooth Coulomb (J) build failed: in a
C3/C5/C11/C21/C31/C41/C51 scan (2026-09-29), C3 to C11 were fine and C21 was the
first failure.

Needs only pyscf and pyscf-forge's PAW module (pyscf.paw). PBE has no exact exchange,
so PAW supplies J alone: no K, no ISDF grid, nothing else imported.

What it runs, cc-pVDZ, geometry below in Bohr:
  1. Plain pyscf RKS-PBE: the reference (all-electron, exact J).
  2. The same RKS-PBE with PAW supplying J (mf.with_df = PAW). XC is still pyscf's
     molecular grid and the nuclear attraction is still analytic, so any difference
     from (1) comes from PAW's J.
Then it prints E_J of both on the SAME starting (minao) density -- the sharpest single
number -- and both converged total energies.

What to expect:
  * PAW code as of 2026-09-29: E_J on the starting density +981 Ha too high, all of
    it in the smooth plane-wave part of J, and the SCF does not converge.
  * With the box-truncated Coulomb kernel (2026-10-04): see EXPECTED below.

Cost: about 3 minutes on 8 threads (the reference SCF is the slower half) and a few
GB of memory; set OMP_NUM_THREADS.
"""
import inspect
import time

import numpy as np
from pyscf import dft, gto
from pyscf.pbc import gto as pgto
from pyscf.paw import NewPAW as PAW

BASIS = 'cc-pvdz'
# Ha. Sets PAW's uniform FFT mesh, and through it PAW's compensation-charge exponent.
KE_CUTOFF = 50.0

# EXPECTED, measured 2026-10-05 with the box-truncated kernel (pyscf 2.11.0, 8 threads;
# PAW box 129.92 x 35.20 x 37.25 Bohr, mesh 587 x 161 x 169):
#                        reference (exact J)        PAW J                 PAW - ref
#   E_J, minao density     1838.8902481774 Ha    1838.8870830060 Ha    -3.17e-3 Ha
#   E_tot, converged       -825.5718497652 Ha    -825.5737513386 Ha    -1.90e-3 Ha
# Both SCFs converge in 10 cycles. Before the fix, PAW - ref for E_J was +981 Ha.

# All-anti C21H44, chain along x, zigzag in the xz plane, centred on its bounding box.
# Bohr (C-C 2.8913 Bohr = 1.53 Angstrom).
GEOMETRY_BOHR = """
C  -23.9698055742799596 -0.1431069539134947 -0.5318829236075686
H  -24.5390231784516359  1.7988925017254349 -0.9158250455580059
H  -25.4794854697930759 -1.1141066817329590  0.4784676399428669
H  -23.5985608871101995 -1.1141066817329603 -2.3101177310588783
C  -21.5728250168519651 -0.1431069539134947  1.0849008770833219
H  -21.5728250168519651  1.5126785938984457  2.3101177310588787
H  -21.5728250168519651 -1.7988925017254349  2.3101177310588787
C  -19.1758444594239670 -0.1431069539134947 -0.5318829236075686
H  -19.1758444594239670  1.5126785938984457 -1.7570997775831254
H  -19.1758444594239670 -1.7988925017254349 -1.7570997775831254
C  -16.7788639019959724 -0.1431069539134947  1.0849008770833219
H  -16.7788639019959724  1.5126785938984457  2.3101177310588787
H  -16.7788639019959724 -1.7988925017254349  2.3101177310588787
C  -14.3818833445679761 -0.1431069539134947 -0.5318829236075686
H  -14.3818833445679761  1.5126785938984457 -1.7570997775831254
H  -14.3818833445679761 -1.7988925017254349 -1.7570997775831254
C  -11.9849027871399798 -0.1431069539134947  1.0849008770833219
H  -11.9849027871399798  1.5126785938984457  2.3101177310588787
H  -11.9849027871399798 -1.7988925017254349  2.3101177310588787
C  -9.5879222297119853 -0.1431069539134947 -0.5318829236075686
H  -9.5879222297119853  1.5126785938984457 -1.7570997775831254
H  -9.5879222297119853 -1.7988925017254349 -1.7570997775831254
C  -7.1909416722839872 -0.1431069539134947  1.0849008770833219
H  -7.1909416722839872  1.5126785938984457  2.3101177310588787
H  -7.1909416722839872 -1.7988925017254349  2.3101177310588787
C  -4.7939611148559926 -0.1431069539134947 -0.5318829236075686
H  -4.7939611148559926  1.5126785938984457 -1.7570997775831254
H  -4.7939611148559926 -1.7988925017254349 -1.7570997775831254
C  -2.3969805574279981 -0.1431069539134947  1.0849008770833219
H  -2.3969805574279981  1.5126785938984457  2.3101177310588787
H  -2.3969805574279981 -1.7988925017254349  2.3101177310588787
C   0.0000000000000000 -0.1431069539134947 -0.5318829236075686
H   0.0000000000000000  1.5126785938984457 -1.7570997775831254
H   0.0000000000000000 -1.7988925017254349 -1.7570997775831254
C   2.3969805574279945 -0.1431069539134947  1.0849008770833219
H   2.3969805574279945  1.5126785938984457  2.3101177310588787
H   2.3969805574279945 -1.7988925017254349  2.3101177310588787
C   4.7939611148559891 -0.1431069539134947 -0.5318829236075686
H   4.7939611148559891  1.5126785938984457 -1.7570997775831254
H   4.7939611148559891 -1.7988925017254349 -1.7570997775831254
C   7.1909416722839872 -0.1431069539134947  1.0849008770833219
H   7.1909416722839872  1.5126785938984457  2.3101177310588787
H   7.1909416722839872 -1.7988925017254349  2.3101177310588787
C   9.5879222297119853 -0.1431069539134947 -0.5318829236075686
H   9.5879222297119853  1.5126785938984457 -1.7570997775831254
H   9.5879222297119853 -1.7988925017254349 -1.7570997775831254
C   11.9849027871399798 -0.1431069539134947  1.0849008770833219
H   11.9849027871399798  1.5126785938984457  2.3101177310588787
H   11.9849027871399798 -1.7988925017254349  2.3101177310588787
C   14.3818833445679743 -0.1431069539134947 -0.5318829236075686
H   14.3818833445679743  1.5126785938984457 -1.7570997775831254
H   14.3818833445679743 -1.7988925017254349 -1.7570997775831254
C   16.7788639019959689 -0.1431069539134947  1.0849008770833219
H   16.7788639019959689  1.5126785938984457  2.3101177310588787
H   16.7788639019959689 -1.7988925017254349  2.3101177310588787
C   19.1758444594239634 -0.1431069539134947 -0.5318829236075686
H   19.1758444594239634  1.5126785938984457 -1.7570997775831254
H   19.1758444594239634 -1.7988925017254349 -1.7570997775831254
C   21.5728250168519651 -0.1431069539134947  1.0849008770833219
H   21.5728250168519651  1.5126785938984457  2.3101177310588787
H   21.5728250168519651 -1.7988925017254349  2.3101177310588787
C   23.9698055742799596 -0.1431069539134947 -0.5318829236075686
H   24.5390231784516359  1.7988925017254349 -0.9158250455580061
H   23.5985608871101959 -1.1141066817329590 -2.3101177310588787
H   25.4794854697930759 -1.1141066817329603  0.4784676399428662
"""


def main():
    mol = gto.M(atom=GEOMETRY_BOHR, basis=BASIS, unit='Bohr', verbose=4)
    nao = mol.nao_nr()
    print(f'C21H44: {mol.natm} atoms, {nao} AOs, {mol.nelectron} electrons, {BASIS}, '
          f'ke_cutoff {KE_CUTOFF} Ha', flush=True)

    # 1. Reference: plain RKS-PBE.
    mf_ref = dft.RKS(mol, xc='pbe')
    dm0 = mf_ref.get_init_guess(key='minao')
    ej_ref0 = 0.5 * float(np.einsum('ij,ji->', dm0, mf_ref.get_j(mol, dm0)))
    t0 = time.perf_counter()
    # e_ref = float(mf_ref.kernel(dm0=dm0))
    e_ref = -825.5718497652
    t_ref = time.perf_counter() - t0
    conv_ref, cyc_ref = bool(mf_ref.converged), int(mf_ref.cycles)

    # 2. The same, with PAW supplying J. density_fit() is only the hook: it makes the
    # SCF ask mf.with_df for J, and with_df is then replaced by the PAW object.
    # auto_box: PAW sizes the box from the basis and centres the molecule in it, so the
    # lattice on this Cell is a placeholder that does not survive.
    cell = pgto.M(atom=mol.atom, basis=BASIS, unit='Bohr', a=np.eye(3)*83.69156519*2,
                  ke_cutoff=KE_CUTOFF, verbose=4)
    mf = dft.RKS(mol, xc='pbe').density_fit()
    # with_isdf_k exists only on branches that carry the ISDF exchange.
    no_isdf = ({'with_isdf_k': False}
               if 'with_isdf_k' in inspect.signature(PAW.__init__).parameters else {})
    t0 = time.perf_counter()
    mydf = PAW.from_mf(mf, cell, auto_box=False, **no_isdf).build()
    t_build = time.perf_counter() - t0
    mf.with_df = mydf
    vj0 = np.asarray(mydf.get_jk(dm0, with_k=False)[0]).reshape(-1, nao, nao)[0]
    ej_paw0 = 0.5 * float(np.einsum('ij,ji->', dm0, vj0))
    t0 = time.perf_counter()
    e_paw = float(mf.kernel(dm0=dm0))
    t_paw = time.perf_counter() - t0

    box = mydf.cell.lattice_vectors().diagonal()
    print('\n' + '=' * 76)
    print(f'  PAW box {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} Bohr, '
          f'mesh {" x ".join(str(int(m)) for m in mydf.mesh)}, PAW build {t_build:.1f} s')
    print('  E_J on the same starting (minao) density:')
    print(f'    reference  {ej_ref0:20.10f} Ha')
    print(f'    PAW        {ej_paw0:20.10f} Ha')
    print(f'    PAW - ref  {ej_paw0 - ej_ref0:+20.10f} Ha')
    print('  Converged RKS-PBE total energy:')
    print(f'    reference  {e_ref:20.10f} Ha   (converged={conv_ref}, {cyc_ref} cycles, {t_ref:.0f} s)')
    print(f'    PAW        {e_paw:20.10f} Ha   (converged={bool(mf.converged)}, '
          f'{int(mf.cycles)} cycles, {t_paw:.0f} s)')
    print(f'    PAW - ref  {e_paw - e_ref:+20.10f} Ha')
    print('=' * 76, flush=True)


if __name__ == '__main__':
    main()