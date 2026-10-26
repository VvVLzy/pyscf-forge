'''Compare CP2K GAPW and PySCF PAW He2 energies against the exact FFTDF result.

usage: python compare.py <alpha0> <Rb>      e.g. python compare.py 12.5 1.7

Reads cp2k/a_<alpha0>_r_<Rb>/He2bondLength_cp2k.csv and every PySCF csv in
pyscf/data/ with the same alpha0/Rb, writes He2_compare_a_<alpha0>_r_<Rb>.csv
and He2_error_vs_exact_a_<alpha0>_r_<Rb>.png.
'''
import os
import sys
import csv
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
HA2MHA = 1e3

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'
REFERENCE = '#8a8984'   # neutral gray for the reproduced old result

# series key -> (label, color, marker, linestyle); entity colors match the timing plots
SERIES = {
    'cp2k':              ('CP2K GAPW', '#2a78d6', 'o', '-'),
    'paw_gdfNuc_False':  ('PySCF PAW, gdfNuc=False', '#eb6834', 's', '-'),
    'paw_gdfNuc_True':   ('PySCF PAW, gdfNuc=True', '#1baf7a', '^', '-'),
}

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def load(alpha0, rb):
    cp2k = read_csv(os.path.join(base_dir, 'cp2k', f'a_{alpha0}_r_{rb}', 'He2bondLength_cp2k.csv'))
    pattern = os.path.join(base_dir, 'pyscf', 'data', f'He2bondLength_*_a_{alpha0}_r_{rb}_*gdfNuc_*.csv')
    paw_runs = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)[:-4]
        key = 'paw_' + name[name.index('gdfNuc_'):].replace('_exact_2000', '')
        paw_runs[key] = read_csv(path)

    exact_ref = None
    rows = []
    for i, c in enumerate(cp2k):
        r = float(c['r'])
        row = {'r': r}
        for key, data in paw_runs.items():
            # CP2K bond lengths are the same values written with 8 decimals
            assert abs(float(data[i]['r']) - r) < 1e-6
            e_exact = float(data[i]['etot_exact'])
            if 'etot_exact' not in row:
                row['etot_exact'] = e_exact
            assert abs(e_exact - row['etot_exact']) < 1e-8   # all runs share the same reference
            row['etot_' + key] = float(data[i]['etot_paw'])
            row['conv_' + key] = data[i]['conv_paw'] == 'True' and data[i]['conv_exact'] == 'True'
        row['etot_cp2k'] = float(c['etot_cp2k'])
        row['conv_cp2k'] = c['conv_cp2k'] == 'True'
        for key in ['cp2k'] + list(paw_runs):
            row['err_' + key] = (row['etot_' + key] - row['etot_exact']) * HA2MHA
        rows.append(row)
    return rows, ['cp2k'] + list(paw_runs)

def plot(rows, keys, alpha0, rb, fname):
    r = [row['r'] for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.grid(axis='both', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

    for key in keys:
        label, color, marker, ls = SERIES[key]
        err = [row['err_' + key] for row in rows]
        ref = color == REFERENCE
        # the legend carries the value at the shortest bond, where the curves differ most
        ax.plot(r, err, color=color, linewidth=1.5 if ref else 2, linestyle=ls, marker=marker,
                markersize=5 if ref else 6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label='%s   %+.3f mHa at %.1f Å' % (label, err[0], r[0]), zorder=1 if ref else 2)

    ax.set_xlabel('He–He bond length (Å)', color=TEXT_SECONDARY)
    ax.set_ylabel('E − E$_{exact}$ (mHa)', color=TEXT_SECONDARY)
    ax.set_title('He2, 3 Å box, no XC, 200 Ha / 400 Ry: α0 = %s, R = %s bohr' % (alpha0, rb),
                 loc='left', fontsize=12, color=TEXT_PRIMARY, pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    for text in leg.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.tight_layout()
    out = os.path.join(base_dir, fname)
    fig.savefig(out, dpi=300, facecolor=SURFACE)
    print('Saved', out)

def main():
    alpha0, rb = sys.argv[1], sys.argv[2]
    rows, keys = load(alpha0, rb)
    suffix = f'a_{alpha0}_r_{rb}'

    out = os.path.join(base_dir, f'He2_compare_{suffix}.csv')
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print('errors vs exact FFTDF (mHa)')
    short = {'cp2k': 'CP2K', 'paw_gdfNuc_False': 'PAW gdf=F', 'paw_gdfNuc_True': 'PAW gdf=T'}
    print('%7s ' % 'r (A)' + ' '.join('%14s' % short.get(k, k[:14]) for k in keys))
    for row in rows:
        print('%7.4f ' % row['r'] + ' '.join('%+14.4f' % row['err_' + k] for k in keys))
    for key in keys:
        err = [row['err_' + key] for row in rows]
        n_better = sum(abs(row['err_cp2k']) < abs(row['err_' + key]) for row in rows)
        print('%-32s max |err| %.4f  mean |err| %.4f mHa%s' % (
            key, max(abs(e) for e in err), sum(abs(e) for e in err) / len(err),
            '' if key == 'cp2k' else '  (CP2K closer at %d/%d)' % (n_better, len(rows))))
    print('all converged:', all(v for row in rows for k, v in row.items() if k.startswith('conv_')))
    print('Saved', out)
    plot(rows, keys, alpha0, rb, f'He2_error_vs_exact_{suffix}.png')

if __name__ == '__main__':
    main()
