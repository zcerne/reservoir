"""Difference maps: T·amp (MEEP basis) vs GPUmeep direct, source_mnist seed-42.

Five columns per row (Ex, Ey, Ez):
    1) |E| from MEEP T·amp (peak-normalised)
    2) |E| from GPUmeep direct (peak-normalised)
    3) absolute difference |I_meep_norm - I_gpu_norm|
    4) signed difference (I_meep_norm - I_gpu_norm)
    5) MEEP / GPUmeep amplitude ratio (should be ~constant if normalisation is global)
"""
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_MEEP = '/home/ziga/Orion/resevoir/data/source_mnist'
BASE_GPU  = '/home/ziga/Orion/resevoir/data/source_mnist_run'

T   = np.load(f'{BASE_MEEP}/simulation_T/T_matrix.npz')
amp = np.array(json.load(open(f'{BASE_GPU}/simulation_data.json'))['source_1']['amplitude'])
g   = np.load(f'{BASE_GPU}/simulation/monitor_2.npz')

fig, axes = plt.subplots(3, 5, figsize=(22, 11))
extent = [-9, 9, -7, 7]  # z µm, y µm

for i, lbl in enumerate(['Ex', 'Ey', 'Ez']):
    p  = (T[f'T_{lbl}'] @ amp).reshape(282, 360)[1:281]
    gg = g[lbl][0]
    Ip, Ig = np.abs(p)**2, np.abs(gg)**2
    Ipn, Ign = Ip / Ip.max(), Ig / Ig.max()

    diff       = Ipn - Ign
    diff_abs   = np.abs(diff)
    ratio      = np.abs(p) / (np.abs(gg) + 1e-30)
    rms        = np.sqrt(np.mean(diff**2))
    Icorr      = np.sum(Ip*Ig)/(np.linalg.norm(Ip)*np.linalg.norm(Ig))

    im0 = axes[i,0].imshow(Ipn, extent=extent, origin='lower', aspect='auto',
                           vmin=0, vmax=1, cmap='inferno')
    axes[i,0].set_title(f'MEEP T·amp  |{lbl}|² peak-norm')
    plt.colorbar(im0, ax=axes[i,0], fraction=0.046)

    im1 = axes[i,1].imshow(Ign, extent=extent, origin='lower', aspect='auto',
                           vmin=0, vmax=1, cmap='inferno')
    axes[i,1].set_title(f'GPUmeep direct  |{lbl}|² peak-norm')
    plt.colorbar(im1, ax=axes[i,1], fraction=0.046)

    im2 = axes[i,2].imshow(diff_abs, extent=extent, origin='lower', aspect='auto',
                           vmin=0, vmax=diff_abs.max(), cmap='viridis')
    axes[i,2].set_title(f'|Δ peak-norm |{lbl}|²|   (RMS {rms:.3f}, max {diff_abs.max():.2f})')
    plt.colorbar(im2, ax=axes[i,2], fraction=0.046)

    vmax = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[i,3].imshow(diff, extent=extent, origin='lower', aspect='auto',
                           vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    axes[i,3].set_title(f'Δ signed (MEEP − GPUmeep)')
    plt.colorbar(im3, ax=axes[i,3], fraction=0.046)

    # Amplitude ratio — should be ~constant if it's pure global normalisation;
    # spatial variation reveals where the two solvers actually disagree.
    # Mask low-amplitude regions (noisy ratio there)
    mask = (np.abs(gg) > 0.1*np.abs(gg).max())
    ratio_show = np.where(mask, ratio, np.nan)
    med = np.nanmedian(ratio_show)
    im4 = axes[i,4].imshow(ratio_show, extent=extent, origin='lower', aspect='auto',
                           vmin=0.7*med, vmax=1.3*med, cmap='RdBu_r')
    axes[i,4].set_title(f'|MEEP|/|GPU|  (median {med:.2f})')
    plt.colorbar(im4, ax=axes[i,4], fraction=0.046)

    for ax in axes[i]:
        ax.set_xlabel('z (µm)')
    axes[i,0].set_ylabel(f'y (µm)\n[{lbl}]')

    print(f'{lbl}: |E|² peak-norm RMS={rms:.4f}, max diff={diff_abs.max():.3f}, '
          f'intensity corr={Icorr:.4f}, median |MEEP|/|GPU|={med:.2f}')

fig.suptitle('Difference maps: MEEP T·amp  vs  GPUmeep direct   (source_mnist, seed-42, res=20)',
             fontsize=12)
fig.tight_layout(); fig.subplots_adjust(top=0.93)
out = f'{BASE_GPU}/compare_tmatrix_vs_gpumeep_diff.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
print(f'Saved {out}')
