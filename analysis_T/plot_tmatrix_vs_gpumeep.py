"""
Compare T-matrix prediction (from MEEP basis files) against GPUmeep direct
solve for the same random seed-42 source amplitude.

MEEP uses e^{-iωt} convention; GPUmeep effectively uses e^{+iωt} — so the
two solvers' complex DFT fields differ by complex conjugation. Intensity
|E|² is convention-independent and matches directly.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_MEEP = '/home/ziga/Orion/resevoir/data/source_mnist'
BASE_GPU  = '/home/ziga/Orion/resevoir/data/source_mnist_run'

T   = np.load(f'{BASE_MEEP}/simulation_T/T_matrix.npz')
amp = np.array(json.load(open(f'{BASE_GPU}/simulation_data.json'))['source_1']['amplitude'])
g   = np.load(f'{BASE_GPU}/simulation/monitor_2.npz')

fig, axes = plt.subplots(3, 4, figsize=(18, 11))
extent = [-9, 9, -7, 7]   # z µm, y µm

for i, lbl in enumerate(['Ex', 'Ey', 'Ez']):
    # MEEP T@amp -> (282, 360), center-crop y to 280 to match GPUmeep
    p  = (T[f'T_{lbl}'] @ amp).reshape(282, 360)[1:281]
    gg = g[lbl][0]                                              # GPUmeep (280, 360)
    gg_c = gg.conj()                                            # convention-fixed

    Ip, Ig = np.abs(p)**2, np.abs(gg)**2
    Ipn, Ign = Ip / Ip.max(), Ig / Ig.max()

    # Correlations
    Icorr = np.sum(Ip*Ig) / (np.linalg.norm(Ip)*np.linalg.norm(Ig))
    ccorr = np.abs(np.vdot(p.ravel(), gg_c.ravel())) / (np.linalg.norm(p)*np.linalg.norm(gg))
    rms   = np.sqrt(np.mean((Ipn - Ign)**2))

    im0 = axes[i,0].imshow(Ip,  extent=extent, origin='lower', aspect='auto', cmap='inferno')
    axes[i,0].set_title(f'MEEP T·amp  |{lbl}|²   (peak {Ip.max():.0f})')
    plt.colorbar(im0, ax=axes[i,0], fraction=0.046)

    im1 = axes[i,1].imshow(Ig,  extent=extent, origin='lower', aspect='auto', cmap='inferno')
    axes[i,1].set_title(f'GPUmeep direct  |{lbl}|²   (peak {Ig.max():.2g})')
    plt.colorbar(im1, ax=axes[i,1], fraction=0.046)

    im2 = axes[i,2].imshow(Ipn, extent=extent, origin='lower', aspect='auto',
                            vmin=0, vmax=1, cmap='inferno')
    axes[i,2].set_title(f'MEEP T·amp (peak-norm)')
    plt.colorbar(im2, ax=axes[i,2], fraction=0.046)

    im3 = axes[i,3].imshow(Ign, extent=extent, origin='lower', aspect='auto',
                            vmin=0, vmax=1, cmap='inferno')
    axes[i,3].set_title(f'GPUmeep direct (peak-norm)\n'
                        f'|E|² corr = {Icorr:.3f},  complex corr (conj) = {ccorr:.3f}',
                        fontsize=9)
    plt.colorbar(im3, ax=axes[i,3], fraction=0.046)

    for ax in axes[i]:
        ax.set_xlabel('z (µm)')
    axes[i,0].set_ylabel(f'y (µm)\n[{lbl}]')

    print(f'{lbl}:  intensity corr={Icorr:.4f},  complex corr (after conj)={ccorr:.4f},  peak-norm RMS={rms:.4f}')

fig.suptitle('T-matrix (MEEP basis) · seed-42 amplitude   vs   GPUmeep direct solve\n'
             'source_mnist  res=20  monitor_2 (yz after reservoir)   —   intensity matches; '
             'GPUmeep complex field = conj(MEEP) (time-FT sign convention)',
             fontsize=12)
fig.tight_layout()
fig.subplots_adjust(top=0.92)
out = f'{BASE_GPU}/compare_tmatrix_vs_gpumeep.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
print(f'Saved {out}')
