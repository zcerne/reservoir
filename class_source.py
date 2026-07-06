import numpy as np
import meep as mp


class Source:
    def __init__(self, args):
        self.args = args
        self.center = args["center"]
        self.size = args["size"]
        self.source_type = args.get("source_type", args.get("type", "continuous"))
        self.lam = args["lam"]
        self.dlam = args.get("dlam", 0.0)
        self.fwidth = (1/(self.lam - self.dlam) - 1/(self.lam + self.dlam)) if self.dlam > 0 else 0.0
        
        self.source = None
        self.source_amp_func = None
        self.amplitude = 1.0

        _comp_map = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez,
                     "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz}
        self.component = _comp_map.get(args.get("component", "Ez"), mp.Ez)
        self._set_source()
        self._set_beam_profile_function()

    def return_source_object(self) -> list[mp.Source]:
        if isinstance(self.args.get("amplitude"), list):
            if self.args.get("grid_shape") is not None:
                return self.return_gridsource()
            return self.return_quadrosource()
        if self.amplitude is None:
            return [mp.Source(self.source, self.component, center=self.center, size=self.size,
                              amp_func=self.source_amp_func)]
        return [mp.Source(self.source, self.component, center=self.center, size=self.size,
                          amplitude=self.amplitude)]

    # 1 Meep time unit = a/c0 = 1 µm / c0 = 3.33564 fs
    _FS_PER_MEEP = 3.335640952

    def _set_source(self):
        if self.source_type == "pulsed":
            # Temporal Gaussian pulse (article STED pump/depletion): carrier at 1/lam
            # under a Gaussian envelope of temporal FWHM `pulse_fwhm_fs`, delayed by
            # `pulse_delay_fs` (e.g. pump→STED 2000 fs). MEEP GaussianSource `width` is
            # the temporal std in Meep units → width = FWHM/(2.35482·fs_per_meep).
            fwhm_fs  = float(self.args.get("pulse_fwhm_fs", 1309.0))
            delay_fs = float(self.args.get("pulse_delay_fs", 0.0))
            width = (fwhm_fs / self._FS_PER_MEEP) / 2.35482
            kw = {"width": width}
            if delay_fs > 0:
                kw["start_time"] = delay_fs / self._FS_PER_MEEP
            self.source = mp.GaussianSource(1/self.lam, **kw)
        elif self.source_type == "gaussian":
            if self.fwidth is not None:
                self.source = mp.GaussianSource(1/self.lam, fwidth=self.fwidth)
            else:
                dfreq = 1/(self.lam - self.dlam/2) - 1/(self.lam + self.dlam/2)
                self.source = mp.GaussianSource(1/self.lam, dfreq=dfreq)
        else:
            self.source = mp.ContinuousSource(1/self.lam)

    def _set_beam_profile_function(self):
        amp = self.args["amplitude"]
        if self.source_type == "gaussian_flat":
            w0 = self.args.get("w0", None)
            if w0 is not None:
                self.source_amp_func = lambda r, _w0=w0, _amp=amp: _gaussian_amp(r, _w0, _amp)
                self.amplitude = None
            else:
                self.amplitude = amp
        elif self.source_type == "flattop":
            half = self.size.x / 2
            edge_w = self.args.get("edge_w", 0.1)
            self.source_amp_func = lambda r, _amp=amp, _half=half, _w=edge_w: float(
                _amp * np.exp(-((max(0.0, abs(r.x) - (_half - 3*_w)) / _w) ** 2))
            )
            self.amplitude = None
        else:
            self.source_amp_func = None
            self.amplitude = amp

    def return_quadrosource(self) -> list[mp.Source]:
        amps = self.args["amplitude"]  # list of N amplitudes → N equal y-strips
        n = len(amps)
        sy = float(self.size.y)
        strip_sy = sy / n
        sources = []
        for i, amp in enumerate(amps):
            cy = float(self.center.y) - sy / 2 + (i + 0.5) * strip_sy
            center = mp.Vector3(float(self.center.x), cy, float(self.center.z))
            size   = mp.Vector3(float(self.size.x), strip_sy, float(self.size.z))
            sources.append(mp.Source(self.source, self.component, center=center, size=size,
                                     amplitude=float(amp)))
        return sources


    def return_gridsource(self) -> list[mp.Source]:
        """2D grid source: amplitude is flat list of ny*nz values, grid_shape=[ny, nz]."""
        amps = self.args["amplitude"]
        ny, nz = self.args["grid_shape"]
        sy = float(self.size.y)
        sz = float(self.size.z)
        dy = sy / ny
        dz = sz / nz
        sources = []
        for j in range(ny):
            for k in range(nz):
                cy = float(self.center.y) - sy / 2 + (j + 0.5) * dy
                cz = float(self.center.z) - sz / 2 + (k + 0.5) * dz
                center = mp.Vector3(float(self.center.x), cy, cz)
                size   = mp.Vector3(float(self.size.x), dy, dz)
                sources.append(mp.Source(self.source, self.component, center=center, size=size,
                                         amplitude=float(amps[j * nz + k])))
        return sources


def _gaussian_amp(r, _w0, _amp):
    return _amp * np.exp(-r.y**2 / _w0**2)
