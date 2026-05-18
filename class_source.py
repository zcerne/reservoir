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

        self._set_source()
        self._set_beam_profile_function()

    def return_source_object(self):
        if self.amplitude is None:
            return mp.Source(self.source, mp.Ez, center=self.center, size=self.size,
                             amp_func=self.source_amp_func)
        return mp.Source(self.source, mp.Ez, center=self.center, size=self.size,
                         amplitude=self.amplitude)

    def _set_source(self):
        if self.source_type == "gaussian":
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


def _gaussian_amp(r, _w0, _amp):
    return _amp * np.exp(-r.y**2 / _w0**2)
