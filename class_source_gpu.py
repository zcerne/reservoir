"""gpumeep twin of class_source.Source.

Time profiles: continuous / gaussian / pulsed (fs-parameterized). Amplitude
forms: scalar, strip-list (N equal y-strips), grid (ny×nz cells).
Beam-profile amp_funcs (gaussian_flat w0 / flattop) are MEEP-only."""
from gpumeep_setup import gm, FS_PER_MEEP


class SourceGPU:
    def __init__(self, args):
        self.args = args
        self.center = args["center"]
        self.size = args["size"]
        self.component = args.get("component", "Ez")
        self.lam = float(args["lam"])
        self.dlam = float(args.get("dlam", 0.0))
        st = args.get("source_type", args.get("type", "continuous"))
        if st in ("gaussian_flat", "flattop") and args.get("w0") is not None:
            raise NotImplementedError(f"SourceGPU: {st} amp profile is MEEP-only")
        self.source_type = st
        self.src = self._time_profile()

    def _time_profile(self):
        f0 = 1.0 / self.lam
        if self.source_type == "pulsed":
            fwhm_fs = float(self.args.get("pulse_fwhm_fs", 1309.0))
            delay_fs = float(self.args.get("pulse_delay_fs", 0.0))
            width = (fwhm_fs / FS_PER_MEEP) / 2.35482
            return gm.GaussianSource(f0, width=width,
                                     start_time=delay_fs / FS_PER_MEEP)
        if self.source_type == "gaussian":
            fwidth = (1 / (self.lam - self.dlam) - 1 / (self.lam + self.dlam)
                      if self.dlam > 0 else None)
            if fwidth:
                return gm.GaussianSource(f0, fwidth=fwidth)
            return gm.GaussianSource(f0, fwidth=0.2 * f0)
        return gm.ContinuousSource(f0)

    def return_source_object(self):
        amp = self.args.get("amplitude", 1.0)
        if isinstance(amp, list):
            if self.args.get("grid_shape") is not None:
                return self._grid_sources(amp)
            return self._strip_sources(amp)
        return [gm.Source(self.src, self.component, center=self.center,
                          size=self.size, amplitude=float(amp))]

    def _strip_sources(self, amps):
        n = len(amps)
        sy = float(self.size.y)
        strip = sy / n
        out = []
        for i, a in enumerate(amps):
            cy = float(self.center.y) - sy / 2 + (i + 0.5) * strip
            out.append(gm.Source(
                self.src, self.component,
                center=gm.Vector3(float(self.center.x), cy, float(self.center.z)),
                size=gm.Vector3(float(self.size.x), strip, float(self.size.z)),
                amplitude=float(a)))
        return out

    def _grid_sources(self, amps):
        ny, nz = self.args["grid_shape"]
        sy, sz = float(self.size.y), float(self.size.z)
        dy, dz = sy / ny, sz / nz
        out = []
        for j in range(ny):
            for k in range(nz):
                cy = float(self.center.y) - sy / 2 + (j + 0.5) * dy
                cz = float(self.center.z) - sz / 2 + (k + 0.5) * dz
                out.append(gm.Source(
                    self.src, self.component,
                    center=gm.Vector3(float(self.center.x), cy, cz),
                    size=gm.Vector3(float(self.size.x), dy, dz),
                    amplitude=float(amps[j * nz + k])))
        return out
