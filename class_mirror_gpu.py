"""gpumeep twin of class_mirror.Mirror (isotropic quarter-wave DBR stack)."""
from gpumeep_setup import gm


class MirrorGPU:
    """Quarter-wave DBR stack as isotropic gm Blocks; layer_type 'lc' is
    MEEP-only."""

    def __init__(self, args):
        if args.get("layer_type", "isotropic") != "isotropic":
            raise NotImplementedError("MirrorGPU: isotropic layers only")
        self.indices = args.get("n_indexes", args.get("indexes", [1.0, 1.0]))
        self.lam = float(args["lam"])
        self.n_layers = int(args["n_layers_resolved"])
        self.x_start = float(args["x_start_meep"])
        self.direction = args.get("orientation", 1)
        self.size_y = float(args.get("size_y", 0.0)) or gm.inf

    def get_geometry_blocks(self):
        blocks, x = [], self.x_start
        for i in range(self.n_layers):
            n = float(self.indices[i % 2])
            lw = self.lam / 4.0 / n
            x += self.direction * lw / 2.0
            blocks.append(gm.Block(center=gm.Vector3(x, 0, 0),
                                   size=gm.Vector3(lw, self.size_y, gm.inf),
                                   material=gm.Medium(index=n)))
            x += self.direction * lw / 2.0
        return blocks
