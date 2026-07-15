"""gpumeep twin of class_guide.Guide."""
from gpumeep_setup import gm


class GuideGPU:
    def __init__(self, args):
        self.n_index = args.get("refractive_index", args.get("index", 1.0))
        self.sizes = args["sizes"]
        self.center = args.get("center", gm.Vector3())

    def get_geometry_blocks(self):
        return [gm.Block(center=self.center, size=self.sizes,
                         material=gm.Medium(index=self.n_index))]
