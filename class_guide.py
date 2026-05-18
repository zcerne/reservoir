import meep as mp
class Guide:
    def __init__(self, args):
        self.n_index = args.get("refractive_index", args.get("index", 1.0))
        self.sizes = args["sizes"]
        self.center = args.get("center", mp.Vector3())

    def get_geometry_block(self):
        return mp.Block(center=self.center, material=mp.Medium(index=self.n_index), size=self.sizes)

    def get_geometry_blocks(self):
        return [self.get_geometry_block()]

