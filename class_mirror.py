import meep as mp
import numpy as np
from class_layer import Layer


class Mirror:
    def __init__(self, args):
        self.indices = args.get("n_indexes", args.get("indexes", [1.0, 1.0]))
        self.wavelenght = args["lam"]
        if "n_layers" in args:
            self.n_of_layers = int(args["n_layers"])
        else:
            T_target = float(args["transmission"])
            self.n_of_layers = self._n_layers_for_transmission(T_target, self.indices)
        self.front_edge = args["x_start"]
        self.direction = args.get("orientation", 1)
        self.layer_type = args.get("layer_type", "isotropic")
        self.layers: list[Layer] = []
        self.resolution = args["resolution"]
        self.cell_y: float = float(args.get("cell_y", 6.0))
        _size_y = args.get("size_y", None)
        self.size_y: float = float(_size_y) if _size_y is not None else mp.inf
        self.dimentions = mp.Vector3(0, self.size_y, mp.inf)
        self._set_layers()

    @staticmethod
    def _n_layers_for_transmission(T_target: float, indices) -> int:
        """Return total number of quarter-wave layers for T <= T_target at the Bragg wavelength.

        Uses the analytic DBR formula for N bilayer pairs in air:
          R = [(ρ−1)/(ρ+1)]²  where ρ = (n_H/n_L)^(2N)
        Inverts to give N = ceil( log(ρ) / (2 log(n_H/n_L)) ), returns 2*N layers.
        """
        if T_target <= 0:
            raise ValueError(f"Mirror: transmission target must be > 0, got {T_target}")
        if T_target >= 1.0:
            return 2
        n_H = float(max(indices))
        n_L = float(min(indices))
        if n_H <= n_L:
            raise ValueError("Mirror: n_H must be > n_L to form a Bragg reflector")
        sqrtR = np.sqrt(1.0 - T_target)
        # ρ = (1 + sqrt(R)) / (1 − sqrt(R))
        rho = (1.0 + sqrtR) / (1.0 - sqrtR)
        N = np.log(rho) / (2.0 * np.log(n_H / n_L))
        n_pairs = max(1, int(np.ceil(N)))
        return 2 * n_pairs

    def _set_layers(self):
        current_x = self.front_edge
        for i in range(self.n_of_layers):
            n = float(self.indices[i % 2])
            theta = [0, np.pi / 2][i % 2]
            l_w = self.wavelenght / 4.0 * (1.0 / n)
            current_x += self.direction * l_w / 2.0
            if self.layer_type == "lc":
                n_low, n_high = sorted(self.indices)
                n_indices = [n_low, n_high]
            else:
                n_indices = [n, n]
            pde_cell_y = self.size_y if self.size_y != mp.inf else self.cell_y
            layer = Layer({
                "layer_type": self.layer_type,
                "n_indices": n_indices,
                "size":   mp.Vector3(l_w, self.dimentions.y, self.dimentions.z),
                "center": mp.Vector3(current_x, 0, 0),
                "theta_0": theta,
                "resolution": self.resolution,
                "cell_y": pde_cell_y,
            })
            self.layers.append(layer)
            current_x += self.direction * l_w / 2.0

    def get_lenght(self):
        return sum(layer.size.x for layer in self.layers)

    def get_layer_blocks(self):
        return [layer.get_geometry_block() for layer in self.layers]

    def get_geometry_blocks(self):
        return self.get_layer_blocks()
