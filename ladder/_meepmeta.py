import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ladder import build_json  # noqa: E402
import meep as mp  # noqa: F401,E402
from class_simulation import Simulation  # noqa: E402

p = build_json(4)
sm = Simulation(p); sm._set_everything(); sm.simulation.init_sim()
sen = sm.sensors[0]
xs, ys, zs, ws = sm.simulation.get_array_metadata(center=sen.center, size=sen.size)
xs = list(xs)
print("MEEP monitor sen.center.x =", float(sen.center.x))
print("MEEP monitor metadata n_x =", len(xs), " x-values:", xs[:4])
print("MEEP source snapped center.x =", float(sm.simulation.sources[0].center.x))
xa, ya, za, wa = sm.simulation.get_array_metadata()
xa = list(xa)
print("MEEP full-grid x[0]=%.6f x[1]=%.6f dx=%.6f" % (xa[0], xa[1], xa[1]-xa[0]))
print("MEEP grid x near 3.35:", [round(v, 6) for v in xa if 3.33 < v < 3.38])
print("MEEP grid x near -5.6:", [round(v, 6) for v in xa if -5.62 < v < -5.58])
print("DONE_META")
