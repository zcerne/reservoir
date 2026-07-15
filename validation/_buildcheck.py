from class_simulation import Simulation
sim = Simulation("data/reservoir_clasifications/14_2D_sted_resonator")
sim._set_everything()
print("BUILD OK cell_x=%.2f cell_y=%.2f" % (sim._cell_x, sim._cell_y))
for o in sim.objects_args:
    c = o.get("center"); xs = o.get("x_start")
    cx = getattr(c, "x", None)
    print("  %9s | %16s | size_x=%.3f | center_x=%s | x_start=%s" % (
        o.get("_key"), o.get("class"), o.get("size_x", 0), cx, xs))
try:
    sim.plot_setup(); print("setup.png written")
except Exception as e:
    print("plot_setup err:", repr(e))
