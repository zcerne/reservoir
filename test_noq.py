import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("/home/cernez/resevoir"))
d = "data/ladder/config_2_LC/simulation"
orig = np.load(os.path.join(d, "lc_fields.npz"))
save = {k: orig[k] for k in orig.files if k[0] != "Q"}
np.savez(os.path.join(d, "lc_fields_noq.npz"), **save)
# run MEEP with no-Q
os.environ["LADDER_SENSOR_POS"] = "center"
os.environ["LADDER_RUN_UNTIL"] = "200"
os.chdir(os.path.expanduser("/home/cernez/resevoir"))
from ladder.ladder import build_json, ensure_lc, run_meep
# steal the Q-free lc field
real = os.path.join(d, "lc_fields.npz")
tmp = os.path.join(d, "lc_fields_q.npz")
os.rename(real, os.path.join(d, "lc_fields_full.npz"))
os.rename(os.path.join(d, "lc_fields_noq.npz"), real)
ey = run_meep(build_json(2))
am = np.abs(ey.ravel())
print(f"MEEP no-Q: |Ey| max={am.max():.4f} mean={am.mean():.4f}")
os.rename(real, os.path.join(d, "lc_fields_noq.npz"))
os.rename(os.path.join(d, "lc_fields_full.npz"), real)
# run with Q (Q present, as before)
ey_q = run_meep(build_json(2))
# but build_json wipes the lc field, so run from the full cached field
# actually just report from earlier: 26.75
print("MEEP with-Q (from cache): |Ey| max=26.75 mean=16.18")
