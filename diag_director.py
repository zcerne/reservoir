import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
from ladder.ladder import build_json
p = build_json(2)
import importlib.util
def load_gpu():
    sys.modules.pop("class_simulation_gpu",None)
    s=importlib.util.spec_from_file_location("class_simulation_gpu","/home/cernez/resevoir/class_simulation_gpu.py")
    m=importlib.util.module_from_spec(s); sys.modules["class_simulation_gpu"]=m; s.loader.exec_module(m); return m
csg = load_gpu()
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg._set_data(); sg._update_all_args(); sg._setup_lc_interp()
ip_g, it_g, gx0,gx1,gy0,gy1 = sg._lc_interp
res = next(o for o in sg.objects_args if o.get("class") in ("reservoir","voltage_reservoir"))
rx0=res["edge_x_meep"]; rx1=rx0+float(res["size_x"])
sizes=res.get("sizes"); ry=(float(sizes[1]) if isinstance(sizes,list) and len(sizes)>1 else sg.cell_y)
lc=np.load(sg.folder_path+'/simulation/lc_fields.npz')
print("lc_fields phi shape:", lc["phi"].shape, "x-range[%.3f,%.3f] y-range[%.3f,%.3f]"%(lc["x"].min(),lc["x"].max(),lc["y"].min(),lc["y"].max()))
print("gpu: res size_x=%.3f sizes[1]=%.3f edge_x=%.3f -> interp x[%.3f,%.3f] y[%.3f,%.3f]"%(float(res["size_x"]),ry,rx0,gx0,gx1,gy0,gy1))

from class_reservoir import Reservoir as R
rr = R(sg.folder_path); rr.load_fields()
phi_m, theta_m, *_ = rr.get_results_2d()
cell = rr._cell_size(); sx=float(cell[0]); sy=float(cell[1])
nx_pts,ny_pts=phi_m.shape
from scipy.interpolate import RectBivariateSpline
x_lc=np.linspace(-sx/2,sx/2,nx_pts); y_lc=np.linspace(-sy/2,sy/2,ny_pts)
ip_m=RectBivariateSpline(x_lc,y_lc,phi_m); it_m=RectBivariateSpline(x_lc,y_lc,theta_m)
cx_res=rr._meep_center_x
print("MEEP: _cell_size sx=%.3f sy=%.3f cx_res=%.3f get_results_2d shape=%s -> interp x[%.3f,%.3f]+cx y[%.3f,%.3f]"%(sx,sy,cx_res,phi_m.shape,x_lc[0]+cx_res,x_lc[-1]+cx_res,y_lc[0],y_lc[-1]))
# compare ny^2 over reservoir
xr=np.linspace(rx0+0.1,rx1-0.1,20); yr=np.linspace(-2,2,20); nyg=[];nym=[]
for x in xr:
    for y in yr:
        pg=float(ip_g.ev(x,y)); tg=float(it_g.ev(x,y))
        pm=float(ip_m.ev(x-cx_res,y)); tm=float(it_m.ev(x-cx_res,y))
        nyg.append((np.sin(tg)*np.sin(pg))**2); nym.append((np.sin(tm)*np.sin(pm))**2)
print("<ny^2> gpu=%.4f MEEP=%.4f"%(np.mean(nyg),np.mean(nym)))
print("DONE")
