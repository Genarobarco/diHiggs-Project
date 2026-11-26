import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import vector as vec
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import stats as st
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.colors import LogNorm

vec.register_awkward()

saving_path = '/home/gbarco/DiHiggsProject/RecAnalysis'

os.makedirs(saving_path, exist_ok=True)

rutas = {
    'signal': "/mnt/R5/gbarco/BM500_gghhj_1800k/ROOTs/output_ggHH_bbtata_BP1_81_0_delphes_events.root",
#    'bkg': "/home/usuario/Madgraph/projects/pp_tthj-1800k_reduced/output_pp_tthj_h-tata_1800k_104_0_delphes_events.root",
}

truth_keys = ["Particle.PID", "Particle.Status", "Particle.D1", "Particle.D2", "Particle.M1", "Particle.M2",
              "Particle.Px", "Particle.Py", "Particle.Pz", "Particle.E",
              "Particle.X", "Particle.Y", "Particle.Z"]

def load_truth_taus(path, step_size="500 MB"):
    trozos = []
    for batch in up.iterate(f"{path}:Delphes", truth_keys, step_size=step_size):
        trozos.append(ak.zip({
            "px":  batch["Particle.Px"],
            "py":  batch["Particle.Py"],
            "pz":  batch["Particle.Pz"],
            "E":   batch["Particle.E"],
            "pid": batch["Particle.PID"],
            "status": batch["Particle.Status"],
            "M1":  batch["Particle.M1"],
            "M2":  batch["Particle.M2"],
            "D1":  batch["Particle.D1"],
            "D2":  batch["Particle.D2"],
            "vx":  batch["Particle.X"],
            "vy":  batch["Particle.Y"],
            "vz":  batch["Particle.Z"],
        }, with_name="Momentum4D"))
    return ak.concatenate(trozos)

truth = load_truth_taus(rutas['signal'])

taus = truth[abs(truth.pid) == 15] # selecciono los tau

# Calculo los taus que decaen inmediatamente del Higgs

m1_ok = ak.where(truth.M1 >= 0, truth.M1, 0)       # las sin madre traen -1
madre = ak.where(truth.M1 >= 0, truth.pid[m1_ok], 0)

taus_del_higgs = truth[(abs(truth.pid) == 15) & (madre == 25)]


# Y ahora los taus que decaen exclusivamente a partículas que no son taus
d1_ok = ak.where(truth.D1 >= 0, truth.D1, 0)
d2_ok = ak.where(truth.D2 >= 0, truth.D2, 0)

hija1 = ak.where(truth.D1 >= 0, truth.pid[d1_ok], 0)
hija2 = ak.where(truth.D2 >= 0, truth.pid[d2_ok], 0)

taus_que_decaen = truth[(abs(truth.pid) == 15)
                        & (truth.D1 >= 0)          # que tenga hijas
                        & (abs(hija1) != 15)
                        & (abs(hija2) != 15)]

def mas_duro(col):
    return col[ak.argsort(col.pt, axis=1, ascending=False)][:, 0]

# separo por carga:  pid +15 = tau-,  -15 = tau+
hm = taus_del_higgs[taus_del_higgs.pid > 0]
hp = taus_del_higgs[taus_del_higgs.pid < 0]
dm = taus_que_decaen[taus_que_decaen.pid > 0]
dp = taus_que_decaen[taus_que_decaen.pid < 0]

ok = ((ak.num(hm, axis=1) >= 1) & (ak.num(hp, axis=1) >= 1)
      & (ak.num(dm, axis=1) >= 1) & (ak.num(dp, axis=1) >= 1))

hm, hp = mas_duro(hm[ok]), mas_duro(hp[ok])
dm, dp = mas_duro(dm[ok]), mas_duro(dp[ok])

print(f"eventos usables: {ak.sum(ok)}/{len(ok)}")

def componentes(a, b):
    return (np.abs(ak.to_numpy(a.vx - b.vx)),
            np.abs(ak.to_numpy(a.vy - b.vy)),
            np.abs(ak.to_numpy(a.vz - b.vz)))

dx_m, dy_m, dz_m = componentes(hm, dm)
dx_p, dy_p, dz_p = componentes(hp, dp)

dx = np.concatenate([dx_m, dx_p])
dy = np.concatenate([dy_m, dy_p])
dz = np.concatenate([dz_m, dz_p])

d_vuelo = np.sqrt(dx**2 + dy**2 + dz**2)

paneles = [
    (dx,      r"$|\Delta x|$ [mm]", (0, 2)),
    (dy,      r"$|\Delta y|$ [mm]", (0, 2)),
    (dz,      r"$|\Delta z|$ [mm]", (0, 5)),
    (d_vuelo, r"$d_{vuelo}$ [mm]",  (0, 10)),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (datos, xlab, rango) in zip(axes.flat, paneles):
    ax.hist(datos, bins=100, range=rango, histtype="step", linewidth=2)
    ax.axvline(np.median(datos), color="r", ls="--", lw=1,
               label=f"mediana {np.median(datos):.3f} mm")
    ax.set_xlabel(xlab)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes[0, 0].set_ylabel("Taus")
axes[1, 0].set_ylabel("Taus")

plt.tight_layout()
plt.savefig(f'{saving_path}/Distancia.png', dpi = 300)
plt.close()

E_higgs  = ak.to_numpy(ak.flatten(taus_del_higgs.E))
E_decaen = ak.to_numpy(ak.flatten(taus_que_decaen.E))

plt.figure(figsize=(8, 6))

for datos, etiqueta in [(E_higgs,  "taus del Higgs"),
                        (E_decaen, "taus que decaen")]:
    plt.hist(datos, bins=60, range=(0, 500), histtype="step", linewidth=2,
             label=f"{etiqueta}  (N={len(datos):,}, mediana {np.median(datos):.1f} GeV)")

plt.xlabel(r"$E$ [GeV]")
plt.ylabel("Taus")
plt.grid(alpha=0.3)
plt.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{saving_path}/Energia.png', dpi = 300)
plt.close()
