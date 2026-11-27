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

carpeta_truth = f'{saving_path}/Truth'

os.makedirs(carpeta_truth, exist_ok=True)

rutas = {
    'signal': "/mnt/R5/gbarco/BM500_gghhj_1800k/ROOTs/output_ggHH_bbtata_BP1_81_*_delphes_events.root",
    'lep': "/mnt/R5/gbarco/pp_tthj_1800k/ROOTs/output_pp_tthj_h-tata_1800k_104_*_delphes_events.root",
    'had': '/mnt/R5/gbarco/pp_tthj-had/ROOTs/output_pp_tthj_had_119_*_delphes_events.root',
    'semiA': '/mnt/R5/gbarco/pp_tthj-semilep_A/ROOTs/output_pp_tthj_semilep_A_126_*_delphes_events.root' 
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

def mas_duro(col):
    return col[ak.argsort(col.pt, axis=1, ascending=False)][:, 0]

def componentes(a, b):
    return (np.abs(ak.to_numpy(a.vx - b.vx)),
            np.abs(ak.to_numpy(a.vy - b.vy)),
            np.abs(ak.to_numpy(a.vz - b.vz)))


# ============================================================================
# Calculo, una muestra por vez. Los resultados se guardan para plotear al final.
# ============================================================================

Distancias = {}     # key -> dict con dx, dy, dz, d_vuelo
Energias   = {}     # key -> dict con E_higgs, E_decaen

for key, ruta in rutas.items():

    print('')
    print(f'=== {key} ===')

    truth = load_truth_taus(ruta)

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

    dx_m, dy_m, dz_m = componentes(hm, dm)
    dx_p, dy_p, dz_p = componentes(hp, dp)

    dx = np.concatenate([dx_m, dx_p])
    dy = np.concatenate([dy_m, dy_p])
    dz = np.concatenate([dz_m, dz_p])

    d_vuelo = np.sqrt(dx**2 + dy**2 + dz**2)

    Distancias[key] = {"dx": dx, "dy": dy, "dz": dz, "d_vuelo": d_vuelo}

    Energias[key] = {"higgs":  ak.to_numpy(ak.flatten(taus_del_higgs.E)),
                     "decaen": ak.to_numpy(ak.flatten(taus_que_decaen.E))}


# ============================================================================
# Plots: una figura por observable, todas las muestras superpuestas
# ============================================================================

COLOR_KEY = {key: c for key, c in zip(rutas.keys(),
                                      plt.rcParams["axes.prop_cycle"].by_key()["color"])}

# ---- distancias ----

paneles = [
    ("dx",      r"$|\Delta x|$ [mm]", (0, 2)),
    ("dy",      r"$|\Delta y|$ [mm]", (0, 2)),
    ("dz",      r"$|\Delta z|$ [mm]", (0, 5)),
    ("d_vuelo", r"$d_{vuelo}$ [mm]",  (0, 10)),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (campo, xlab, rango) in zip(axes.flat, paneles):

    for key in Distancias:
        datos = Distancias[key][campo]
        w = np.full(len(datos), 1.0 / len(datos))      # fraccion, para comparar formas
        ax.hist(datos, bins=100, range=rango, weights=w, histtype="step",
                linewidth=2, color=COLOR_KEY[key],
                label=f"{key}  (mediana {np.median(datos):.3f} mm)")
        ax.axvline(np.median(datos), color=COLOR_KEY[key], ls="--", lw=1, alpha=0.7)

    ax.set_xlabel(xlab)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes[0, 0].set_ylabel("Fracción de taus")
axes[1, 0].set_ylabel("Fracción de taus")

plt.tight_layout()
plt.savefig(f'{carpeta_truth}/Distancia.png', dpi = 300)
plt.close()

# ---- energias ----
# color por muestra, estilo de linea por conjunto de taus

plt.figure(figsize=(9, 6))

for key in Energias:
    for campo, etiqueta, ls in [("higgs",  "del Higgs",  "-"),
                                ("decaen", "que decaen", "--")]:
        datos = Energias[key][campo]
        w = np.full(len(datos), 1.0 / len(datos))
        plt.hist(datos, bins=60, range=(0, 500), weights=w, histtype="step",
                 linewidth=2, linestyle=ls, color=COLOR_KEY[key],
                 label=f"{key} - taus {etiqueta}  (N={len(datos):,}, mediana {np.median(datos):.1f} GeV)")

plt.xlabel(r"$E$ [GeV]")
plt.ylabel("Fracción de taus")
plt.yscale("log")
plt.grid(alpha=0.3)
plt.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{carpeta_truth}/Energia.png', dpi = 300)
plt.close()

print('')
print(f'Guardado en {carpeta_truth}')
