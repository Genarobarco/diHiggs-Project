import h5py
import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vector as vec
import numpy as np
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.colors import LogNorm
import Functions as f

vec.register_awkward()

# ============================================================================
# Parameters
# ============================================================================

saving_path = '/home/gbarco/DiHiggsProject/RecAnalysis/'

# Jets preselection cuts
Pre_PT_jet = 20       # GeV
Pre_eta_jet = 2.5

# Tracks preselection cuts
Pre_PT_track = 0      # GeV
Pre_eta_track = 2.5

Delphes = "HL-LHC"
Energy_label = "14TeV"
BenchMark = "BM500"
process = "gg-hhj"

# --- Archivos de test (entrada de salt test) y salidas del clasificador (salida de salt test)
TEST_DIR = Path("/mnt/R5/gbarco/TrainingV01")
CKPT_DIR = Path("/home/gbarco/DiHiggsProject/OutTrainings/GN2Tracks_v01_20261105-T131007/ckpts")
N_EPOCH = "049"

# --- Corte en el clasificador: D = log(p_tau / p_qcd)
# GNN_WP: eficiencia de señal deseada; el corte en D se elige para que pase esa fracción de la señal.
# GNN_CUT: si no es None, se usa este valor de D directamente e ignora GNN_WP.
GNN_WP = 0.80
GNN_CUT = None

# nombre de cada muestra : (sufijo de los archivos de test, color)
muestras = {
    "LQ - gg > hhj": ("signal", "tab:blue"),
    "BG - pp > tthj": ("background", "tab:red"),
}

# ============================================================================
# Data Loading
# ============================================================================

print('Loading Data...')


def padded_to_jagged(array, valid):
    """(N, n_max) con relleno -> lista de largo variable por evento (como lo da uproot)."""
    counts = valid.sum(axis=1)
    return ak.unflatten(array[valid], counts)


def load_test_h5(sample):
    """Lee pp_output_test_<sample>.h5 y lo devuelve con los nombres de branch de Delphes."""
    with h5py.File(TEST_DIR / f"pp_output_test_{sample}.h5", "r") as h5:
        jets = h5["delphes_jets"][:]
        tracks = h5["tracks"][:]
        labels = h5["jets"]["flavour_label"][:]

    jet_valid = np.isfinite(jets["PT"])   # el dumper rellena con NaN los jets que no existen
    trk_valid = tracks["valid"]           # make_salt_h5 marcó los tracks reales con valid = True

    data = ak.zip(
        {
            "Jet.PT": padded_to_jagged(jets["PT"], jet_valid),
            "Jet.Eta": padded_to_jagged(jets["Eta"], jet_valid),
            "Jet.Phi": padded_to_jagged(jets["Phi"], jet_valid),
            "Jet.Mass": padded_to_jagged(jets["Mass"], jet_valid),
            "Jet.BTag": padded_to_jagged(jets["BTag"], jet_valid),
            "Jet.TauTag": padded_to_jagged(jets["TauTag"], jet_valid),
            "Track.PT": padded_to_jagged(tracks["PT"], trk_valid),
            "Track.Eta": padded_to_jagged(tracks["Eta"], trk_valid),
            "Track.Phi": padded_to_jagged(tracks["Phi"], trk_valid),
            # Track.Mass no está en el h5. Solo se usa para armar el cuadrivector en
            # f.vectors_buildup y ningún plot de tracks depende de ella.
            "Track.Mass": padded_to_jagged(np.zeros(tracks["PT"].shape, np.float32), trk_valid),
            "Track.D0": padded_to_jagged(tracks["D0"], trk_valid),
            "Track.DZ": padded_to_jagged(tracks["DZ"], trk_valid),
        },
        depth_limit=1,  # jets y tracks tienen distinta cantidad por evento: no combinarlos
    )
    return data, labels


def load_discriminant(sample):
    """Lee la salida de salt test y devuelve D = log(p_tau / p_qcd) por evento."""
    files = sorted(CKPT_DIR.glob(f"epoch={N_EPOCH}-*__test_{sample}*.h5"))
    if not files:
        raise FileNotFoundError(f"No encuentro la salida de salt test '{sample}' para epoch={N_EPOCH} en {CKPT_DIR}")
    with h5py.File(files[0], "r") as h5:
        j = h5["jets"]
        prefix = next(n[: -len("_phadtaujets")] for n in j.dtype.names if n.endswith("_phadtaujets"))
        p_tau = j.fields(f"{prefix}_phadtaujets")[:]
        p_qcd = j.fields(f"{prefix}_pqcdjets")[:]
        labels = j.fields("flavour_label")[:]
    print(f"  {sample}: predicciones de {files[0].name}")
    return np.log((p_tau + 1e-8) / (p_qcd + 1e-8)), labels


# 1) Test completo (antes del clasificador) y discriminante de cada evento
test_data, D = {}, {}
for name, (sample, _) in muestras.items():
    test_data[name], labels_in = load_test_h5(sample)
    D[name], labels_out = load_discriminant(sample)
    # la fila i de la salida de salt test es el evento i del test: lo verificamos
    if len(D[name]) != len(test_data[name]) or not np.array_equal(labels_in, labels_out):
        raise RuntimeError(f"{name}: el test.h5 y la salida de salt test no están alineados.")

# 2) Corte en el clasificador
signal_name = next(n for n, (s, _) in muestras.items() if s == "signal")
GNN_cut = GNN_CUT if GNN_CUT is not None else float(np.quantile(D[signal_name], 1 - GNN_WP))
print(f"Corte GNN: D > {GNN_cut:.3f}")

# 3) Las 4 muestras a comparar: cada una antes y después del clasificador
all_data, style, n_test = {}, {}, {}
for name, (sample, color) in muestras.items():
    passed = D[name] > GNN_cut
    all_data[name] = test_data[name]
    all_data[f"{name} (GNN)"] = test_data[name][passed]
    style[name] = dict(color=color, linestyle="-")
    style[f"{name} (GNN)"] = dict(color=color, linestyle="--")
    n_test[name] = n_test[f"{name} (GNN)"] = len(test_data[name])
    print(f"  {name}: {len(passed):,} eventos de test -> {passed.sum():,} pasan la GNN ({passed.mean():.1%})")

wp_tag = f"GNN_WP{int(round(GNN_WP * 100))}" if GNN_CUT is None else f"GNN_D{GNN_cut:.2f}"
data_to_analyse = f"{Delphes}_{Energy_label}_{BenchMark}_{process}_{wp_tag}"
Global_folder_path = f'{saving_path}/{data_to_analyse}'

f.print_sample_info(Delphes, Energy_label, BenchMark, process, data_to_analyse)

os.makedirs(Global_folder_path, exist_ok=True)

GNN_text = (fr"GNN: D > {GNN_cut:.2f}" + (f" ({GNN_WP:.0%} eff. señal)" if GNN_CUT is None else ""))


# ============================================================================
# Jets & Tracks construction
# ============================================================================

print('Building Jets and Tracks')

SubFolder_Jets = f"{Global_folder_path}/Jets/PT{Pre_PT_jet}_Eta{Pre_eta_jet}"
SubFolder_Tracks = f"{Global_folder_path}/Tracks/PT{Pre_PT_track}_Eta{Pre_eta_track}"

Data_text_ID_jets = ("BenchMark: " f"{BenchMark}" "\n"
                f"Process: {process}" "\n"
                fr"Delphes: {Delphes}" "\n"
                r"$\sqrt{s}$" f" = {Energy_label}" "\n"
                r"MET > 100 GeV" "\n"
                fr"$p_T$ > {Pre_PT_jet} GeV" "\n"
                rf"$|\eta| < {Pre_eta_jet}$" "\n"
                f"{GNN_text}")

Data_text_ID_tracks = ("BenchMark: " f"{BenchMark}" "\n"
                    f"Process: {process}" "\n"
                    fr"Delphes: {Delphes}" "\n"
                    r"$\sqrt{s}$" f" = {Energy_label}" "\n"
                    r"MET > 100 GeV" "\n"
                    fr"$p_T$ > {Pre_PT_track} GeV" "\n"
                    rf"$|\eta| < {Pre_eta_track}$" "\n"
                    f"{GNN_text}")


os.makedirs(SubFolder_Jets, exist_ok=True)
os.makedirs(SubFolder_Tracks, exist_ok=True)


def build_objects(data, Pre_PT_jet, Pre_eta_jet, Pre_PT_track, Pre_eta_track):
    # Igual que f.Selected_data pero sin el corte en MissingET.MET
    # (el corte en MET ya se aplicó al generar los h5).
    jets, tracks = {}, {}
    for key, d in data.items():
        j, t = f.vectors_buildup(d)
        jets[key] = j[f.preselection_cut(j, Pre_PT_jet, Pre_eta_jet)]
        tracks[key] = t[f.preselection_cut(t, Pre_PT_track, Pre_eta_track)]
    return jets, tracks

jets, tracks = build_objects(all_data, Pre_PT_jet, Pre_eta_jet, Pre_PT_track, Pre_eta_track)


# ============================================================================
# Normalizacion: pre y post GNN comparten denominador
# ============================================================================
# Cada muestra aparece dos veces en all_data: "<nombre>" (todo el test) y
# "<nombre> (GNN)" (los que pasan el clasificador). Para que la curva post-GNN
# muestre la eficiencia del corte en vez de reescalarse a la misma altura,
# las dos se dividen por el MISMO numero: el de la muestra pre-GNN.

GNN_SUFFIX = " (GNN)"


def base_of(key):
    """'LQ - gg > hhj (GNN)' -> 'LQ - gg > hhj'."""
    return key[: -len(GNN_SUFFIX)] if key.endswith(GNN_SUFFIX) else key


# denominadores, tomados siempre de la muestra pre-GNN
N_tracks_base = {key: int(ak.sum(ak.num(tracks[base_of(key)], axis=1))) for key in tracks}
N_events_base = {key: len(jets[base_of(key)]) for key in jets}


def peso(n, denom):
    """Array de pesos 1/denom, del largo de los datos."""
    return np.full(n, 1.0 / denom) if denom else np.zeros(n)


# ============================================================================
# ===                                                                      ===
# ===                         Track Analysis                               ===
# ===                                                                      ===
# ============================================================================

print(' ')
print('====================')
print('   Track Analysis   ')
print('====================')
print(' ')

# ============================================================================
# Eta, Phi and PT
# ============================================================================


print('Plotting Eta, Phi and PT Distributions of Tracks')

pt_data = {key: ak.to_numpy(ak.flatten(tracks[key].pt)) for key in tracks}
eta_data = {key: ak.to_numpy(ak.flatten(tracks[key].eta)) for key in tracks}
phi_data = {key: ak.to_numpy(ak.flatten(tracks[key].phi)) for key in tracks}

lim_inf_pt, lim_sup_pt, p_debajo_pt, p_dentro_pt, p_encima_pt = f.calculo_rangos(0, 100, list(pt_data.values()))
lim_inf_eta, lim_sup_eta, p_debajo_eta, p_dentro_eta, p_encima_eta = f.calculo_rangos(-5,5, list(eta_data.values()))
lim_inf_phi, lim_sup_phi, p_debajo_phi, p_dentro_phi, p_encima_phi = f.calculo_rangos(-3.2,3.2, list(phi_data.values()))

texto_pt = (f"Debajo de {lim_inf_pt} GeV : {p_debajo_pt:.2f}%\n"f"Dentro del rango : {p_dentro_pt:.2f}%\n"f"Encima de {lim_sup_pt} GeV : {p_encima_pt:.2f}%")
texto_eta = (f"Debajo de {lim_inf_eta}: {p_debajo_eta:.2f}%\n"f"Dentro del rango : {p_dentro_eta:.2f}%\n"f"Encima de {lim_sup_eta}: {p_encima_eta:.2f}%")
texto_phi = (f"Debajo de {lim_inf_phi}: {p_debajo_phi:.2f}%\n"f"Dentro del rango : {p_dentro_phi:.2f}%\n"f"Encima de {lim_sup_phi}: {p_encima_phi:.2f}%")

kw = dict(bins=50, histtype="step", linewidth=2)

fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=True)


for i, (key, pt) in enumerate(pt_data.items()):
        axes[0].hist(pt, range=(lim_inf_pt, lim_sup_pt), **kw, **style[key], label=f"{key}-Tracks",
                     weights=peso(len(pt), N_tracks_base[key]))

for j, (key, eta) in enumerate(eta_data.items()):
        axes[1].hist(eta, range=(lim_inf_eta, lim_sup_eta), **kw, **style[key], label=f"{key}-Tracks",
                     weights=peso(len(eta), N_tracks_base[key]))

for k, (key, phi) in enumerate(phi_data.items()):
        axes[2].hist(phi, range=(lim_inf_phi, lim_sup_phi), **kw, **style[key], label=f"{key}-Tracks",
                     weights=peso(len(phi), N_tracks_base[key]))

axes[0].text(0.98, 0.97,texto_pt,transform=axes[0].transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))
axes[0].set_ylabel("Fracción de tracks (pre-GNN)")
axes[0].set_xlabel("pT [GeV]")
axes[0].legend(loc = 'center right')

axes[1].text(0.98, 0.97,texto_eta,transform=axes[1].transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))
axes[1].set_xlabel(r"$\eta$")
axes[1].legend(loc = 'center right')

axes[2].text(0.98, 0.97,texto_phi,transform=axes[2].transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))
axes[2].xaxis.set_major_locator(MultipleLocator(np.pi / 2))
axes[2].xaxis.set_major_formatter(FuncFormatter(f.pi_formatter))
axes[2].set_xlabel(r"$\phi$")
axes[2].legend(loc = 'center right')

for ax in axes:
        ax.grid()
        ax.text(0.98, 0.82, Data_text_ID_tracks, transform=ax.transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")


plt.tight_layout()
plt.savefig(f"{SubFolder_Tracks}/Tracks_distributions.png", dpi=300)
plt.close()


# ============================================================================
# 2D Eta and Phi
# ============================================================================

print('Plotting 2D Distributions tracks eta and phi')


list_keys = list(all_data.keys())
rango = [[-4, 4], [-4, 4]]

fig, axes = plt.subplots(1, len(list_keys), figsize=(5.5*len(list_keys), 6),
                         sharex=True, sharey=True, squeeze=False)
axes = axes[0]

for ax, key in zip(axes, list_keys):
    x, y = eta_data[key], phi_data[key]
    h = ax.hist2d(x, y, bins=50, range=rango, cmap="magma", cmin=1)

    ax.set_facecolor("black")
    ax.set_title(key, fontsize=12)
    ax.set_xlabel(r"$\eta$", fontsize=13)
    ax.text(0.02, 0.97, f"N = {len(x)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="white")
    ax.text(0.97, 0.99, Data_text_ID_tracks, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="DejaVu Serif", color="white")

axes[0].set_ylabel(r"$\phi$", fontsize=13)

fig.colorbar(h[3], ax=axes, label="Tracks", fraction=0.025, pad=0.02)

plt.savefig(f"{SubFolder_Tracks}/2D-eta_phi_por_muestra.png", dpi=300)
plt.close()


# ============================================================================
# Vertex Parameters -- D0 and Dz
# ============================================================================

print('Plotting Vertex Parameters')

d0_data = {key: ak.to_numpy(ak.flatten(tracks[key].d0)) for key in tracks}
dz_data = {key: ak.to_numpy(ak.flatten(tracks[key].dz)) for key in tracks}

kw = dict(bins=50, histtype="step", linewidth=2)

lim_inf_d0, lim_sup_d0, p_debajo_d0, p_dentro_d0, p_encima_d0 = f.calculo_rangos(-6,6, list(d0_data.values()))
lim_inf_dz, lim_sup_dz, p_debajo_dz, p_dentro_dz, p_encima_dz = f.calculo_rangos(-30,30, list(dz_data.values()))

texto0 = (f"Debajo de {lim_inf_d0} mm : {p_debajo_d0:.2f}%\n"f"Dentro del rango : {p_dentro_d0:.2f}%\n"f"Encima de {lim_sup_d0} mm : {p_encima_d0:.2f}%")
texto1 = (f"Debajo de {lim_inf_dz} mm : {p_debajo_dz:.2f}%\n"f"Dentro del rango : {p_dentro_dz:.2f}%\n"f"Encima de {lim_sup_dz} mm : {p_encima_dz:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

axes[0].text(0.98, 0.97,texto0,transform=axes[0].transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))
axes[1].text(0.98, 0.97,texto1,transform=axes[1].transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))

for i, (key, d0) in enumerate(d0_data.items()):
        axes[0].hist(d0, range=(lim_inf_d0, lim_sup_d0), **kw, **style[key], label=f"{key}",
                     weights=peso(len(d0), N_tracks_base[key]))

for j, (key, dz) in enumerate(dz_data.items()):
        axes[1].hist(dz, range=(lim_inf_dz, lim_sup_dz), **kw, **style[key], label=f"{key}",
                     weights=peso(len(dz), N_tracks_base[key]))

axes[0].set_xlabel(r"$d_0$ [mm]")
axes[0].set_ylabel("Fracción de tracks (pre-GNN)")
axes[0].set_yscale('log')
axes[0].set_title('Transverse displacement (d0)')

axes[1].set_xlabel(r"$d_z$ [mm]")
axes[1].set_title('Longitudinal displacement (dz)')

for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(loc = 'upper left')
    ax.text(0.98, 0.8, Data_text_ID_tracks, transform=ax.transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")

plt.tight_layout()
plt.savefig(f"{SubFolder_Tracks}/Vertex_displacement.png", dpi=300)
plt.close()



# ============================================================================
# Tracks per Event
# ============================================================================

print('Plotting Number of Tracks per Event')

ntrk_data = {key: ak.to_numpy(ak.num(tracks[key], axis=1)) for key in tracks}

lim_inf_ntrk, lim_sup_ntrk, p_debajo_ntrk, p_dentro_ntrk, p_encima_ntrk = f.calculo_rangos(0,200, list(ntrk_data.values()))

texto0 = (f"Debajo de {lim_inf_ntrk}: {p_debajo_ntrk:.2f}%\n"f"Dentro del rango : {p_dentro_ntrk:.2f}%\n"f"Encima de {lim_sup_ntrk} : {p_encima_ntrk:.2f}%")

plt.figure(figsize=(8,6))

plt.text(0.98, 0.97,texto0,transform=plt.gca().transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))
plt.text(0.98, 0.8, Data_text_ID_tracks, transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")

# normalizado por los eventos pre-GNN: la curva post-GNN cae segun la eficiencia
for i, (key, ntrk) in enumerate(ntrk_data.items()):
        plt.hist(ntrk, bins=200, range=(lim_inf_ntrk, lim_sup_ntrk), histtype="step", linewidth=2,
                 **style[key], label=f"{key}", weights=peso(len(ntrk), N_events_base[key]))

plt.xlabel(r"Number of Tracks")
plt.ylabel("Fracción de eventos (pre-GNN)")
plt.grid()
plt.legend(loc = 'upper left')

plt.tight_layout()
plt.savefig(f"{SubFolder_Tracks}/Number_of_Tracks.png", dpi=300)
plt.close()

# ============================================================================
# ===                                                                      ===
# ===                           JETS ANALYSIS                              ===
# ===                                                                      ===
# ============================================================================

print(' ')
print('====================')
print('    Jet Analysis    ')
print('====================')
print(' ')

# ============================================================================
# Selection and Observables buildup
# ============================================================================

def build_taus_and_bjets(jets_input):

    taus, bjets, NEvents, NTaus, NBjets = {}, {}, {}, {}, {}
    for key, d in jets_input.items():
        taus[key], bjets[key], NEvents[key], NTaus[key], NBjets[key] = f.Taus_and_bjets(d)
    return taus, bjets, NEvents, NTaus, NBjets

def build_selected_pairs(taus, bjets):
    out = {}
    for key in taus:
        t1, t2, bb1, bb2, tc, bc = f.kinematic_hierarchical_order(taus[key], bjets[key])
        out[key] = {"tau1": t1, "tau2": t2, "b1": bb1, "b2": bb2,
                    "taus": tc, "bjets": bc}
    return out


def build_observables(sel):
    obs = {}
    for key, s in sel.items():
        t1, t2, bb1, bb2 = s["tau1"], s["tau2"], s["b1"], s["b2"]
        tup_tt, tup_bb, tup_hh = f.PT_and_IM_of_products(t1, t2, bb1, bb2)

        obs[key] = {
            "m_tautau": tup_tt[0],  "m_bb": tup_bb[0],  "m_hh": tup_hh[0],
            "pt_tautau": tup_tt[1], "pt_bb": tup_bb[1], "pt_hh": tup_hh[1],
            "dR_tautau": t1.deltaR(t2),
            "dR_bb":     bb1.deltaR(bb2),
            "dR_hh": (t1+t2).deltaR(bb1 + bb2)
        }
    return obs


print('Identifying Taus and Bjets')

taus, bjets, NEvents, NTaus, NBjets = build_taus_and_bjets(jets)    # Selecciono los Taus y Bs de cada Dataset

print('Selection of 2tau+2bjets events')

Selection_dictionary = build_selected_pairs(taus, bjets)            # Recorto a eventos con al menos 2 Taus y 2 Bs, y los ordeno por Pt

print('Calculating Observables')

Observables = build_observables(Selection_dictionary)               # Calculo los observables de cada par de Taus y Bs seleccionados

ncols = len(Observables)

# eventos que pasan 2tau+2b en la muestra PRE-GNN: denominador de los
# observables de pares, compartido por la curva pre y la post
N_sel_base = {key: len(Selection_dictionary[base_of(key)]["taus"]) for key in Selection_dictionary}


# ============================================================================
# Aceptancies
# ============================================================================

print('Calculating Aceptancies')

def doubletag_events(j):
    dt = (j['TauTag'] == 1) & (j['BTag'] == 1)
    n_dt = int(ak.sum(ak.num(j[dt], axis=1) > 0))
    n_ev = len(j)
    return n_dt, 100 * n_dt / n_ev if n_ev else 0.0

lineas = [Data_text_ID_jets, ""]
lineas.append(f"{'Muestra':<22} {'Test':>8} {'Pasan GNN':>10} {'Efic. GNN':>10} "
              f"{'Sel. 2t+2b':>11} {'Acept.':>8} {'Acept. tot.':>12} "
              f"{'Ev. doble tag':>14} {'% doble':>8}")
lineas.append("-" * 112)

for key in all_data:
    n_tst = n_test[key]          # eventos de test de esa muestra (antes de la GNN)
    n_ev = len(jets[key])        # eventos en esta muestra (= n_tst sin GNN, o los que pasan la GNN)
    n_sel = len(Selection_dictionary[key]["taus"])

    acc_gnn = 100 * n_ev / n_tst if n_tst else 0.0
    acc_sel = 100 * n_sel / n_ev if n_ev else 0.0
    acc_tot = 100 * n_sel / n_tst if n_tst else 0.0

    n_dt, pct_dt = doubletag_events(jets[key])

    lineas.append(f"{key:<22} {n_tst:>8} {n_ev:>10} {acc_gnn:>9.2f}% "
                  f"{n_sel:>11} {acc_sel:>7.2f}% {acc_tot:>11.2f}% "
                  f"{n_dt:>14} {pct_dt:>7.2f}%")

texto = "\n".join(lineas)


print(' ')
print('===============================================================================')
print('JETs Cutflow Summary:')
print(' ')
print(texto)
_ = (Path(SubFolder_Jets) / "cutflow.txt").write_text(texto + "\n")
print(' ')
print('===============================================================================')
print(' ')

# ============================================================================
# Taus and Bjets per event
# ============================================================================

print('Plotting Num of Taus and Bjets per event')

lim_inf_Ntaus, lim_sup_Ntaus, p_debajo_Ntaus, p_dentro_Ntaus, p_encima_Ntaus = f.calculo_rangos(0, 10, list(NTaus.values()))
lim_inf_Nbjets, lim_sup_Nbjets, p_debajo_Nbjets, p_dentro_Nbjets, p_encima_Nbjets = f.calculo_rangos(0, 10, list(NBjets.values()))

kw = dict(bins=10, histtype="step", linewidth=2)

fig, axes = plt.subplots(1, 2, figsize=(15,5), sharey=True)

axes[0].text(0.98, 0.97, Data_text_ID_jets, transform=axes[0].transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")

for key, arr in NTaus.items():
    a = ak.to_numpy(arr)
    axes[0].hist(a, range=(lim_inf_Ntaus, lim_sup_Ntaus), **kw, **style[key], label=key,
                 weights=peso(len(a), N_events_base[key]))

axes[0].set_xlabel(r"Number of Taus per event")
axes[0].set_ylabel("Fracción de eventos (pre-GNN)")
axes[0].legend(loc = "center right")
axes[0].grid(alpha=0.3)

for key, arr in NBjets.items():
    a = ak.to_numpy(arr)
    axes[1].hist(a, range=(lim_inf_Nbjets, lim_sup_Nbjets), **kw, **style[key], label=key,
                 weights=peso(len(a), N_events_base[key]))
axes[1].set_xlabel(r"Number of b-jets per event")
axes[1].legend(loc = "center right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SubFolder_Jets}/Taus&Bjets_in_events.png", dpi=300)
plt.close()



# ============================================================================
# PT Distribution
# ============================================================================

print('Plotting PT Distribution of single particles')

tau1_pt_data = {key: ak.to_numpy(Selection_dictionary[key]["tau1"].pt) for key in Selection_dictionary}
tau2_pt_data = {key: ak.to_numpy(Selection_dictionary[key]["tau2"].pt) for key in Selection_dictionary}
bjet1_pt_data = {key: ak.to_numpy(Selection_dictionary[key]["b1"].pt) for key in Selection_dictionary}
bjet2_pt_data = {key: ak.to_numpy(Selection_dictionary[key]["b2"].pt) for key in Selection_dictionary}

pt_all = {key: [tau1_pt_data[key],tau2_pt_data[key],bjet1_pt_data[key],bjet2_pt_data[key]]
          for key in Selection_dictionary}

rangos = {key: f.calculo_rangos(0, 500, pt_all[key]) for key in pt_all}

textos_rangos_pt = {}
for key, (li, ls, p_deb, p_den, p_enc) in rangos.items():
    textos_rangos_pt[key] = (f"Debajo de {li} GeV: {p_deb:.2f}%\n"
                   f"Dentro del rango: {p_den:.2f}%\n"
                   f"Encima de {ls} GeV: {p_enc:.2f}%")

labels = [r"$\tau_{1, \text{vis}}$", r"$\tau_{2, \text{vis}}$", r"$b\text{-}jet_1$", r"$b\text{-}jet_2$"]
kw = dict(bins=50, range=(0,500), histtype="step", linewidth=2)

fig, axes = plt.subplots(1, len(Observables.keys()), figsize=(5.5*ncols,5), sharey=True, squeeze=False)
axes = axes[0]
for i, (key) in enumerate(Selection_dictionary.keys()):
        w = peso(len(tau1_pt_data[key]), N_sel_base[key])
        axes[i].hist(tau1_pt_data[key], **kw, weights=w, label=labels[0])
        axes[i].hist(tau2_pt_data[key], **kw, weights=w, label=labels[1])
        axes[i].hist(bjet1_pt_data[key], **kw, weights=w, label=labels[2])
        axes[i].hist(bjet2_pt_data[key], **kw, weights=w, label=labels[3])
        axes[i].set_title(fr"{key}")
        axes[i].set_xlabel("pT [GeV]")
        axes[i].grid()
        axes[i].legend(loc = 'lower right')
        axes[i].text(0.98, 0.97, textos_rangos_pt[key], transform=axes[i].transAxes,
                ha="right", va="top", bbox=dict(facecolor="white", alpha=0.85))

axes[0].text(0.98, 0.8, Data_text_ID_jets, transform=axes[0].transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")
axes[0].set_ylabel("Fracción de eventos (pre-GNN)")

plt.tight_layout()
plt.savefig(f"{SubFolder_Jets}/pt_distribution.png", dpi=300)
plt.close()



# ============================================================================
# Pairs Analysis - PT and IM
# ============================================================================

print('Plotting PT and IM Distribution of pairs')

filas = [
    ("pt_tautau", "pt_bb", (0, 600),  50, r"$p_T$ [GeV]",
     r"$p_T(\tau_{\mathrm{vis}}\tau_{\mathrm{vis}})$", r"$p_T(b\bar{b})$"),
    ("m_tautau",  "m_bb",  (15, 500), 30, r"Invariant Mass [GeV]",
     r"$M(\tau_{\mathrm{vis}}\tau_{\mathrm{vis}})$", r"$M(b\bar{b})$"),
]

fig, axes = plt.subplots(2, len(Observables.keys()), figsize=(5.5*ncols, 10), sharey='row',squeeze=False)
for r, (k_tt, k_bb, (lo, hi), nbins, xlabel, lab_tt, lab_bb) in enumerate(filas):
    for c, key in enumerate(Observables):
        ax = axes[r, c]
        li, ls, p_deb, p_den, p_enc = f.calculo_rangos(lo, hi, [Observables[key][k_tt], Observables[key][k_bb]])

        a_tt = ak.to_numpy(Observables[key][k_tt])
        a_bb = ak.to_numpy(Observables[key][k_bb])
        w = peso(len(a_tt), N_sel_base[key])

        kw = dict(bins=nbins, range=(li, ls), histtype="step", linewidth=2, weights=w)
        ax.hist(a_tt, label=lab_tt, **kw)
        ax.hist(a_bb, label=lab_bb, **kw)

        ax.set_xlabel(xlabel)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

        txt = (f"Debajo de {li} GeV: {p_deb:.2f}%\n"
               f"Dentro del rango: {p_den:.2f}%\n"
               f"Encima de {ls} GeV: {p_enc:.2f}%")
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        if r == 0:
            ax.set_title(fr'{key}')

axes[0, 0].set_ylabel("Fracción de eventos (pre-GNN)")
axes[1, 0].set_ylabel("Fracción de eventos (pre-GNN)")

axes[0, 0].text(0.98, 0.80, Data_text_ID_jets, transform=axes[0, 0].transAxes, ha="right", va="top",
         fontsize=9, family="DejaVu Serif")

plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(f"{SubFolder_Jets}/pT_and_IM_pairs.png", dpi=300)
plt.close()


# ============================================================================
# Pairs IM - antes vs después de la GNN (nuevo)
# ============================================================================

print('Plotting IM of pairs before/after GNN')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (k, xlabel, (lo, hi)) in zip(axes, [
        ("m_tautau", r"$M(\tau_{\mathrm{vis}}\tau_{\mathrm{vis}})$ [GeV]", (15, 500)),
        ("m_bb", r"$M(b\bar{b})$ [GeV]", (15, 500))]):
    for key in Observables:
        a = ak.to_numpy(Observables[key][k])
        ax.hist(a, bins=30, range=(lo, hi), weights=peso(len(a), N_sel_base[key]),
                histtype="step", linewidth=2, **style[key],
                label=f"{key} (N = {len(a)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fracción de eventos (pre-GNN)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[1].text(0.98, 0.70, Data_text_ID_jets, transform=axes[1].transAxes,
             ha="right", va="top", fontsize=9, family="DejaVu Serif")
plt.tight_layout()
plt.savefig(f"{SubFolder_Jets}/IM_pairs_GNN_comparison.png", dpi=300)
plt.close()


# ============================================================================
# Decayed Products IM
# ============================================================================


print('Plotting IM Distribution of (tatabb)')


Higgs_mass = {key: ak.to_numpy(Observables[key]["m_hh"]) for key in Observables}

lim_inf_mhh, lim_sup_mhh, p_debajo_mhh, p_dentro_mhh, p_encima_mhh = f.calculo_rangos(130,1500, list(Higgs_mass.values()))

texto_Higgs = (f"Debajo de {lim_inf_mhh} GeV : {p_debajo_mhh:.2f}%\n"f"Dentro del rango : {p_dentro_mhh:.2f}%\n"f"Encima de {lim_sup_mhh} GeV : {p_encima_mhh:.2f}%")

plt.figure(figsize=(8,6))

plt.text(0.98, 0.84, Data_text_ID_jets, transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")

plt.text(0.98, 0.98,texto_Higgs,transform=plt.gca().transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))

for key, arr in Higgs_mass.items():
    a = ak.to_numpy(arr)
    plt.hist(a, bins=40, range=(lim_inf_mhh, lim_sup_mhh), histtype="step", linewidth=2,
             **style[key], label=fr'{key}', weights=peso(len(a), N_sel_base[key]))

plt.xlabel(r"$M$ [GeV]")
plt.ylabel("Fracción de eventos (pre-GNN)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SubFolder_Jets}/InvariantMass_distribution.png", dpi=300)
plt.close()




# ============================================================================
# Angular Analysis
# ============================================================================

print('Plotting Angular Distributions')

DeltaR_TT = {key: ak.to_numpy(Observables[key]["dR_tautau"]) for key in Observables}
DeltaR_BB = {key: ak.to_numpy(Observables[key]["dR_bb"]) for key in Observables}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6), sharey=True)

ax1.text(0.98, 0.80, Data_text_ID_jets, transform=ax1.transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")

ax2.text(0.98, 0.80, Data_text_ID_jets, transform=ax2.transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")

kw = dict(bins=50,range=(0,6),histtype="step", linewidth=2)

for key, arr in DeltaR_TT.items():
    ax1.hist(arr, **kw, **style[key], label=fr"{key}",
             weights=peso(len(arr), N_sel_base[key]))

ax1.set_xlabel(r"$\Delta R(\tau_{\text{vis}},\tau_{\text{vis}})$", fontsize=14)
ax1.set_ylabel("Fracción de eventos (pre-GNN)", fontsize=14)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=10)

for key, arr in DeltaR_BB.items():
    ax2.hist(arr, **kw, **style[key], label=fr"{key}",
             weights=peso(len(arr), N_sel_base[key]))

ax2.set_xlabel(r"$\Delta R(b,\bar{b})$", fontsize=14)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f"{SubFolder_Jets}/DeltaR_distribution.png", dpi=300)
plt.close()


# ============================================================================
# 2D Analysis tau-tau and b-b
# ============================================================================

list_keys = list(DeltaR_TT.keys())
rango = [[0, 6], [0, 6]]
norm  = LogNorm(vmin=1, vmax=50)

fig, axes = plt.subplots(1, len(list_keys), figsize=(5.5*len(list_keys), 6),
                         sharex=True, sharey=True, squeeze=False)
axes = axes[0]

for ax, key in zip(axes, list_keys):
    x = ak.to_numpy(DeltaR_TT[key])
    y = ak.to_numpy(DeltaR_BB[key])
    h = ax.hist2d(x, y, bins=50, range=rango, cmap="magma", norm = norm, cmin=1)

    ax.set_facecolor("black")
    ax.set_title(key, fontsize=12)
    ax.set_xlabel(r"$\Delta R(\tau_{\rm vis},\tau_{\rm vis})$", fontsize=13)
    ax.text(0.02, 0.97, f"N = {len(x)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="white")
    ax.text(0.97, 0.99, Data_text_ID_jets, ha="right", va="top", transform=ax.transAxes,
         fontsize=9, family="DejaVu Serif", color = 'white')

axes[0].set_ylabel(r"$\Delta R(b,\bar{b})$", fontsize=13)

fig.colorbar(h[3], ax=axes, label="Eventos", fraction=0.025, pad=0.02)

plt.savefig(f"{SubFolder_Jets}/2D-DeltaR_por_muestra.png", dpi=300)
plt.close()



# ============================================================================
# Higgses Analysis
# ============================================================================

print(' ')
print('====================')
print('   Higgs Analysis   ')
print('====================')
print(' ')


# ============================================================================
# 2D PT Distribution of H(tautau) and H(bb)
# ============================================================================

print('Plotting PT correlation between leptonic and hadronic H')


pT_h_tautau = {key: ak.to_numpy(Observables[key]["pt_tautau"]) for key in Observables}
pt_h_bb = {key: ak.to_numpy(Observables[key]["pt_bb"]) for key in Observables}

rango = [[0, 500], [0, 500]]

fig, axes = plt.subplots(1, len(list_keys), figsize=(5.5*len(list_keys), 6),
                         sharex=True, sharey=True, squeeze=False)
axes = axes[0]

for ax, key in zip(axes, list_keys):
    x = ak.to_numpy(pT_h_tautau[key])
    y = ak.to_numpy(pt_h_bb[key])
    h = ax.hist2d(x, y, bins=50, range=rango, cmap="magma", cmin=1)


    ax.axline((0, 0), slope=1, color="red", linestyle="--", linewidth=1)
    ax.set_facecolor("black")
    ax.set_title(key, fontsize=12)
    ax.set_xlabel(r"$p_T(\tau \tau)$ [GeV]", fontsize=13)
    ax.text(0.02, 0.97, f"N = {len(x)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="white")
    ax.text(0.97, 0.99, Data_text_ID_jets, ha="right", va="top", transform=ax.transAxes,
         fontsize=9, family="DejaVu Serif", color = 'white')

axes[0].set_ylabel(r"$p_T(bb)$ [GeV]", fontsize=13)

fig.colorbar(h[3], ax=axes, label="Eventos", fraction=0.025, pad=0.02)

fig.savefig(f"{SubFolder_Jets}/2D-PT_hpair_distribution.png", dpi=300)
plt.close()


# ============================================================================
# DR Distribution between H(tautau) and H(bb)
# ============================================================================

print('Plotting DR distribution of H(tautau) and H(bb)')

DR_of_Higgses = {key: ak.to_numpy(Observables[key]["dR_hh"]) for key in Observables}

lim_inf_dRhh, lim_sup_dRhh, p_debajo_dRhh, p_dentro_dRhh, p_encima_dRhh = f.calculo_rangos(0, 6, list(DR_of_Higgses.values()))

texto_Higgs = (f"Debajo de {lim_inf_dRhh} : {p_debajo_dRhh:.2f}%\n"f"Dentro del rango : {p_dentro_dRhh:.2f}%\n"f"Encima de {lim_sup_dRhh} : {p_encima_dRhh:.2f}%")

plt.figure(figsize=(8,6))

plt.text(0.98, 0.84, Data_text_ID_jets, transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=10, family="DejaVu Serif")

plt.text(0.98, 0.98,texto_Higgs,transform=plt.gca().transAxes,ha="right",va="top",bbox=dict(facecolor="white", alpha=0.85))

for key, arr in DR_of_Higgses.items():
    a = ak.to_numpy(arr)
    plt.hist(a, bins=40, range=(lim_inf_dRhh, lim_sup_dRhh), histtype="step", linewidth=2,
             **style[key], label=fr'{key}', weights=peso(len(a), N_sel_base[key]))

plt.xlabel(r"$\Delta R(h_{\tau \tau}, h_{bb})$")
plt.ylabel("Fracción de eventos (pre-GNN)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SubFolder_Jets}/DR_Htautau_Hbb.png", dpi=300)
plt.close()

print(f"\nFiguras en {Global_folder_path}")
