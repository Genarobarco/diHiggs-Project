"""
Distribuciones de las variables de tracks (PT, Eta, Phi, D0, DZ y número de tracks)
para señal y fondo, antes y después de cortar en el discriminante de la red.

Cómo vincula tracks y predicciones:
  - Los tracks están en los archivos de test de entrada (pp_output_test_signal.h5 / _background.h5).
  - Las predicciones están en la salida de `salt test` (<ckpt>__test_signal.h5 / _background.h5).
  - `salt test` no baraja: la fila i de la salida es el evento i de la entrada. El script verifica
    que coincidan el número de eventos y los labels.

El corte se define por eficiencia de señal (working point): para WP = 0.7 se elige el valor de D
tal que pasa el 70 % de la señal de test, y ese mismo corte se aplica al fondo.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------------------------
# Configuración
DATA_DIR = Path("/mnt/R5/gbarco/BM500_gg-hh_pp-tthj_h5s_trainingfiles")  # salida de make_salt_h5.py
RUN_DIR = Path("/home/gbarco/DiHiggsProject/OutTrainings/GN2Tracks_v01_20261104-T200117")
N_EPOCH = "047"

WPS = [0.7]  # eficiencias de señal a estudiar (una figura por variable y WP)
OUT_DIR = Path("plots_after_cut")
EPS = 1e-8

# variable: (bins, etiqueta del eje x, escala log en y)
VARIABLES = {
    "PT": (np.linspace(0, 100, 51), "Track $p_T$ [GeV]", True),
    "Eta": (np.linspace(-4, 4, 41), "Track $\\eta$", False),
    "Phi": (np.linspace(-np.pi, np.pi, 41), "Track $\\phi$", False),
    "D0": (np.linspace(-1, 1, 51), "Track $d_0$ [mm]", True),
    "DZ": (np.linspace(-20, 20, 51), "Track $d_z$ [mm]", True),
}
N_TRACKS_BINS = np.arange(0, 101, 2)
LEADING_ONLY = False  # True: usar solo el track de mayor PT de cada evento (los tracks ya están ordenados)


# ----------------------------------------------------------------------------------------------
# Lectura
def find_pred_file(sample: str) -> Path:
    m = sorted((RUN_DIR / "ckpts").glob(f"epoch={N_EPOCH}-*__test_{sample}*.h5"))
    if not m:
        raise FileNotFoundError(f"No encuentro la salida de salt test '{sample}' para epoch={N_EPOCH}")
    return m[0]


def load_sample(sample: str):
    """Devuelve (D por evento, tracks estructurados (N, n_rows), label)."""
    pred_file = find_pred_file(sample)
    in_file = DATA_DIR / f"pp_output_test_{sample}.h5"

    with h5py.File(pred_file, "r") as f:
        jets = f["jets"]
        prefix = next(n[: -len("_phadtaujets")] for n in jets.dtype.names if n.endswith("_phadtaujets"))
        p_tau = jets.fields(f"{prefix}_phadtaujets")[:]
        p_qcd = jets.fields(f"{prefix}_pqcdjets")[:]
        lab_pred = jets.fields("flavour_label")[:]

    with h5py.File(in_file, "r") as f:
        tracks = f["tracks"][:]
        lab_in = f["jets"].fields("flavour_label")[:]

    if len(tracks) != len(p_tau):
        raise RuntimeError(
            f"{sample}: la entrada tiene {len(tracks)} eventos y la salida {len(p_tau)}. "
            "¿Corriste salt test con num_test: -1 sobre este mismo archivo?"
        )
    if not np.array_equal(lab_in, lab_pred):
        raise RuntimeError(f"{sample}: los labels de entrada y salida no coinciden fila a fila.")

    d = np.log((p_tau + EPS) / (p_qcd + EPS))
    print(f"{sample:10s}: {len(d):,} eventos  ({pred_file.name})")
    return d, tracks


def track_values(tracks, mask_events, var):
    """Valores de 'var' de los tracks válidos de los eventos seleccionados."""
    t = tracks[mask_events]
    if LEADING_ONLY:
        t = t[:, :1]
    return t[var][t["valid"]]


# ----------------------------------------------------------------------------------------------
# Plots
def plot_var(ax, curves, bins, xlabel, logy):
    """curves: lista de (valores, n_eventos, color, estilo, etiqueta)."""
    for values, n_ev, color, ls, label in curves:
        label = f"{label} ({n_ev:,} ev.)"
        if len(values) == 0:
            ax.plot([], [], color=color, ls=ls, label=label + " - vacío")
            continue
        ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.8, color=color, linestyle=ls, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalizado a 1")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=9, frameon=False)


def curves_for(vals_sig_all, vals_sig_pass, vals_bkg_all, vals_bkg_pass, pass_sig, pass_bkg):
    return [
        (vals_sig_all, len(pass_sig), "tab:blue", "-", "Señal (todo)"),
        (vals_sig_pass, int(pass_sig.sum()), "tab:blue", "--", "Señal (pasa corte)"),
        (vals_bkg_all, len(pass_bkg), "tab:red", "-", "Fondo (todo)"),
        (vals_bkg_pass, int(pass_bkg.sum()), "tab:red", "--", "Fondo (pasa corte)"),
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d_sig, tr_sig = load_sample("signal")
    d_bkg, tr_bkg = load_sample("background")

    for wp in WPS:
        cut = np.quantile(d_sig, 1 - wp)  # pasa una fracción wp de la señal
        pass_sig = d_sig > cut
        pass_bkg = d_bkg > cut
        eff_b = pass_bkg.mean()
        print(
            f"\nWP {wp:.0%}: corte D > {cut:.3f} | eff señal = {pass_sig.mean():.3f} | "
            f"eff fondo = {eff_b:.4f} (rechazo {'inf' if eff_b == 0 else f'{1 / eff_b:.2f}'})"
        )
        tag = f"WP{int(round(wp * 100))}"
        title = f"Corte en D para {wp:.0%} de eficiencia de señal (D > {cut:.2f})"
        all_s = np.ones(len(d_sig), bool)
        all_b = np.ones(len(d_bkg), bool)

        for var, (bins, xlabel, logy) in VARIABLES.items():
            fig, ax = plt.subplots(figsize=(6.5, 5))
            plot_var(
                ax,
                curves_for(
                    track_values(tr_sig, all_s, var),
                    track_values(tr_sig, pass_sig, var),
                    track_values(tr_bkg, all_b, var),
                    track_values(tr_bkg, pass_bkg, var),
                    pass_sig,
                    pass_bkg,
                ),
                bins,
                xlabel + (" (track líder)" if LEADING_ONLY else ""),
                logy,
            )
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{var}_{tag}.png", dpi=150)
            plt.close(fig)

        # número de tracks por evento
        fig, ax = plt.subplots(figsize=(6.5, 5))
        n_s, n_b = tr_sig["valid"].sum(axis=1), tr_bkg["valid"].sum(axis=1)
        plot_var(ax, curves_for(n_s, n_s[pass_sig], n_b, n_b[pass_bkg], pass_sig, pass_bkg), N_TRACKS_BINS, "Número de tracks por evento", False)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"nTracks_{tag}.png", dpi=150)
        plt.close(fig)

        # distribución del discriminante con el corte marcado
        fig, ax = plt.subplots(figsize=(6.5, 5))
        bins = np.linspace(min(d_sig.min(), d_bkg.min()), max(d_sig.max(), d_bkg.max()), 61)
        ax.hist(d_sig, bins=bins, histtype="step", density=True, lw=1.8, color="tab:blue", label="Señal")
        ax.hist(d_bkg, bins=bins, histtype="step", density=True, lw=1.8, color="tab:red", label="Fondo")
        ax.axvline(cut, color="k", ls=":", label=f"Corte {tag}")
        ax.set_xlabel("$D = \\log(p_{\\tau} / p_{QCD})$")
        ax.set_ylabel("Normalizado a 1")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"discriminant_{tag}.png", dpi=150)
        plt.close(fig)

    print(f"\nFiguras en {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
