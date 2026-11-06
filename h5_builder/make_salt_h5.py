#!/usr/bin/env python3
"""
make_salt_h5.py  (versión simple)

Toma los .h5 de señal y de fondo generados por h5_dumper.py (cada uno con los
datasets de Tracks y de Jets) y arma:

  pp_output_train.h5            -> tracks + label          (sin Jets)
  pp_output_val.h5              -> tracks + label          (sin Jets)
  pp_output_test_signal.h5      -> tracks + label + Jets
  pp_output_test_background.h5  -> tracks + label + Jets
  norm_dict.yaml, class_dict.yaml  (calculados solo con train)

Uso:
  python make_salt_h5.py --signal 'sig_part*.h5' --background 'bkg_part*.h5' --outdir salida/
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# NOMBRES DE LOS DATASETS
# ---------------------------------------------------------------------------
TRACKS_IN = "tracks"        # nombre del dataset de tracks en la salida de h5_dumper
JETS_IN = "Jets"            # nombre del dataset de Jets de Delphes en la salida de h5_dumper

TRACKS_OUT = "tracks"       # así lo lee Salt (input_map: tracks: tracks)
LABEL_OUT = "jets"          # así lo lee Salt (global_object: jets, labels: jets: flavour_label)
JETS_OUT = "delphes_jets"   # los Jets de Delphes se guardan con otro nombre para no
                            # chocar con el dataset 'jets' que usa Salt para el label

SIGNAL_LABEL = 0            # índice 0 de class_names -> hadtaujets
BACKGROUND_LABEL = 1        # índice 1 de class_names -> qcdjets


# ---------------------------------------------------------------------------
# PASO 1: leer todas las partes de una clase y juntarlas en un solo array
# ---------------------------------------------------------------------------
def read_class(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Ningún archivo coincide con {pattern}")

    tracks_list = []
    jets_list = []
    for name in files:
        with h5py.File(name, "r") as f:
            tracks_list.append(f[TRACKS_IN][:])
            jets_list.append(f[JETS_IN][:])
        print(f"  leído {name}: {len(tracks_list[-1]):,} eventos")

    tracks = np.concatenate(tracks_list)
    jets = np.concatenate(jets_list)
    return tracks, jets


# ---------------------------------------------------------------------------
# PASO 2: preparar los tracks para Salt
#   - agregar el campo 'valid' (True = track real, False = relleno)
#   - reemplazar los NaN del relleno por 0
#   - ordenar los tracks de cada evento por PT, de mayor a menor
# ---------------------------------------------------------------------------
def prepare_tracks(tracks):
    # 2a. nuevo array con los mismos campos + 'valid'
    new_dtype = tracks.dtype.descr + [("valid", "?")]
    out = np.zeros(tracks.shape, dtype=new_dtype)
    for field in tracks.dtype.names:
        out[field] = tracks[field]

    # 2b. el dumper rellena con NaN los lugares sin track
    out["valid"] = np.isfinite(tracks["PT"])

    # 2c. NaN -> 0 en todas las variables
    for field in tracks.dtype.names:
        column = out[field]
        column[~np.isfinite(column)] = 0

    # 2d. ordenar por PT descendente (el relleno queda al final)
    pt_for_sorting = np.where(out["valid"], out["PT"], -np.inf)
    order = np.argsort(-pt_for_sorting, axis=1)
    out = np.take_along_axis(out, order, axis=1)
    return out


# ---------------------------------------------------------------------------
# PASO 3: crear el label de cada evento (un número por evento)
# ---------------------------------------------------------------------------
def make_labels(n_events, label_value):
    labels = np.zeros(n_events, dtype=[("flavour_label", "i4")])
    labels["flavour_label"] = label_value
    return labels


# ---------------------------------------------------------------------------
# PASO 4: separar los índices de eventos en train / val / test
# ---------------------------------------------------------------------------
def split_indices(n_events, frac_train, frac_val, rng):
    shuffled = rng.permutation(n_events)       # índices 0..N-1 en orden aleatorio
    n_train = int(frac_train * n_events)
    n_val = int(frac_val * n_events)
    idx_train = shuffled[:n_train]
    idx_val = shuffled[n_train:n_train + n_val]
    idx_test = shuffled[n_train + n_val:]       # el resto
    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------------
# PASO 5: juntar señal y fondo y mezclarlos (para train y val)
# ---------------------------------------------------------------------------
def merge_and_shuffle(tracks_sig, labels_sig, tracks_bkg, labels_bkg, rng):
    tracks = np.concatenate([tracks_sig, tracks_bkg])
    labels = np.concatenate([labels_sig, labels_bkg])
    order = rng.permutation(len(labels))       # MISMO orden para tracks y labels
    return tracks[order], labels[order]


# ---------------------------------------------------------------------------
# PASO 6: escribir un archivo .h5
# ---------------------------------------------------------------------------
def write_h5(path, datasets):
    with h5py.File(path, "w") as f:
        for name, array in datasets.items():
            f.create_dataset(name, data=array, compression="lzf")
    print(f"  escrito {path}: " + ", ".join(f"{k} {v.shape}" for k, v in datasets.items()))


# ---------------------------------------------------------------------------
# PASO 7: norm_dict (media y desvío de cada variable, solo tracks válidos de train)
# ---------------------------------------------------------------------------
def make_norm_dict(tracks_train):
    valid = tracks_train["valid"]
    norm = {}
    for field in tracks_train.dtype.names:
        if field == "valid":
            continue
        values = tracks_train[field][valid].astype(np.float64)
        std = float(values.std())
        if std == 0:
            std = 1.0  # Salt no acepta std = 0
        norm[field] = {"mean": float(values.mean()), "std": std}
    return {TRACKS_OUT: norm}


# ---------------------------------------------------------------------------
# PASO 8: class_dict (pesos de clase, solo con train)
# ---------------------------------------------------------------------------
def make_class_dict(labels_train):
    n_sig = int(np.sum(labels_train["flavour_label"] == SIGNAL_LABEL))
    n_bkg = int(np.sum(labels_train["flavour_label"] == BACKGROUND_LABEL))
    n_tot = n_sig + n_bkg
    weights = [n_tot / (2 * n_sig), n_tot / (2 * n_bkg)]  # orden: [señal, fondo]
    return {LABEL_OUT: {"flavour_label": weights}}


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", required=True, help="patrón de los .h5 de señal, entre comillas")
    parser.add_argument("--background", required=True, help="patrón de los .h5 de fondo, entre comillas")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--frac-train", type=float, default=0.8)
    parser.add_argument("--frac-val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # PASO 1
    print("Leyendo señal...")
    tracks_sig, jets_sig = read_class(args.signal)
    print("Leyendo fondo...")
    tracks_bkg, jets_bkg = read_class(args.background)

    # PASO 2
    tracks_sig = prepare_tracks(tracks_sig)
    tracks_bkg = prepare_tracks(tracks_bkg)

    # PASO 3
    labels_sig = make_labels(len(tracks_sig), SIGNAL_LABEL)
    labels_bkg = make_labels(len(tracks_bkg), BACKGROUND_LABEL)

    # PASO 4 (cada clase por separado)
    sig_train, sig_val, sig_test = split_indices(len(tracks_sig), args.frac_train, args.frac_val, rng)
    bkg_train, bkg_val, bkg_test = split_indices(len(tracks_bkg), args.frac_train, args.frac_val, rng)

    # PASO 5
    tracks_train, labels_train = merge_and_shuffle(
        tracks_sig[sig_train], labels_sig[sig_train], tracks_bkg[bkg_train], labels_bkg[bkg_train], rng
    )
    tracks_val, labels_val = merge_and_shuffle(
        tracks_sig[sig_val], labels_sig[sig_val], tracks_bkg[bkg_val], labels_bkg[bkg_val], rng
    )

    # Los test se ordenan por índice para que sigan el orden original del dumper
    sig_test = np.sort(sig_test)
    bkg_test = np.sort(bkg_test)

    # PASO 6
    print("Escribiendo archivos...")
    write_h5(outdir / "pp_output_train.h5", {LABEL_OUT: labels_train, TRACKS_OUT: tracks_train})
    write_h5(outdir / "pp_output_val.h5", {LABEL_OUT: labels_val, TRACKS_OUT: tracks_val})
    write_h5(
        outdir / "pp_output_test_signal.h5",
        {LABEL_OUT: labels_sig[sig_test], TRACKS_OUT: tracks_sig[sig_test], JETS_OUT: jets_sig[sig_test]},
    )
    write_h5(
        outdir / "pp_output_test_background.h5",
        {LABEL_OUT: labels_bkg[bkg_test], TRACKS_OUT: tracks_bkg[bkg_test], JETS_OUT: jets_bkg[bkg_test]},
    )

    # PASOS 7 y 8
    with open(outdir / "norm_dict.yaml", "w") as f:
        yaml.safe_dump(make_norm_dict(tracks_train), f, sort_keys=False)
    with open(outdir / "class_dict.yaml", "w") as f:
        yaml.safe_dump(make_class_dict(labels_train), f)

    # Resumen
    print("\nEventos       señal      fondo")
    print(f"  train  {len(sig_train):>10,} {len(bkg_train):>10,}")
    print(f"  val    {len(sig_val):>10,} {len(bkg_val):>10,}")
    print(f"  test   {len(sig_test):>10,} {len(bkg_test):>10,}")


if __name__ == "__main__":
    main()
