#!/usr/bin/env python3
"""
make_salt_h5.py  (versión unificada: varias rutas de señal + varios fondos)

Toma los .h5 de señal y de fondo generados por h5_dumper.py (cada uno con los
datasets de Tracks y de Jets) y arma:

  pp_output_train.h5            -> tracks + label          (sin Jets)
  pp_output_val.h5              -> tracks + label          (sin Jets)
  pp_output_test_signal.h5      -> tracks + label + Jets + process
  pp_output_test_background.h5  -> tracks + label + Jets + process
  norm_dict.yaml, class_dict.yaml  (calculados solo con train)
  split_summary.yaml             -> eventos por proceso, fracciones y eventos
                                    de cada proceso en train / val / test

Uso:
  python make_salt_h5.py \
      --signal 'sig_carpetaA/*.h5' 'sig_carpetaB/*.h5' \
      --background 'ttH/*.h5' \
      --background 'ttZ/*.h5' \
      --background 'ttW_v1/*.h5' 'ttW_v2/*.h5' \
      --outdir salida/

  --signal      : una o más rutas/patrones. TODAS se juntan como una única señal.
  --background  : se repite una vez por proceso de fondo. Cada aparición es un
                  proceso distinto y puede llevar una o más rutas/patrones
                  (todas las rutas de una misma aparición son el mismo proceso).

  Una ruta puede ser un patrón entre comillas ('carpeta/*.h5'), un archivo o una
  carpeta (en ese caso se toman todos los .h5 de la carpeta).

Por defecto se balancea: se toma la misma cantidad de eventos de cada proceso de
fondo (n_keep = la del menor) y de señal se toman (cantidad de fondos) x n_keep
eventos. Si la señal no alcanza, se baja n_keep. Con --no-balance se usan todos. En los test se agrega el dataset 'process'
(0 = señal, 1, 2, 3... = cada fondo, en el orden en que se pasan los --background).
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

SIGNAL_LABEL = 0            # índice 0 de class_names -> señal
BACKGROUND_LABEL = 1        # índice 1 de class_names -> fondo


# ---------------------------------------------------------------------------
# PASO 0: convertir una lista de rutas/patrones en una lista de archivos
# ---------------------------------------------------------------------------
def find_files(patterns):
    files = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if path.is_dir():                         # carpeta -> todos sus .h5
            matches = glob.glob(str(path / "*.h5"))
        else:                                     # patrón o archivo
            matches = glob.glob(str(path))
        if not matches:
            raise FileNotFoundError(f"Ningún archivo coincide con {pattern}")
        files.extend(sorted(matches))

    # sacar repetidos (por si dos patrones agarran el mismo archivo), manteniendo el orden
    unique = []
    seen = set()
    for name in files:
        real = str(Path(name).resolve())
        if real not in seen:
            seen.add(real)
            unique.append(name)
    return unique


# ---------------------------------------------------------------------------
# PASO 1: leer todos los archivos de un proceso y juntarlos en un solo array
# ---------------------------------------------------------------------------
def read_process(patterns):
    files = find_files(patterns)

    tracks_list = []
    jets_list = []
    for name in files:
        with h5py.File(name, "r") as f:
            tracks_list.append(f[TRACKS_IN][:])
            jets_list.append(f[JETS_IN][:])
        print(f"  leído {name}: {len(tracks_list[-1]):,} eventos")

    # todos los archivos tienen que tener la misma estructura para poder juntarlos
    for name, t, j in zip(files, tracks_list, jets_list):
        if t.dtype != tracks_list[0].dtype or t.shape[1:] != tracks_list[0].shape[1:]:
            raise ValueError(f"{name}: los tracks no tienen la misma estructura que {files[0]}")
        if j.dtype != jets_list[0].dtype or j.shape[1:] != jets_list[0].shape[1:]:
            raise ValueError(f"{name}: los Jets no tienen la misma estructura que {files[0]}")

    tracks = np.concatenate(tracks_list)
    jets = np.concatenate(jets_list)
    return tracks, jets


# ---------------------------------------------------------------------------
# PASO 1b: juntar varios procesos de fondo en un único fondo
#   - cada --background es un proceso distinto (p.ej. ttH, ttZ, ...)
#   - con balance=True:
#       * se toma la MISMA cantidad de eventos de cada proceso de fondo
#         (n_keep = la del proceso con menos eventos)
#       * y la señal se ajusta a N_señal = (cantidad de fondos) x n_keep
#         (eso lo aplica main() con el valor que devuelve esta función)
#     Si no hay suficiente señal, se baja n_keep para que alcance.
#   - se guarda de qué proceso viene cada evento (1, 2, 3, ...)
# ---------------------------------------------------------------------------
def read_backgrounds(groups, balance, rng, n_sig):
    tracks_list, jets_list = [], []
    for number, patterns in enumerate(groups, start=1):
        print(f"Leyendo fondo {number}: {' '.join(patterns)}")
        tracks, jets = read_process(patterns)
        tracks_list.append(tracks)
        jets_list.append(jets)

    counts = [len(t) for t in tracks_list]
    counts_read = list(counts)                   # antes de balancear (para el resumen)
    print("  eventos por proceso de fondo:", counts)

    n_sig_keep = n_sig                           # sin balance se usa toda la señal
    if balance:
        n_proc = len(groups)
        n_keep = min(counts)
        if n_proc * n_keep > n_sig:              # no alcanza la señal
            n_keep = n_sig // n_proc
            print(f"  AVISO: hay {n_sig:,} eventos de señal, no alcanzan para {n_proc} x {min(counts):,}.")
            print(f"         Se baja a {n_keep:,} eventos por proceso de fondo.")
        if n_keep == 0:
            raise ValueError("Balanceando quedan 0 eventos por proceso; revisá las entradas o usá --no-balance")
        n_sig_keep = n_proc * n_keep
        print(f"  balanceando: {n_keep:,} eventos de cada fondo y {n_sig_keep:,} de señal "
              f"({n_proc} x {n_keep:,})")
        for k in range(len(tracks_list)):
            chosen = np.sort(rng.permutation(counts[k])[:n_keep])  # eventos elegidos al azar
            tracks_list[k] = tracks_list[k][chosen]
            jets_list[k] = jets_list[k][chosen]

    process_list = [np.full(len(t), number, dtype="i4") for number, t in enumerate(tracks_list, start=1)]

    tracks = np.concatenate(tracks_list)
    jets = np.concatenate(jets_list)
    process = np.concatenate(process_list)
    return tracks, jets, process, counts_read, n_sig_keep


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
# PASO 9: resumen de la separación (split_summary.yaml en la carpeta de salida)
#   - eventos leídos y usados de cada proceso
#   - fracciones de train / val / test
#   - cuántos eventos de cada proceso quedaron en cada archivo
# ---------------------------------------------------------------------------
def write_summary(path, args, n_sig_read, n_sig, bkg_counts_read, process_bkg, sig_idx, bkg_idx):
    frac_test = 1 - args.frac_train - args.frac_val
    summary = {
        "seed": args.seed,
        "balance_fondos": not args.no_balance,
        "fracciones": {
            "train": args.frac_train,
            "val": args.frac_val,
            "test": round(frac_test, 10),
        },
        "procesos": {},
        "totales": {},
    }

    # señal (proceso 0)
    summary["procesos"]["0_senal"] = {
        "rutas": list(args.signal),
        "eventos_leidos": int(n_sig_read),
        "eventos_usados": int(n_sig),
        "train": int(len(sig_idx["train"])),
        "val": int(len(sig_idx["val"])),
        "test": int(len(sig_idx["test"])),
    }

    # cada fondo (procesos 1, 2, 3, ...)
    for number, patterns in enumerate(args.background, start=1):
        entry = {
            "rutas": list(patterns),
            "eventos_leidos": int(bkg_counts_read[number - 1]),
            "eventos_usados": int(np.sum(process_bkg == number)),
        }
        for split in ("train", "val", "test"):
            entry[split] = int(np.sum(process_bkg[bkg_idx[split]] == number))
        summary["procesos"][f"{number}_fondo"] = entry

    # totales por archivo
    for split in ("train", "val", "test"):
        n_s = int(len(sig_idx[split]))
        n_b = int(len(bkg_idx[split]))
        summary["totales"][split] = {"senal": n_s, "fondo": n_b, "total": n_s + n_b}

    with open(path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    print(f"  escrito {path}")


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signal", required=True, nargs="+",
        help="una o más rutas/patrones de señal (se juntan todas como una sola señal)",
    )
    parser.add_argument(
        "--background", required=True, nargs="+", action="append",
        help="repetir una vez por proceso de fondo; cada uno con una o más rutas/patrones",
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="no balancear: usa todos los eventos de señal y de cada fondo",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--frac-train", type=float, default=0.8)
    parser.add_argument("--frac-val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.frac_train + args.frac_val >= 1:
        parser.error("--frac-train + --frac-val tiene que ser menor que 1 (el resto es test)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # PASO 1
    print(f"Leyendo señal: {' '.join(args.signal)}")
    tracks_sig, jets_sig = read_process(args.signal)
    n_sig_read = len(tracks_sig)
    tracks_bkg, jets_bkg, process_bkg, bkg_counts_read, n_sig_keep = read_backgrounds(
        args.background, not args.no_balance, rng, n_sig_read
    )

    # PASO 1c: con balance, la señal se reduce a (cantidad de fondos) x n_keep eventos
    if n_sig_keep < n_sig_read:
        chosen = np.sort(rng.permutation(n_sig_read)[:n_sig_keep])  # eventos elegidos al azar
        tracks_sig = tracks_sig[chosen]
        jets_sig = jets_sig[chosen]
    process_sig = np.zeros(len(tracks_sig), dtype="i4")  # proceso 0 = señal

    # señal y fondo tienen que tener los mismos campos de tracks
    if tracks_sig.dtype != tracks_bkg.dtype or tracks_sig.shape[1:] != tracks_bkg.shape[1:]:
        raise ValueError("Los tracks de señal y de fondo no tienen la misma estructura")

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
        {LABEL_OUT: labels_sig[sig_test], TRACKS_OUT: tracks_sig[sig_test], JETS_OUT: jets_sig[sig_test],
         "process": process_sig[sig_test]},
    )
    write_h5(
        outdir / "pp_output_test_background.h5",
        {LABEL_OUT: labels_bkg[bkg_test], TRACKS_OUT: tracks_bkg[bkg_test], JETS_OUT: jets_bkg[bkg_test],
         "process": process_bkg[bkg_test]},
    )

    # PASOS 7 y 8
    with open(outdir / "norm_dict.yaml", "w") as f:
        yaml.safe_dump(make_norm_dict(tracks_train), f, sort_keys=False)
    with open(outdir / "class_dict.yaml", "w") as f:
        yaml.safe_dump(make_class_dict(labels_train), f)

    # PASO 9
    write_summary(
        outdir / "split_summary.yaml", args, n_sig_read, len(tracks_sig), bkg_counts_read, process_bkg,
        {"train": sig_train, "val": sig_val, "test": sig_test},
        {"train": bkg_train, "val": bkg_val, "test": bkg_test},
    )

    # Resumen
    print("\nEventos       señal      fondo")
    print(f"  train  {len(sig_train):>10,} {len(bkg_train):>10,}")
    print(f"  val    {len(sig_val):>10,} {len(bkg_val):>10,}")
    print(f"  test   {len(sig_test):>10,} {len(bkg_test):>10,}")
    print("\nFondo por proceso (train / val / test):")
    for number, patterns in enumerate(args.background, start=1):
        n = [int(np.sum(process_bkg[idx] == number)) for idx in (bkg_train, bkg_val, bkg_test)]
        print(f"  proceso {number} ({' '.join(patterns)}): {n[0]:,} / {n[1]:,} / {n[2]:,}")


if __name__ == "__main__":
    main()
