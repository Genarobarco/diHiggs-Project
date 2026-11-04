#!/usr/bin/env python3
"""
Construye los .h5 de train / val / test para Salt a partir de la salida de h5_dumper.py.

Entrada (por clase): uno o más .h5 con
  - 'tracks' : dataset estructurado 2D (n_events, n_rows)  (padding = NaN, como escribe h5_dumper.py)
  - 'jets'   : OPCIONAL. Si no existe, se crea con solo el label (Salt necesita un objeto global
               que lleve el label, pero no hace falta que tenga variables de entrada).

Salida (en --outdir):
  - pp_output_train.h5 / pp_output_val.h5 : señal + fondo mezclados y barajados
  - pp_output_test_signal.h5 / pp_output_test_background.h5 : test por clase
  - pp_output_test.h5 : test combinado
  - norm_dict.yaml  : media/std por variable de tracks, SOLO con train
  - class_dict.yaml : pesos de clase, SOLO con train

Agrega:
  - jets['flavour_label']  (0 = señal, 1 = fondo; mismo orden que class_names del YAML de Salt)
  - tracks['valid']        (bool; False en el padding)
Reemplaza NaN/inf por 0 y (opcional) ordena los tracks por --sort-by descendente,
para que el recorte num_inputs de Salt se quede con los de mayor PT.

Uso:
  python make_salt_h5.py --signal sig_part*.h5 --background bkg_part*.h5 \
      --outdir /scratch/ATLAS/TauTrigger/salt_inputs_tracks_v01 \
      --fracs 0.8 0.1 0.1 --seed 42 --const-name tracks --valid-var PT --sort-by PT
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import yaml

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- lectura
class Source:
    """Concatena virtualmente varios .h5 de una misma clase."""

    def __init__(self, paths: list[str], jets: str, const: str):
        self.files = [h5py.File(p, "r") for p in paths]
        self.has_jets = all(jets in f for f in self.files)
        sizes = []
        for f, p in zip(self.files, paths):
            if const not in f:
                raise KeyError(f"{p}: no tiene dataset '{const}' (tiene {list(f.keys())}).")
            nc = f[const].shape[0]
            if self.has_jets and f[jets].shape[0] != nc:
                raise RuntimeError(f"{p}: '{jets}' tiene {f[jets].shape[0]} filas y '{const}' {nc}.")
            sizes.append(nc)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.n = int(self.offsets[-1])

    def read(self, ds: str, start: int, stop: int) -> np.ndarray:
        parts = []
        for k, f in enumerate(self.files):
            lo, hi = max(start, self.offsets[k]), min(stop, self.offsets[k + 1])
            if lo < hi:
                parts.append(f[ds][lo - self.offsets[k] : hi - self.offsets[k]])
        return np.concatenate(parts, axis=0)

    def close(self):
        for f in self.files:
            f.close()


# --------------------------------------------------------------------------- escritura
class Writer:
    def __init__(self, path: Path, chunk_rows: int = 1000):
        self.f = h5py.File(path, "w")
        self.chunk_rows = chunk_rows
        self.n = 0

    def append(self, arrays: dict[str, np.ndarray]):
        n_new = len(next(iter(arrays.values())))
        if n_new == 0:
            return
        for name, a in arrays.items():
            if name not in self.f:
                self.f.create_dataset(
                    name,
                    shape=(0,) + a.shape[1:],
                    maxshape=(None,) + a.shape[1:],
                    dtype=a.dtype,
                    chunks=(self.chunk_rows,) + a.shape[1:],
                    compression="lzf",
                )
            d = self.f[name]
            d.resize(self.n + n_new, axis=0)
            d[self.n :] = a
        self.n += n_new

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- transformaciones
def add_field(arr: np.ndarray, name: str, dtype, values) -> np.ndarray:
    new_dtype = np.dtype(arr.dtype.descr + [(name, dtype)])
    out = np.empty(arr.shape, dtype=new_dtype)
    for f in arr.dtype.names:
        out[f] = arr[f]
    out[name] = values
    return out


def clean_floats(arr: np.ndarray) -> None:
    for f in arr.dtype.names:
        if np.issubdtype(arr.dtype[f], np.floating):
            col = arr[f]
            col[~np.isfinite(col)] = 0.0


def transform(jets, const, label: int, valid_var: str, sort_by: str | None):
    if valid_var not in const.dtype.names:
        raise KeyError(f"'{valid_var}' no está en las constituyentes: {const.dtype.names}")
    valid = np.isfinite(const[valid_var])
    const = add_field(const, "valid", np.bool_, valid)
    clean_floats(const)
    if sort_by:
        key = np.where(const["valid"], const[sort_by], -np.inf)
        order = np.argsort(-key, axis=1, kind="stable")
        const = np.take_along_axis(const, order, axis=1)
    if jets is None:
        jets = np.zeros(len(const), dtype=[("flavour_label", np.int32)])
        jets["flavour_label"] = label
    else:
        clean_floats(jets)
        jets = add_field(jets, "flavour_label", np.int32, label)
    return jets, const


# --------------------------------------------------------------------------- estadística (train)
class Stats:
    def __init__(self):
        self.s, self.s2, self.n = {}, {}, {}

    def update(self, group: str, arr: np.ndarray, mask=None):
        for f in arr.dtype.names:
            if f in ("valid", "flavour_label") or not np.issubdtype(arr.dtype[f], np.number):
                continue
            x = arr[f] if mask is None else arr[f][mask]
            x = x.astype(np.float64).ravel()
            k = (group, f)
            self.s[k] = self.s.get(k, 0.0) + x.sum()
            self.s2[k] = self.s2.get(k, 0.0) + (x * x).sum()
            self.n[k] = self.n.get(k, 0) + x.size

    def norm_dict(self) -> dict:
        out: dict = {}
        for (g, f), n in self.n.items():
            if n == 0:
                continue
            mean = self.s[(g, f)] / n
            var = max(self.s2[(g, f)] / n - mean * mean, 0.0)
            std = float(np.sqrt(var)) or 1.0  # Salt rechaza std = 0
            out.setdefault(g, {})[f] = {"mean": float(mean), "std": std}
        return out


# --------------------------------------------------------------------------- split
def assign_splits(n: int, fracs, rng) -> np.ndarray:
    perm = rng.permutation(n)
    n_tr, n_va = int(fracs[0] * n), int(fracs[1] * n)
    split = np.empty(n, dtype=np.int8)
    split[perm[:n_tr]] = 0
    split[perm[n_tr : n_tr + n_va]] = 1
    split[perm[n_tr + n_va :]] = 2
    return split


def expand(patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        m = sorted(glob.glob(p)) or [p]
        out += [x for x in m if x not in out]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", nargs="+", required=True)
    ap.add_argument("--background", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fracs", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jets-name", default="jets", help="objeto global (lleva el label)")
    ap.add_argument("--const-name", default="tracks", help="dataset de constituyentes (input de la red)")
    ap.add_argument("--valid-var", default="PT", help="variable que define 'valid' (NaN = padding)")
    ap.add_argument("--sort-by", default="PT", help="ordenar constituyentes descendente por esta variable ('' = no ordenar)")
    ap.add_argument("--block-size", type=int, default=20000, help="eventos leídos por bloque")
    ap.add_argument("--group", type=int, default=10, help="bloques mezclados por buffer (memoria ~ group*block)")
    args = ap.parse_args()

    if abs(sum(args.fracs) - 1) > 1e-6:
        raise ValueError("--fracs debe sumar 1")
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sources = {
        0: Source(expand(args.signal), args.jets_name, args.const_name),
        1: Source(expand(args.background), args.jets_name, args.const_name),
    }
    if sources[0].has_jets != sources[1].has_jets:
        raise RuntimeError(f"'{args.jets_name}' existe en una clase y no en la otra.")
    splits = {lab: assign_splits(src.n, args.fracs, rng) for lab, src in sources.items()}

    J, C = args.jets_name, args.const_name
    writers = {
        "train": Writer(outdir / "pp_output_train.h5"),
        "val": Writer(outdir / "pp_output_val.h5"),
        "test": Writer(outdir / "pp_output_test.h5"),
        ("test", 0): Writer(outdir / "pp_output_test_signal.h5"),
        ("test", 1): Writer(outdir / "pp_output_test_background.h5"),
    }
    stats = Stats()
    counts = {s: [0, 0] for s in SPLITS}

    # Bloques en orden aleatorio dentro de cada clase (mezcla aunque los ROOT vengan ordenados)
    # y repartidos en proporción entre grupos -> cada buffer tiene ~la misma fracción señal/fondo.
    per_class = {}
    for lab, src in sources.items():
        starts = np.arange(0, src.n, args.block_size)
        per_class[lab] = [(lab, int(b), int(min(b + args.block_size, src.n))) for b in starts[rng.permutation(len(starts))]]
    n_groups = max(1, -(-sum(len(v) for v in per_class.values()) // args.group))
    n_groups = min(n_groups, min(len(v) for v in per_class.values()))
    groups = [[] for _ in range(n_groups)]
    for lab, bl in per_class.items():
        for k, chunk in enumerate(np.array_split(np.arange(len(bl)), n_groups)):
            groups[k] += [bl[i] for i in chunk]

    for g, group_blocks in enumerate(groups):
        buf = {s: {J: [], C: []} for s in SPLITS}
        for lab, start, stop in group_blocks:
            src = sources[lab]
            j = src.read(J, start, stop) if src.has_jets else None
            c = src.read(C, start, stop)
            j, c = transform(j, c, lab, args.valid_var, args.sort_by or None)
            sid = splits[lab][start:stop]
            for k, s in enumerate(SPLITS):
                m = sid == k
                if not m.any():
                    continue
                buf[s][J].append(j[m])
                buf[s][C].append(c[m])
                counts[s][lab] += int(m.sum())
                if s == "test":
                    writers[("test", lab)].append({J: j[m], C: c[m]})

        for s in SPLITS:
            if not buf[s][J]:
                continue
            jj, cc = np.concatenate(buf[s][J]), np.concatenate(buf[s][C])
            p = rng.permutation(len(jj))
            jj, cc = jj[p], cc[p]
            if s == "train":
                stats.update(J, jj)
                stats.update(C, cc, mask=cc["valid"])
            writers[s].append({J: jj, C: cc})
        print(f"  grupo {g + 1}/{n_groups}", flush=True)

    for w in writers.values():
        w.close()
    for src in sources.values():
        src.close()

    n_sig, n_bkg = counts["train"]
    tot = n_sig + n_bkg
    class_w = [tot / (2 * n_sig), tot / (2 * n_bkg)]
    with open(outdir / "class_dict.yaml", "w") as f:
        yaml.safe_dump({J: {"flavour_label": [float(w) for w in class_w]}}, f)
    with open(outdir / "norm_dict.yaml", "w") as f:
        yaml.safe_dump(stats.norm_dict(), f, sort_keys=False)

    print("\nEventos (señal, fondo):")
    for s in SPLITS:
        print(f"  {s:5s}: {counts[s][0]:>9d}  {counts[s][1]:>9d}")
    print(f"Salida en {outdir}")


if __name__ == "__main__":
    main()
