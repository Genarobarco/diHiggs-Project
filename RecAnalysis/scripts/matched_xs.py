#!/usr/bin/env python3
"""
Promedia la 'Matched cross-section' de todos los .out de un directorio.

Uso:
    python matched_xs.py /ruta/a/los/out
    python matched_xs.py /ruta/a/los/out --patron "job_ggHH_bbtata_BP1_*.out"
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

# "     Matched cross-section :   0.006997 +- 1.871e-05 pb"
PATRON_XS = re.compile(
    r"Matched cross-section\s*:\s*"
    r"([0-9.eE+-]+)\s*"          # valor
    r"\+-\s*"
    r"([0-9.eE+-]+)",            # error
)

# "     Nb of events after matching/merging :  59093"
PATRON_N = re.compile(r"Nb of events after matching/merging\s*:\s*(\d+)")


def leer_out(ruta):
    """Devuelve (xs, err, n_ev) del ULTIMO match del archivo, o None."""
    xs = err = n_ev = None
    with open(ruta, "r", errors="ignore") as f:
        for linea in f:
            m = PATRON_XS.search(linea)
            if m:
                xs, err = float(m.group(1)), float(m.group(2))
                continue
            m = PATRON_N.search(linea)
            if m:
                n_ev = int(m.group(1))
    if xs is None:
        return None
    return xs, err, n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directorio")
    ap.add_argument("--patron", default="*.out")
    ap.add_argument("--recursivo", action="store_true")
    ap.add_argument("--detalle", action="store_true", help="imprime archivo por archivo")
    args = ap.parse_args()

    if args.recursivo:
        rutas = glob.glob(os.path.join(args.directorio, "**", args.patron), recursive=True)
    else:
        rutas = glob.glob(os.path.join(args.directorio, args.patron))
    rutas.sort()

    if not rutas:
        sys.exit(f"No encontre archivos con patron {args.patron!r} en {args.directorio}")

    xs, err, nev, ok, fallidos = [], [], [], [], []
    for r in rutas:
        res = leer_out(r)
        if res is None:
            fallidos.append(r)
            continue
        xs.append(res[0])
        err.append(res[1])
        nev.append(res[2] if res[2] is not None else 0)
        ok.append(r)

    if not xs:
        sys.exit("Ningun archivo contenia 'Matched cross-section'.")

    xs = np.array(xs)
    err = np.array(err)
    nev = np.array(nev)
    n = len(xs)

    if args.detalle:
        print(f"{'archivo':<55} {'xs [pb]':>12} {'err [pb]':>12} {'N ev':>8}")
        for r, v, e, ne in zip(ok, xs, err, nev):
            print(f"{os.path.basename(r):<55} {v:>12.6g} {e:>12.4g} {ne:>8d}")
        print()

    # --- promedios pedidos ---
    media_xs = xs.mean()
    media_err = err.mean()

    # --- extras utiles ---
    disp = xs.std(ddof=1) if n > 1 else 0.0     # dispersion entre corridas
    err_media = disp / np.sqrt(n) if n > 1 else media_err
    w = 1.0 / err**2                             # promedio ponderado
    xs_pond = np.sum(w * xs) / np.sum(w)
    err_pond = 1.0 / np.sqrt(np.sum(w))

    print(f"Archivos leidos          : {n}  (fallidos: {len(fallidos)})")
    print(f"Eventos tras matching    : {nev.sum()}")
    print()
    print(f"Promedio matched xs      : {media_xs:.6g} pb")
    print(f"Promedio de los errores  : {media_err:.4g} pb")
    print()
    print(f"Desv. estandar entre runs: {disp:.4g} pb")
    print(f"Error de la media        : {err_media:.4g} pb")
    print(f"Promedio ponderado 1/s^2 : {xs_pond:.6g} +- {err_pond:.4g} pb")

    if fallidos:
        print("\nSin 'Matched cross-section':")
        for r in fallidos:
            print("  ", r)


if __name__ == "__main__":
    main()
