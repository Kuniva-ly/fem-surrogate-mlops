"""Generate synthetic stress field images from FEM simulation data.

Structure de sortie (compatible Keras image_dataset_from_directory) :
    data/unstructured/images/
        train/with_hole/          *.png
        train/without_hole/       *.png
        train/with_hole_moving/   *.png
        val/...
        test/...

Normalisation GLOBALE : toutes les images partagent la meme echelle de couleur.
  pixel = kirsch_vm / global_vm_max
  => le blanc (1.0) correspond au max physique global du dataset.

Targets encodees dans le nom de fichier (pas de CSV) :
  {sim_id}_{idx:05d}_V{vm_int}_D{disp_int}.png
  - vm_int   = round(log10(max_von_mises_pa) * 10000)
  - disp_int = round((log10(max_displacement_m) + 12) * 10000)

Recuperation dans le notebook :
  log10(vm)   = vm_int / 10000
  log10(disp) = disp_int / 10000 - 12

Usage :
    python scripts/generate_images.py
    python scripts/generate_images.py --size 128
    python scripts/generate_images.py --max-per-class 500
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUT_DIR  = DATA_DIR / 'unstructured' / 'images'

_EPS = 1e-12


# Champ de contrainte Von Mises (solution de Kirsch)

def stress_field(row: pd.Series, size: int = 64) -> np.ndarray:
    L   = float(row['length_m'])
    H   = float(row['height_m'])
    sig = float(row['traction_pa'])
    geo = str(row['geometry_type'])

    x = np.linspace(0, L, size)
    y = np.linspace(0, H, size)
    X, Y = np.meshgrid(x, y)

    has_hole = 'without_hole' not in geo

    if has_hole:
        cx_r = row.get('hole_cx_ratio', np.nan)
        cy_r = row.get('hole_cy_ratio', np.nan)
        a    = row.get('radius_abs',    np.nan)

        cx = float(cx_r) * L if not pd.isna(cx_r) else L / 2
        cy = float(cy_r) * H if not pd.isna(cy_r) else H / 2
        a  = float(a)         if not pd.isna(a)    else 0.1 * min(L, H)

        dx   = X - cx
        dy   = Y - cy
        r    = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1e-9)
        th   = np.arctan2(dy, dx)
        a2r2 = np.minimum((a / r) ** 2, 1.0)
        a4r4 = a2r2 ** 2
        c2   = np.cos(2 * th)
        s2   = np.sin(2 * th)

        srr = (sig / 2) * (1 - a2r2)     + (sig / 2) * (1 - 4 * a2r2 + 3 * a4r4) * c2
        stt = (sig / 2) * (1 + a2r2)     - (sig / 2) * (1 + 3 * a4r4) * c2
        trt = -(sig / 2) * (1 + 2 * a2r2 - 3 * a4r4) * s2

        vm = np.sqrt(srr ** 2 + stt ** 2 - srr * stt + 3 * trt ** 2)
        vm[r <= a] = 0.0
    else:
        vm = np.ones((size, size)) * sig
        for i in range(min(4, size)):
            factor = 0.65 + 0.09 * i
            vm[i, :]      *= factor
            vm[-i - 1, :] *= factor

    return vm


def _encode_targets(vm_pa: float, disp_m: float) -> tuple[int, int]:
    vm_int   = round(np.log10(max(vm_pa,   _EPS)) * 10000)
    disp_int = round((np.log10(max(disp_m, _EPS)) + 12) * 10000)
    return vm_int, disp_int


def _load_splits(
    splits: dict[str, pathlib.Path],
    max_per_class: int | None,
) -> dict[str, pd.DataFrame]:
    all_data: dict[str, pd.DataFrame] = {}
    geo_col = 'geometry_type'
    for split_name, parquet_path in splits.items():
        if not parquet_path.exists():
            print(f'[WARN] {parquet_path} introuvable — ignore.')
            continue
        df = pd.read_parquet(parquet_path)
        if max_per_class is not None:
            parts = []
            for geo in sorted(df[geo_col].dropna().unique()):
                sub = df[df[geo_col] == geo]
                parts.append(sub.sample(min(max_per_class, len(sub)), random_state=42))
            df = pd.concat(parts).reset_index(drop=True)
        all_data[split_name] = df
    return all_data


# Pipeline

def generate(size: int = 64, max_per_class: int | None = None) -> None:
    splits = {
        'train': DATA_DIR / 'processed' / 'train.parquet',
        'val':   DATA_DIR / 'processed' / 'val.parquet',
        'test':  DATA_DIR / 'processed' / 'test.parquet',
    }

    all_data = _load_splits(splits, max_per_class)
    if not all_data:
        print('Aucun fichier parquet trouve. Lancez build-features d\'abord.')
        return

    # Passe 1 : calcul du max global Von Mises (Kirsch) pour la normalisation globale
    print('Passe 1/2 : calcul du max global Von Mises...')
    global_vm_max = 0.0
    total_rows = sum(len(df) for df in all_data.values())
    done = 0
    for split_name, df in all_data.items():
        for _, row in df.iterrows():
            vm = stress_field(row, size)
            local_max = float(vm.max())
            if local_max > global_vm_max:
                global_vm_max = local_max
            done += 1
            if done % 5000 == 0:
                print(f'  {done}/{total_rows} lignes traitees...', flush=True)

    global_vm_max = max(global_vm_max, _EPS)
    print(f'Max global Von Mises (Kirsch) : {global_vm_max:.4e} Pa\n')

    # Sauvegarder le max global pour reference (lecture possible dans le notebook)
    meta_dir = DATA_DIR / 'unstructured'
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / 'global_vm_max.txt').write_text(str(global_vm_max))

    # Passe 2 : generation et sauvegarde avec normalisation globale
    print('Passe 2/2 : generation des images...')
    total_saved = 0
    geo_col = 'geometry_type'

    for split_name, df in all_data.items():
        for geo in sorted(df[geo_col].dropna().unique()):
            subset = df[df[geo_col] == geo].reset_index(drop=True)
            out_dir = OUT_DIR / split_name / geo
            out_dir.mkdir(parents=True, exist_ok=True)

            n = len(subset)
            print(f'  {split_name}/{geo:<20} {n:>5} images...', end='', flush=True)

            for i, (_, row) in enumerate(subset.iterrows()):
                sim_id = str(row['simulation_id'])[:16]

                # Targets dans le nom de fichier
                vm_pa  = float(row.get('max_von_mises_pa',  0) or 0)
                disp_m = float(row.get('max_displacement_m', 0) or 0)

                if vm_pa > 0 and disp_m > 0:
                    vm_int, disp_int = _encode_targets(vm_pa, disp_m)
                    filename = f'{sim_id}_{i:05d}_V{vm_int}_D{disp_int}.png'
                else:
                    # Pas de target FEM disponible : nom sans suffixe regression
                    filename = f'{sim_id}_{i:05d}.png'

                out_path = out_dir / filename

                # Normalisation GLOBALE
                vm   = stress_field(row, size)
                vm_n = np.clip(vm / global_vm_max, 0.0, 1.0)
                plt.imsave(out_path, vm_n, cmap='hot', format='png')

                total_saved += 1
                if (i + 1) % 500 == 0:
                    print(f' {i + 1}', end='', flush=True)

            print('  OK')

    print(f'\n=== Termine ===')
    print(f'Images sauvegardees  : {total_saved}')
    print(f'Normalisation globale: {global_vm_max:.4e} Pa  (sauvegardee dans global_vm_max.txt)')
    print(f'Dossier              : {OUT_DIR}')
    print(f'Targets              : encodees dans les noms de fichiers (V{{vm_int}}_D{{disp_int}})')


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description='Genere les images de champ de contrainte.')
    parser.add_argument('--size',          type=int, default=64)
    parser.add_argument('--max-per-class', type=int, default=None)
    args = parser.parse_args()

    print(f'Generation des images — taille {args.size}x{args.size} px')
    if args.max_per_class:
        print(f'Limite : {args.max_per_class} images par classe/split')
    print(f'Sortie : {OUT_DIR}\n')

    generate(size=args.size, max_per_class=args.max_per_class)


if __name__ == '__main__':
    main()
