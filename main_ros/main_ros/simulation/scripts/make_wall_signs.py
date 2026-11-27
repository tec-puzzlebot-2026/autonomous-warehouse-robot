#!/usr/bin/env python3
"""
make_wall_signs.py — ArUco markers and company logos for the Isaac Sim walls
===========================================================================
Writes one plate per image (a thin square showing it, visual only, facing +x,
centered on its origin; the same plate as the QR codes, make_qr_assets.py):

  assets/aruco/aruco_plate_<ID>.usda    from assets/aruco/aruco_<ID>.png
      DICT_ARUCO_ORIGINAL, IDs 5 7 13 21 37. The PNG is 1800 px with a 1400 px black
      marker (white margin 200 px), so the plate is marker side * 9/7 wide: with the
      8.7 cm markers of config/aruco_localization.yaml, 11.19 cm.
  assets/logos/logo_plate_<company>.usda  from assets/logos/<company>.png
      emezon, soriene, walnut (the logo detector's classes), --logo-size wide.

In Isaac, drag a plate onto a wall and turn it so it faces into the arena (its +x).
Each company's logo goes on the wall just LEFT of its dock opening (seen from inside).

    python3 make_wall_signs.py                       # marker 8.7 cm, logos 15 cm
    python3 make_wall_signs.py --logo-size 0.20
"""
# Author: Raymundo Gomez

import argparse
import glob
import os

import yaml

from make_qr_assets import PKG, SIM, plate_usda

ARUCO_PNG_TO_MARKER = 1800 / 1400          # plate side / black marker side


def marker_size(default=0.087):
    """per_marker_size of config/aruco_localization.yaml (they are all the same)."""
    try:
        with open(os.path.join(PKG, 'config', 'aruco_localization.yaml')) as f:
            cfg = yaml.safe_load(f)
        for section in cfg.values():
            sizes = (section or {}).get('ros__parameters', {}).get('per_marker_size')
            if sizes:
                return float(sizes[0])
    except (OSError, ValueError, AttributeError):
        pass
    return default


def write(folder, png_name, usda_name, size, what, prim):
    text = plate_usda(None, f'./{png_name}', size, prim=prim, what=what,
                      script='make_wall_signs.py')
    with open(os.path.join(folder, usda_name), 'w') as f:
        f.write(text)
    print(f'{os.path.basename(folder)}/{usda_name}: {what}, {100 * size:.2f} cm')


def main():
    ap = argparse.ArgumentParser(description='ArUco and logo plates for Isaac Sim')
    ap.add_argument('--marker', type=float, default=None,
                    help='black marker side in m (default: aruco_localization.yaml)')
    ap.add_argument('--logo-size', type=float, default=0.15, help='logo width in m')
    args = ap.parse_args()
    side = (args.marker or marker_size()) * ARUCO_PNG_TO_MARKER
    aruco = os.path.join(SIM, 'assets', 'aruco')
    for png in sorted(glob.glob(os.path.join(aruco, 'aruco_*.png'))):
        marker_id = os.path.basename(png)[len('aruco_'):-len('.png')]
        write(aruco, os.path.basename(png), f'aruco_plate_{marker_id}.usda', side,
              f'ArUco {int(marker_id)} (DICT_ARUCO_ORIGINAL)', 'ArUco')
    logos = os.path.join(SIM, 'assets', 'logos')
    for png in sorted(glob.glob(os.path.join(logos, '*.png'))):
        company = os.path.basename(png)[:-len('.png')]
        write(logos, os.path.basename(png), f'logo_plate_{company}.usda', args.logo_size,
              f'{company} logo', 'Logo')


if __name__ == '__main__':
    main()
