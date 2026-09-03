#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
openscad --render -o rotating_diode.stl rotating_diode.scad
openscad --render -D 'part="left"' -o rotating_diode_left.stl rotating_diode.scad
openscad --render -D 'part="right"' -o rotating_diode_right.stl rotating_diode.scad
