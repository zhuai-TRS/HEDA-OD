#!/usr/bin/env bash
# Reproduce NYC24Q1–Q3 main results (public data).
# Zone OD is private and is not redistributed with this repository.
set -euo pipefail
cd "$(dirname "$0")"

# sh scripts/HEDA_OD/zone.sh

sh scripts/HEDA_OD/nyc24q1.sh
sh scripts/HEDA_OD/nyc24q2.sh
sh scripts/HEDA_OD/nyc24q3.sh

# Ablations require Zone OD:
# sh scripts/Ablation/zone_branch.sh
# sh scripts/Ablation/alpha_grid_zone.sh
