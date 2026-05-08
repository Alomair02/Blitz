"""
parse_triangles.py
Reads the CAS NAIC Schedule P ppauto.csv and comauto.csv files,
filters to complete 10x10 triangles, and writes binary files for
the GPU bootstrap IBNR engine.

Output:
  triangles.bin     -- float32 (N_COMPANIES, 10, 10), cumulative paid losses
  premiums.bin      -- float32 (N_COMPANIES, 10), net earned premium by acc year
  companies.txt     -- one "GRCODE|GRNAME" per line
  tri_meta.txt      -- N_COMPANIES, dimensions, source file

The triangle layout:
  axis 0 : company
  axis 1 : accident year (0=1998 .. 9=2007)
  axis 2 : development lag (0=lag1 .. 9=lag10)
  value  : cumulative paid loss (raw dollars)

Upper triangle (known): ay_idx + lag_idx < 10  (i.e. ay+lag-1 <= 2007)
Lower triangle (actual outcomes, for validation): ay_idx + lag_idx >= 10
"""

import csv
import numpy as np
from collections import defaultdict

SOURCES = [
    ("ppauto.csv",  "CumPaidLoss",  "EarnedPremNet", "ppauto"),
    ("comauto.csv", "CumPaidLoss",  "EarnedPremNet", "comauto"),
]

MIN_LAG   = 10   # require complete 10x10
MIN_YEARS = 10

def load_source(filepath, paid_col, prem_col, label):
    triangles = defaultdict(dict)   # co -> (ay_idx, lag_idx) -> value
    premiums  = defaultdict(dict)   # co -> ay_idx -> premium
    names     = {}

    with open(filepath) as f:
        for row in csv.DictReader(f):
            co  = row['GRCODE']
            names[co] = f"{row['GRNAME']} ({label})"
            ay  = int(row['AccidentYear'])  - 1998   # 0..9
            lag = int(row['DevelopmentLag']) - 1      # 0..9

            try:
                paid = float(row[paid_col])
                prem = float(row[prem_col])
            except (ValueError, KeyError):
                continue

            triangles[co][(ay, lag)] = paid
            premiums[co][ay]         = prem

    return triangles, premiums, names

all_triangles = {}
all_premiums  = {}
all_names     = {}

for filepath, paid_col, prem_col, label in SOURCES:
    try:
        t, p, n = load_source(filepath, paid_col, prem_col, label)
        all_triangles.update(t)
        all_premiums.update(p)
        all_names.update(n)
        print(f"Loaded {filepath}: {len(t)} companies")
    except FileNotFoundError:
        print(f"Skipping {filepath} (not found)")

# Filter: keep only complete 10x10 triangles with positive values
accepted = []
for co, tri in all_triangles.items():
    if len(tri) < 100:
        continue
    vals = list(tri.values())
    if any(v < 0 for v in vals):
        continue
    if all_premiums[co].get(0, 0) <= 0:
        continue
    accepted.append(co)

print(f"\nAccepted companies: {len(accepted)}")
N = len(accepted)

# Build arrays
tri_arr  = np.zeros((N, 10, 10), dtype=np.float32)
prem_arr = np.zeros((N, 10),     dtype=np.float32)

for i, co in enumerate(accepted):
    for (ay, lag), val in all_triangles[co].items():
        tri_arr[i, ay, lag] = val
    for ay, val in all_premiums[co].items():
        if 0 <= ay < 10:
            prem_arr[i, ay] = val

# Normalise: convert to thousands to avoid float32 precision loss on large values
tri_arr  /= 1000.0
prem_arr /= 1000.0

print(f"Triangle array  : {tri_arr.shape}  {tri_arr.nbytes/1e3:.1f}KB")
print(f"Premium array   : {prem_arr.shape}  {prem_arr.nbytes/1e3:.1f}KB")

# Quick sanity: print one triangle
co = accepted[0]
i  = 0
print(f"\nSample — {all_names[co]}  (values in $000s)")
print(f"{'':>6}", end="")
for lag in range(1, 11):
    print(f"  Lag{lag:02d}", end="")
print()
for ay in range(10):
    print(f"{1998+ay}", end="")
    for lag in range(10):
        val = tri_arr[i, ay, lag]
        is_upper = (ay + lag) < 9   # strict upper
        is_diag  = (ay + lag) == 9  # latest diagonal
        if is_upper or is_diag:
            print(f"  {val:6.0f}", end="")
        else:
            print(f" [{val:6.0f}]", end="")
    print()

# Save
tri_arr.tofile("triangles.bin")
prem_arr.tofile("premiums.bin")

with open("companies.txt", "w") as f:
    for co in accepted:
        f.write(f"{co}|{all_names[co]}\n")

with open("tri_meta.txt", "w") as f:
    f.write(f"n_companies={N}\n")
    f.write(f"n_acc_years=10\n")
    f.write(f"n_dev_lags=10\n")
    f.write(f"units=thousands_usd\n")
    f.write(f"acc_year_base=1998\n")
    f.write(f"upper_triangle=ay+lag<9\n")
    f.write(f"latest_diagonal=ay+lag==9\n")
    f.write(f"lower_triangle=ay+lag>9\n")
    f.write(f"triangles_file=triangles.bin\n")
    f.write(f"premiums_file=premiums.bin\n")
    f.write(f"companies_file=companies.txt\n")

print(f"\nWrote triangles.bin, premiums.bin, companies.txt, tri_meta.txt")
print(f"Ready for GPU bootstrap engine ({N} companies x 10x10 triangles)")
