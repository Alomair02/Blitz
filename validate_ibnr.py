"""
validate_ibnr.py
Compares GPU bootstrap IBNR estimates against actual lower triangle outcomes
from the CAS NAIC Schedule P dataset.

For each company:
  - True IBNR  = sum of actual lower triangle payments (from dataset)
  - Bootstrap  = distribution of 10,000 simulated IBNR estimates
  - Posted     = reserve held by company at year-end 2007

Key questions:
  1. Is the bootstrap mean close to true IBNR?  (bias check)
  2. How often does the P75 reserve cover true IBNR?  (should be ~75%)
  3. How often does the P95 reserve cover true IBNR?  (should be ~95%)
  4. How did posted reserves compare to true IBNR?    (industry adequacy)
  5. Which companies were most under/over-reserved?
"""

import numpy as np
import csv
from collections import defaultdict

# -----------------------------------------------------------------------
# 1. Load bootstrap samples
# -----------------------------------------------------------------------
N_COMPANIES = 146
N_BOOT      = 10000

print("Loading bootstrap samples...")
samples = np.fromfile("ibnr_samples.bin", dtype=np.float32) \
            .reshape(N_COMPANIES, N_BOOT)

# Load company index
companies = []
with open("companies.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            grcode, name = line.split("|", 1)
            companies.append((grcode.strip(), name.strip()))

print(f"  {N_COMPANIES} companies, {N_BOOT} bootstrap samples each\n")

# -----------------------------------------------------------------------
# 2. Compute true IBNR from lower triangle
#    Lower triangle: cells where ay + lag > 9 (0-indexed)
#    True IBNR for each company = sum of (C(ay, 9) - C(ay, lag_diag))
#    where C(ay, 9) is the fully developed ultimate (lag 10)
#    and lag_diag = 9 - ay is the latest known lag at valuation
# -----------------------------------------------------------------------
print("Computing true IBNR from lower triangle outcomes...")

# Load raw triangles
tri = np.fromfile("triangles.bin", dtype=np.float32) \
        .reshape(N_COMPANIES, 10, 10)

# True IBNR: for each ay=1..9, ultimate (lag index 9) - latest diagonal
# ay=0 is fully developed, no IBNR
true_ibnr = np.zeros(N_COMPANIES)
for co_idx in range(N_COMPANIES):
    total = 0.0
    for ay in range(1, 10):
        lag_diag  = 9 - ay          # latest known lag index
        ultimate  = tri[co_idx, ay, 9]   # actual ultimate (lag 10)
        diagonal  = tri[co_idx, ay, lag_diag]  # latest diagonal at valuation
        total    += max(0, ultimate - diagonal)
    true_ibnr[co_idx] = total

# -----------------------------------------------------------------------
# 3. Load posted reserves from CSV
# -----------------------------------------------------------------------
print("Loading posted reserves from CSVs...")

posted_reserves = {}   # grcode -> posted reserve (sum across accident years)

for fname in ["ppauto.csv", "comauto.csv"]:
    try:
        with open(fname) as f:
            for row in csv.DictReader(f):
                grcode = row["GRCODE"].strip()
                try:
                    # PostedReserves2007 is reported once per company
                    # (same value repeated across rows) — take max
                    pr = float(row.get("PostedReserves2007", 0) or 0)
                    if grcode not in posted_reserves:
                        posted_reserves[grcode] = pr
                    else:
                        posted_reserves[grcode] = max(posted_reserves[grcode], pr)
                except ValueError:
                    pass
    except FileNotFoundError:
        print(f"  {fname} not found, skipping")

# -----------------------------------------------------------------------
# 4. Per-company statistics
# -----------------------------------------------------------------------
print("Computing statistics...\n")

boot_mean = samples.mean(axis=1)
boot_std  = samples.std(axis=1)
boot_p50  = np.percentile(samples, 50, axis=1)
boot_p75  = np.percentile(samples, 75, axis=1)
boot_p95  = np.percentile(samples, 95, axis=1)
boot_p99  = np.percentile(samples, 99, axis=1)

# Coverage: does percentile reserve cover true IBNR?
cover_p50 = (boot_p50 >= true_ibnr)
cover_p75 = (boot_p75 >= true_ibnr)
cover_p95 = (boot_p95 >= true_ibnr)

# Bias: (boot_mean - true_ibnr) / true_ibnr
with np.errstate(divide='ignore', invalid='ignore'):
    bias_pct = np.where(true_ibnr > 1,
                        (boot_mean - true_ibnr) / true_ibnr * 100,
                        np.nan)

# Posted reserve adequacy
posted = np.array([
    posted_reserves.get(grcode, np.nan) / 1000  # convert to thousands
    for grcode, _ in companies
])
cover_posted = (posted >= true_ibnr)

# -----------------------------------------------------------------------
# 5. Portfolio-level summary
# -----------------------------------------------------------------------
valid = true_ibnr > 1   # exclude companies with near-zero true IBNR
n_valid = valid.sum()

print("=" * 70)
print("PORTFOLIO SUMMARY")
print("=" * 70)
print(f"  Companies analysed          : {N_COMPANIES}")
print(f"  Companies with IBNR > $1K  : {n_valid}")
print()
print(f"  Total true IBNR             : ${true_ibnr.sum():>10,.0f}K")
print(f"  Total bootstrap mean IBNR  : ${boot_mean.sum():>10,.0f}K")
print(f"  Total posted reserves       : ${np.nansum(posted):>10,.0f}K")
print()
print(f"  Bootstrap bias (mean)       : {np.nanmean(bias_pct[valid]):>+.1f}%")
print(f"  Bootstrap bias (median)     : {np.nanmedian(bias_pct[valid]):>+.1f}%")
print()
print("  Reserve adequacy (% of companies where reserve >= true IBNR):")
print(f"    P50 coverage              : {cover_p50[valid].mean()*100:.1f}%  (target ~50%)")
print(f"    P75 coverage              : {cover_p75[valid].mean()*100:.1f}%  (target ~75%)")
print(f"    P95 coverage              : {cover_p95[valid].mean()*100:.1f}%  (target ~95%)")
print(f"    Posted reserve coverage   : {np.nanmean(cover_posted[valid])*100:.1f}%")
print()

# -----------------------------------------------------------------------
# 6. Detailed company table
# -----------------------------------------------------------------------
print("=" * 70)
print("COMPANY DETAIL (sorted by true IBNR, top 30)")
print("=" * 70)
print(f"{'Company':<35} {'TrueIBNR':>9} {'BootMean':>9} "
      f"{'Bias%':>7} {'P75':>8} {'P95':>8} {'Posted':>9} {'Adeq':>5}")
print("-" * 95)

order = np.argsort(-true_ibnr)
for rank, co_idx in enumerate(order[:30]):
    grcode, name = companies[co_idx]
    short_name   = name[:34]
    ti           = true_ibnr[co_idx]
    bm           = boot_mean[co_idx]
    bp           = bias_pct[co_idx]
    p75          = boot_p75[co_idx]
    p95          = boot_p95[co_idx]
    pr           = posted[co_idx]
    adq          = "✓" if (not np.isnan(pr) and pr >= ti) else "✗"

    bias_str = f"{bp:>+.0f}%" if not np.isnan(bp) else "  N/A"
    pr_str   = f"{pr:>9,.0f}" if not np.isnan(pr) else "      N/A"

    print(f"{short_name:<35} {ti:>9,.0f} {bm:>9,.0f} "
          f"{bias_str:>7} {p75:>8,.0f} {p95:>8,.0f} {pr_str} {adq:>5}")

# -----------------------------------------------------------------------
# 7. Calibration analysis
# -----------------------------------------------------------------------
print()
print("=" * 70)
print("CALIBRATION ANALYSIS — Coverage by true IBNR size bucket")
print("=" * 70)
print(f"{'IBNR Bucket':<20} {'N':>4} {'P50 cov':>8} {'P75 cov':>8} {'P95 cov':>8}")
print("-" * 52)

buckets = [
    ("< $10K",   true_ibnr < 10),
    ("$10K-$50K",  (true_ibnr >= 10)  & (true_ibnr < 50)),
    ("$50K-$200K", (true_ibnr >= 50)  & (true_ibnr < 200)),
    ("> $200K",    true_ibnr >= 200),
]
for label, mask in buckets:
    n = mask.sum()
    if n == 0:
        continue
    c50 = cover_p50[mask].mean() * 100
    c75 = cover_p75[mask].mean() * 100
    c95 = cover_p95[mask].mean() * 100
    print(f"{label:<20} {n:>4} {c50:>7.1f}% {c75:>7.1f}% {c95:>7.1f}%")

# -----------------------------------------------------------------------
# 8. Worst misses — companies where even P95 reserve was inadequate
# -----------------------------------------------------------------------
print()
print("=" * 70)
print("WORST MISSES — True IBNR exceeded P95 bootstrap reserve")
print("=" * 70)
p95_miss = (~cover_p95) & valid
if p95_miss.any():
    print(f"{'Company':<35} {'TrueIBNR':>9} {'P95Boot':>9} {'Shortfall':>10}")
    print("-" * 68)
    for co_idx in np.where(p95_miss)[0]:
        grcode, name = companies[co_idx]
        shortfall = true_ibnr[co_idx] - boot_p95[co_idx]
        print(f"{name[:34]:<35} {true_ibnr[co_idx]:>9,.0f} "
              f"{boot_p95[co_idx]:>9,.0f} {shortfall:>10,.0f}")
else:
    print("  None — P95 reserve was adequate for all companies.")

# -----------------------------------------------------------------------
# 9. Save full validation results
# -----------------------------------------------------------------------
with open("ibnr_validation.csv", "w") as f:
    f.write("grcode,name,true_ibnr,boot_mean,boot_std,boot_p50,boot_p75,"
            "boot_p95,boot_p99,bias_pct,posted_reserve,"
            "cover_p50,cover_p75,cover_p95,cover_posted\n")
    for co_idx, (grcode, name) in enumerate(companies):
        f.write(f"{grcode},{name.replace(',','')},{true_ibnr[co_idx]:.2f},"
                f"{boot_mean[co_idx]:.2f},{boot_std[co_idx]:.2f},"
                f"{boot_p50[co_idx]:.2f},{boot_p75[co_idx]:.2f},"
                f"{boot_p95[co_idx]:.2f},{boot_p99[co_idx]:.2f},"
                f"{bias_pct[co_idx]:.1f},{posted[co_idx]:.2f},"
                f"{int(cover_p50[co_idx])},{int(cover_p75[co_idx])},"
                f"{int(cover_p95[co_idx])},{int(cover_posted[co_idx])}\n")

print(f"\nSaved ibnr_validation.csv")
