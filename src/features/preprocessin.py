"""
=============================================================
F1 World Championship — 
Data Merging & Preprocessing
=============================================================
Outputs:
  - df_master.parquet  →  full merged & cleaned dataset
  - df_model_ready.parquet  →  leakage-free, model-ready dataset
  - stage1_report.txt  →  summary statistics
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "dataset\\"
OUT_DIR  = "output\\"


print("=" * 60)
print("Data Merge and Cleaning")
print("=" * 60)

NA_VAL = "\\N"   # F1 dataset uses \N for missing values

results      = pd.read_csv(DATA_DIR + "results.csv",              na_values=NA_VAL)
races        = pd.read_csv(DATA_DIR + "races.csv",                na_values=NA_VAL)
drivers      = pd.read_csv(DATA_DIR + "drivers.csv",              na_values=NA_VAL)
constructors = pd.read_csv(DATA_DIR + "constructors.csv",         na_values=NA_VAL)
qualifying   = pd.read_csv(DATA_DIR + "qualifying.csv",           na_values=NA_VAL)
pit_stops    = pd.read_csv(DATA_DIR + "pit_stops.csv",            na_values=NA_VAL)
status       = pd.read_csv(DATA_DIR + "status.csv",               na_values=NA_VAL)
circuits     = pd.read_csv(DATA_DIR + "circuits.csv",             na_values=NA_VAL)
driver_std   = pd.read_csv(DATA_DIR + "driver_standings.csv",     na_values=NA_VAL)
constr_std   = pd.read_csv(DATA_DIR + "constructor_standings.csv",na_values=NA_VAL)
lap_times    = pd.read_csv(DATA_DIR + "lap_times.csv",            na_values=NA_VAL)

print(f"\n✓ Loaded all CSVs")
print(f"  results:      {results.shape}")
print(f"  races:        {races.shape}  ({races['year'].min()}–{races['year'].max()})")
print(f"  drivers:      {drivers.shape}")
print(f"  qualifying:   {qualifying.shape}")
print(f"  pit_stops:    {pit_stops.shape}")
print(f"  lap_times:    {lap_times.shape}")

# ─────────────────────────────────────────────
# 2. DATA LEAKAGE FENCE
#    Pre-race  → features (safe to use as predictors)
#    Post-race → targets / forbidden features
# ─────────────────────────────────────────────
PRE_RACE_COLS = [
    # results table — only starting conditions
    "raceId", "driverId", "constructorId", "grid",
    "statusId",
    # targets (kept, but NEVER used as input features)
    "positionOrder", "points", "laps",
    "fastestLapTime", "fastestLapSpeed", "milliseconds",
]

# Columns that would leak race-day outcomes into features
LEAKAGE_COLS = ["time", "rank", "fastestLap", "positionText",
                "position", "number"]

results_clean = results[PRE_RACE_COLS].copy()
print(f"  ✓ Dropped leakage columns from results: {LEAKAGE_COLS}")

# ─────────────────────────────────────────────
# 3. ENRICH RACES TABLE
# ─────────────────────────────────────────────
print("\n[3] Races + Circuits Merging...")

circuit_cols = ["circuitId", "name", "country", "lat", "lng", "alt"]
races_enriched = races.merge(
    circuits[circuit_cols].rename(columns={"name": "circuit_name"}),
    on="circuitId", how="left"
)

# Parse race date → year already exists; add round_in_season
races_enriched["date"] = pd.to_datetime(races_enriched["date"], errors="coerce")
races_enriched["season_progress"] = (
    races_enriched.groupby("year")["round"]
    .transform(lambda x: x / x.max())          # 0→1 within each season
)

race_cols_keep = [
    "raceId", "year", "round", "circuitId", "circuit_name",
    "country", "lat", "lng", "alt", "date", "season_progress"
]
races_final = races_enriched[race_cols_keep]
print(f"  ✓ races_enriched shape: {races_final.shape}")

# ─────────────────────────────────────────────
# 4. DRIVER & CONSTRUCTOR META
# ─────────────────────────────────────────────
print("\n[4] Driver & Constructor meta...")

driver_meta = drivers[["driverId", "forename", "surname",
                        "dob", "nationality"]].copy()
driver_meta["driver_name"] = driver_meta["forename"] + " " + driver_meta["surname"]
driver_meta["dob"] = pd.to_datetime(driver_meta["dob"], errors="coerce")
driver_meta = driver_meta.drop(columns=["forename", "surname"])

constr_meta = constructors[["constructorId", "name", "nationality"]].rename(
    columns={"name": "constructor_name", "nationality": "constructor_nationality"}
)

# ─────────────────────────────────────────────
# 5. QUALIFYING FEATURES (pre-race, safe)
# ─────────────────────────────────────────────
print("\n[5] Qualifying Features...")

def parse_laptime_ms(t_series: pd.Series) -> pd.Series:
    """Convert 'M:SS.mmm' string → milliseconds (float)."""
    def _parse(t):
        if pd.isna(t):
            return np.nan
        try:
            parts = str(t).split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60_000 + float(parts[1]) * 1000
            return float(t) * 1000
        except Exception:
            return np.nan
    return t_series.apply(_parse)

qualifying["q1_ms"] = parse_laptime_ms(qualifying["q1"])
qualifying["q2_ms"] = parse_laptime_ms(qualifying["q2"])
qualifying["q3_ms"] = parse_laptime_ms(qualifying["q3"])

# Best quali time = fastest of q1/q2/q3
qualifying["best_quali_ms"] = qualifying[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)

# Gap to pole position (per race)
pole_time = (
    qualifying.groupby("raceId")["best_quali_ms"]
    .min()
    .rename("pole_time_ms")
    .reset_index()
)
qualifying = qualifying.merge(pole_time, on="raceId", how="left")
qualifying["gap_to_pole_ms"] = qualifying["best_quali_ms"] - qualifying["pole_time_ms"]

# Reached Q3?
qualifying["reached_q3"] = qualifying["q3_ms"].notna().astype(int)

quali_feats = qualifying[[
    "raceId", "driverId", "constructorId",
    "position",           # qualifying position
    "q1_ms", "q2_ms", "q3_ms", "best_quali_ms",
    "gap_to_pole_ms", "reached_q3"
]].rename(columns={"position": "quali_position"})

print(f"  ✓ quali_feats shape: {quali_feats.shape}")

# ─────────────────────────────────────────────
# 6. PIT STOP AGGREGATES (post-race → targets only)
# ─────────────────────────────────────────────
print("\n[6] Pit stop aggregates (target group only)...")

pit_agg = (
    pit_stops.groupby(["raceId", "driverId"])
    .agg(
        pit_stop_count   = ("stop", "max"),
        avg_pit_dur_ms   = ("milliseconds", "mean"),
        min_pit_dur_ms   = ("milliseconds", "min"),
    )
    .reset_index()
)
print(f"  ✓ pit_agg shape: {pit_agg.shape}")

# ─────────────────────────────────────────────
# 7. FASTEST LAP PER RACE (target variable helper)
# ─────────────────────────────────────────────


fastest_lap = (
    lap_times.groupby(["raceId", "driverId"])["milliseconds"]
    .min()
    .rename("personal_best_lap_ms")
    .reset_index()
)
race_fastest = (
    lap_times.groupby("raceId")["milliseconds"]
    .min()
    .rename("race_fastest_lap_ms")
    .reset_index()
)
fastest_lap = fastest_lap.merge(race_fastest, on="raceId", how="left")
fastest_lap["gap_to_fastest_ms"] = (
    fastest_lap["personal_best_lap_ms"] - fastest_lap["race_fastest_lap_ms"]
)
print(f"  ✓ fastest_lap shape: {fastest_lap.shape}")

# ─────────────────────────────────────────────
# 8. PRE-RACE STANDINGS (championship points before each race)
# ─────────────────────────────────────────────
print("\n[8] Pre Race Standings")

# driver_standings raceId = standings AFTER that race
# So for race R, we want standings from the previous race
def get_pre_race_standings(standings_df, id_col, points_col, position_col, wins_col):
    """Shift standings by 1 race to get pre-race state."""
    standings_sorted = standings_df.sort_values(["raceId"]).copy()
    # Merge with race year/round
    standings_sorted = standings_sorted.merge(
        races[["raceId", "year", "round"]], on="raceId", how="left"
    )
    standings_sorted = standings_sorted.sort_values(["year", "round"])
    # Lag within each driver/constructor
    standings_sorted[f"pre_{points_col}"]   = standings_sorted.groupby(id_col)[points_col].shift(1)
    standings_sorted[f"pre_{position_col}"] = standings_sorted.groupby(id_col)[position_col].shift(1)
    standings_sorted[f"pre_{wins_col}"]     = standings_sorted.groupby(id_col)[wins_col].shift(1)
    return standings_sorted[["raceId", id_col,
                              f"pre_{points_col}",
                              f"pre_{position_col}",
                              f"pre_{wins_col}"]]

pre_driver_std = get_pre_race_standings(
    driver_std, "driverId", "points", "position", "wins"
).rename(columns={
    "pre_points":   "drv_pre_points",
    "pre_position": "drv_pre_position",
    "pre_wins":     "drv_pre_wins",
})

pre_constr_std = get_pre_race_standings(
    constr_std, "constructorId", "points", "position", "wins"
).rename(columns={
    "pre_points":   "con_pre_points",
    "pre_position": "con_pre_position",
    "pre_wins":     "con_pre_wins",
})
print(f"  ✓ pre_driver_standings shape: {pre_driver_std.shape}")
print(f"  ✓ pre_constructor_standings shape: {pre_constr_std.shape}")

# ─────────────────────────────────────────────
# 9. MASTER MERGE
# ─────────────────────────────────────────────
print("\n[9] Master Merge...")

df = (
    results_clean
    .merge(races_final,    on="raceId",                        how="left")
    .merge(driver_meta,    on="driverId",                      how="left")
    .merge(constr_meta,    on="constructorId",                 how="left")
    .merge(quali_feats,    on=["raceId","driverId","constructorId"], how="left")
    .merge(pit_agg,        on=["raceId","driverId"],           how="left")
    .merge(fastest_lap,    on=["raceId","driverId"],           how="left")
    .merge(pre_driver_std, on=["raceId","driverId"],           how="left")
    .merge(pre_constr_std, on=["raceId","constructorId"],      how="left")
    .merge(status[["statusId","status"]], on="statusId",       how="left")
)

print(f"  ✓ Master df shape: {df.shape}")

# ─────────────────────────────────────────────
# 10. NaN STRATEJİLERİ
# ─────────────────────────────────────────────
print("\n[10] NaN stratejileri...")

# --- A) Qualifying: only available from ~2003 ---
# Fill missing quali positions with median per year (pre-2003 races)
df["quali_position"] = df.groupby("year")["quali_position"].transform(
    lambda x: x.fillna(x.median())
)
df["gap_to_pole_ms"] = df["gap_to_pole_ms"].fillna(df["gap_to_pole_ms"].median())
df["reached_q3"]     = df["reached_q3"].fillna(0).astype(int)
df[["q1_ms","q2_ms","q3_ms","best_quali_ms"]] = (
    df[["q1_ms","q2_ms","q3_ms","best_quali_ms"]]
    .fillna(df[["q1_ms","q2_ms","q3_ms","best_quali_ms"]].median())
)

# --- B) Pit stops: only available from ~2011 ---
df["pit_stop_count"]  = df["pit_stop_count"].fillna(0)
df["avg_pit_dur_ms"]  = df["avg_pit_dur_ms"].fillna(df["avg_pit_dur_ms"].median())
df["min_pit_dur_ms"]  = df["min_pit_dur_ms"].fillna(df["min_pit_dur_ms"].median())

# --- C) Championship standings: 0 for first race of career ---
for col in ["drv_pre_points","drv_pre_position","drv_pre_wins",
            "con_pre_points","con_pre_position","con_pre_wins"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Rename ambiguous columns after merge
rename_map = {}
for c in df.columns:
    if c.endswith("_x"):
        rename_map[c] = c[:-2] + "_driver"
    elif c.endswith("_y"):
        rename_map[c] = c[:-2] + "_constructor"
df.rename(columns=rename_map, inplace=True)

# --- D) Fastest lap: median fill for old races ---
df["personal_best_lap_ms"] = df["personal_best_lap_ms"].fillna(
    df.groupby("year")["personal_best_lap_ms"].transform("median")
)
df["gap_to_fastest_ms"] = df["gap_to_fastest_ms"].fillna(
    df.groupby("year")["gap_to_fastest_ms"].transform("median")
)

# --- E) Circuit altitude: median fill ---
df["alt"] = df["alt"].fillna(df["alt"].median())

# --- F) Driver age at race day ---
df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25
df["driver_age"] = df["driver_age"].fillna(df["driver_age"].median())

# ─────────────────────────────────────────────
# 11. TARGET VARIABLES
# ─────────────────────────────────────────────
print("\n[11] Target Variables...")

# Target 1 — Classification: podium (top 3)
df["is_podium"] = (df["positionOrder"] <= 3).astype(int)

# Target 2 — Regression: personal best lap time (ms)
# already in df["personal_best_lap_ms"]

print(f"  Podium rate: {df['is_podium'].mean():.2%}")
print(f"  Fastest lap coverage: {df['personal_best_lap_ms'].notna().mean():.2%}")

# ─────────────────────────────────────────────
# 12. MODERN ERA FILTER (qualifying + pit data)
# ─────────────────────────────────────────────
print("\n[12] Eras...")

df_full   = df.copy()                          # 1950–2024 everything
df_modern = df[df["year"] >= 2003].copy()      # qualifying era
df_pit    = df[df["year"] >= 2011].copy()      # pit stop era

print(f"  Full dataset:       {df_full.shape}")
print(f"  Modern (2003+):     {df_modern.shape}")
print(f"  Pit-stop era (2011+): {df_pit.shape}")

# ─────────────────────────────────────────────
# 13. MODEL-READY COLUMNS
# ─────────────────────────────────────────────
MODEL_FEATURE_COLS = [
    # Race context (year/round already in ID_COLS, skip duplicates)
    "season_progress", "circuitId",
    "lat", "lng", "alt",
    # Grid / qualifying
    "grid", "quali_position", "gap_to_pole_ms",
    "best_quali_ms", "reached_q3",
    # Pit strategy
    "pit_stop_count", "avg_pit_dur_ms",
    # Championship standing before race
    "drv_pre_points", "drv_pre_position", "drv_pre_wins",
    "con_pre_points", "con_pre_position", "con_pre_wins",
    # Fastest lap context (regression target kept in TARGET_COLS)
    "gap_to_fastest_ms",
    # Driver meta
    "driver_age",
]

TARGET_COLS = ["is_podium", "personal_best_lap_ms"]
ID_COLS     = ["raceId", "driverId", "constructorId",
               "driver_name", "constructor_name",
               "year", "round", "circuit_name", "date"]

# Keep only cols that actually exist
MODEL_FEATURE_COLS = [c for c in MODEL_FEATURE_COLS if c in df.columns]

df_model_ready = df_modern[ID_COLS + MODEL_FEATURE_COLS + TARGET_COLS].copy()

# Final null check
null_pct = df_model_ready.isnull().mean()
remaining_nulls = null_pct[null_pct > 0]
if len(remaining_nulls):
    print(f"\n   Remaining nulls:")
    print(remaining_nulls.to_string())
else:
    print("\n  ✓ No nulls in model-ready dataset")

# ─────────────────────────────────────────────
# 14. SAVE
# ─────────────────────────────────────────────
print("\n[14] Saving...")

df_full.to_parquet(OUT_DIR + "df_master.parquet",      index=False)
df_model_ready.to_parquet(OUT_DIR + "df_model_ready.parquet", index=False)

print("  ✓ df_master.parquet saved")
print("  ✓ df_model_ready.parquet saved")

# ─────────────────────────────────────────────
# 15. STAGE 1 REPORT
# ─────────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════════╗
║      F1 — AŞAMA 1 RAPORU: Veri Birleştirme & Temizlik   ║
╚══════════════════════════════════════════════════════════╝

1. VERİ KAYNAKLARI
   results.csv       →  {results.shape[0]:>7,} satır
   races.csv         →  {races.shape[0]:>7,} satır   ({races['year'].min()}–{races['year'].max()})
   drivers.csv       →  {drivers.shape[0]:>7,} satır
   qualifying.csv    →  {qualifying.shape[0]:>7,} satır  (2003+ kapsamı)
   pit_stops.csv     →  {pit_stops.shape[0]:>7,} satır  (2011+ kapsamı)
   lap_times.csv     →  {lap_times.shape[0]:>7,} satır

2. JOIN ANAHTARLARI
   Ana anahtarlar:  raceId ← driverId ← constructorId
   Circuit join:    circuitId
   Status join:     statusId

3. DATA LEAKAGE ÖNLEMİ
   Düşürülen sütunlar (yarış sonrası): {LEAKAGE_COLS}
   Hedef değişkenler (ASLA feature olarak kullanılmayacak):
     - positionOrder, points, laps, milliseconds
     - fastestLapTime, fastestLapSpeed

4. NaN STRATEJİLERİ
   Qualifying (pre-2003)  → yıl içi medyan ile dolduruldu
   Pit stops (pre-2011)   → count=0, duration=medyan
   Standings (1. yarış)   → 0 ile başlatıldı
   Fastest lap (eski yarış) → yıl içi medyan
   Sürücü yaşı            → yarış tarihinden hesaplandı

5. ÇIKTI VERİLERİ
   df_master.parquet       → {df_full.shape[0]:>7,} satır × {df_full.shape[1]:>3} sütun  (1950–2024)
   df_model_ready.parquet  → {df_model_ready.shape[0]:>7,} satır × {df_model_ready.shape[1]:>3} sütun  (2003–2024)

6. HEDEF DEĞİŞKENLER
   is_podium           →  {df_model_ready['is_podium'].sum():,} pozitif / {len(df_model_ready):,} toplam  ({df_model_ready['is_podium'].mean():.1%})
   personal_best_lap_ms →  ort. {df_model_ready['personal_best_lap_ms'].mean():,.0f} ms

7. MODEL-READY ÖZELLİKLER ({len(MODEL_FEATURE_COLS)} adet)
   {', '.join(MODEL_FEATURE_COLS)}
"""

print(report)

