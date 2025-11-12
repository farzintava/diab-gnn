# notebooks/01_eda.py
import os
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results" / "eda"
RES.mkdir(parents=True, exist_ok=True)

RAW_DIAB = DATA / "diabetic_data.csv"
RAW_IDS = DATA / "IDS_mapping.csv"

OUT_ENC = DATA / "encounters.parquet"
OUT_MED = DATA / "med_events.parquet"
OUT_DIA = DATA / "diag_events.parquet"


# ----------------------------
# Helpers
# ----------------------------
def load_raw():
    assert RAW_DIAB.exists(), f"Missing {RAW_DIAB}"
    assert RAW_IDS.exists(), f"Missing {RAW_IDS}"

    df = pd.read_csv(RAW_DIAB, low_memory=False)
    ids = pd.read_csv(RAW_IDS)

    # Expect typical columns in IDS_mapping (dataset may provide two maps).
    # We keep it flexible by merging on any overlapping key columns.
    # Commonly: encounter_id, patient_nbr, and their mapped equivalents.
    # We'll align on ['encounter_id','patient_nbr'] if present.
    keys = [
        c
        for c in ["encounter_id", "patient_nbr"]
        if c in ids.columns and c in df.columns
    ]
    if not keys:
        print("⚠️ IDS_mapping.csv does not share standard keys; saving raw IDs as-is.")
        df_map = df.copy()
    else:
        # Merge any mapped columns back into df (suffix to avoid collision)
        df_map = df.merge(ids, on=keys, how="left", suffixes=("", "_mapped"))
        # Prefer mapped ids if present
        for col in ["encounter_id", "patient_nbr"]:
            mcol = f"{col}_mapped"
            if mcol in df_map.columns and df_map[mcol].notna().any():
                # replace original with mapped (cast to int when possible)
                with np.errstate(all="ignore"):
                    rep = pd.to_numeric(df_map[mcol], errors="ignore")
                df_map[col] = rep
        # Drop extra mapped columns to keep table clean
        drop_cols = [c for c in df_map.columns if c.endswith("_mapped")]
        df_map = df_map.drop(columns=drop_cols)

    # Cast id cols cleanly
    for c in ["encounter_id", "patient_nbr"]:
        if c in df_map.columns:
            df_map[c] = pd.to_numeric(df_map[c], errors="ignore")

    return df_map


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names (lowercase)
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def derive_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Readmission <=30 days
    if "readmitted" in df.columns:
        df["readmit_30d"] = df["readmitted"].astype(str).str.upper() == "<30"
    else:
        df["readmit_30d"] = pd.NA

    # Mortality (discharge_disposition_id == 11 == 'Expired' in dataset docs)
    if "discharge_disposition_id" in df.columns:
        df["mortality_in_hosp"] = df["discharge_disposition_id"].astype(str) == "11"
    else:
        df["mortality_in_hosp"] = pd.NA

    # Long LOS (>= 75th percentile) using 'time_in_hospital'
    if "time_in_hospital" in df.columns:
        p75 = np.nanpercentile(
            pd.to_numeric(df["time_in_hospital"], errors="coerce"), 75
        )
        df["long_los"] = pd.to_numeric(df["time_in_hospital"], errors="coerce") >= p75
        df.attrs["los_p75"] = float(p75)
    else:
        df["long_los"] = pd.NA

    return df


def meds_columns(df: pd.DataFrame):
    # Diabetes dataset has ~24 medication columns (e.g., 'insulin', 'metformin', 'glipizide', etc.)
    # Heuristic: medication columns contain values in {'No','Steady','Up','Down'} and exclude 'change' word.
    candidates = []
    for c in df.columns:
        vals = set(map(str, df[c].dropna().unique().tolist()))
        if vals.issubset(
            {"No", "NO", "Steady", "Up", "Down", "down", "up", "steady", "no"}
        ):
            candidates.append(c)
    # filter obviously non-med columns if they slipped in
    blacklist = {"readmitted", "payer_code", "age", "gender", "race"}
    meds = [c for c in candidates if c not in blacklist]
    return sorted(meds)


def med_change_to_weight(s: str) -> float:
    s = (s or "").strip().lower()
    if s == "no":
        return 0.0
    if s == "steady":
        return 0.5
    if s == "up":
        return 1.0
    if s == "down":
        return -1.0
    return 0.0


def build_med_events(df: pd.DataFrame, med_cols: list) -> pd.DataFrame:
    """Long-format medication events: each row is (encounter_id, med_name, prescribed_bool, change_weight)"""
    rows = []
    for _, r in df[["encounter_id"] + med_cols].iterrows():
        enc = r["encounter_id"]
        for m in med_cols:
            raw = r[m]
            # prescribed if anything other than "No"
            prescribed = str(raw).strip().lower() != "no"
            w = med_change_to_weight(raw)
            if prescribed:
                rows.append((enc, m, prescribed, w))
    med_events = pd.DataFrame(
        rows, columns=["encounter_id", "med_name", "prescribed", "change_weight"]
    )
    return med_events


def build_diag_events(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format diagnoses: ICD-9 code strings in diag_1/2/3. Also derive ICD3 (first 3 chars) group."""
    out = []
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    keep = ["encounter_id"]
    for _, r in df[keep + diag_cols].iterrows():
        enc = r["encounter_id"]
        for c in diag_cols:
            v = str(r.get(c, "")).strip()
            if v and v.lower() != "nan" and v != "?":
                icd = v
                icd3 = v[:3]
                out.append((enc, c, icd, icd3))
    diag_events = pd.DataFrame(out, columns=["encounter_id", "slot", "icd9", "icd3"])
    return diag_events


def plot_missingness(df: pd.DataFrame, fname: Path):
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    plt.figure(figsize=(8, max(3, len(miss) * 0.2)))
    sns.barplot(x=miss.values, y=miss.index, orient="h")
    plt.xlabel("Missing fraction")
    plt.ylabel("Column")
    plt.title("Missingness by feature")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_distributions(df: pd.DataFrame, fname_prefix: Path):
    # A few key distributions
    def save_hist(series, title, fname):
        s = pd.to_numeric(series, errors="coerce")
        plt.figure(figsize=(6, 4))
        sns.histplot(s.dropna(), kde=False, bins=30)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    if "time_in_hospital" in df.columns:
        save_hist(
            df["time_in_hospital"],
            "Time in hospital (days)",
            fname_prefix.with_name(fname_prefix.name + "_los.png"),
        )
    if "num_lab_procedures" in df.columns:
        save_hist(
            df["num_lab_procedures"],
            "Number of lab procedures",
            fname_prefix.with_name(fname_prefix.name + "_labs.png"),
        )
    if "num_medications" in df.columns:
        save_hist(
            df["num_medications"],
            "Number of medications",
            fname_prefix.with_name(fname_prefix.name + "_num_meds.png"),
        )


def plot_categorical_counts(
    df: pd.DataFrame, cat_cols: list, out_dir: Path, prefix="cat_"
):
    for c in cat_cols:
        if c not in df.columns:
            continue
        vc = df[c].astype(str).value_counts(dropna=False).head(20)
        plt.figure(figsize=(7, 4))
        sns.barplot(x=vc.index, y=vc.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{c} (top 20)")
        plt.tight_layout()
        fname = out_dir / f"{prefix}{c}.png"
        plt.savefig(fname, dpi=200)
        plt.close()


def medication_cooccurrence_heatmap(df: pd.DataFrame, med_cols: list, fname: Path):
    # Binary matrix: prescribed = status != 'No'
    M = (
        (df[med_cols].apply(lambda s: s.astype(str).str.lower() != "no"))
        .astype(int)
        .values
    )
    co = M.T @ M  # co-occurrence counts
    meds = med_cols
    co_df = pd.DataFrame(co, index=meds, columns=meds)
    plt.figure(figsize=(max(6, 0.35 * len(meds)), max(5, 0.35 * len(meds))))
    sns.heatmap(co_df, cmap="viridis", square=True)
    plt.title("Medication Co-prescription (counts)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    return co_df


def top_icd_plot(diag_events: pd.DataFrame, fname: Path, level="icd3", topk=30):
    ct = diag_events[level].value_counts().head(topk)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=ct.values, y=ct.index, orient="h")
    plt.title(f"Top {topk} {level} codes")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def main():
    print("▶️  Loading raw CSVs ...")
    df = load_raw()
    df = normalize_cols(df)

    # Keep only one row per encounter
    if "encounter_id" not in df.columns:
        raise RuntimeError("encounter_id column is required but missing.")
    df = df.drop_duplicates(subset=["encounter_id"]).reset_index(drop=True)

    # Derive outcomes
    print("▶️  Deriving outcome flags ...")
    df = derive_outcomes(df)

    # Identify medication columns
    print("▶️  Inferring medication columns ...")
    med_cols = meds_columns(df)
    if not med_cols:
        print(
            "⚠️  No medication columns inferred automatically. You may need to specify them manually."
        )
    else:
        print(f"   Found {len(med_cols)} medication columns:\n   {med_cols}")

    # Basic EDA summaries
    print("▶️  Writing summary tables ...")
    summary = {
        "n_rows": int(len(df)),
        "n_patients_unique": (
            int(df["patient_nbr"].nunique()) if "patient_nbr" in df.columns else None
        ),
        "readmit_30d_rate": (
            float(np.mean(df["readmit_30d"]))
            if df["readmit_30d"].notna().any()
            else None
        ),
        "mortality_rate": (
            float(np.mean(df["mortality_in_hosp"]))
            if df["mortality_in_hosp"].notna().any()
            else None
        ),
    }
    (RES / "tables").mkdir(exist_ok=True, parents=True)
    with open(RES / "tables" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Missingness plot
    print("▶️  Plotting missingness ...")
    plot_missingness(df, RES / "missingness.png")

    # Key distributions
    print("▶️  Plotting distributions ...")
    plot_distributions(df, RES / "dist")

    # Categorical counts
    cat_cols = [
        c
        for c in [
            "age",
            "gender",
            "race",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "readmitted",
        ]
        if c in df.columns
    ]
    print("▶️  Plotting categorical distributions ...")
    plot_categorical_counts(df, cat_cols, RES, prefix="cat_")

    # Medication co-occurrence
    if med_cols:
        print("▶️  Computing medication co-occurrence heatmap ...")
        co_df = medication_cooccurrence_heatmap(
            df, med_cols, RES / "med_cooccurrence.png"
        )
        co_df.to_csv(RES / "tables" / "med_cooccurrence_counts.csv")

    # Build long-format medication events for downstream modeling
    print("▶️  Building long-format med_events ...")
    med_events = (
        build_med_events(df, med_cols)
        if med_cols
        else pd.DataFrame(
            columns=["encounter_id", "med_name", "prescribed", "change_weight"]
        )
    )
    med_events.to_parquet(OUT_MED, index=False)

    # Build long-format diag_events
    print("▶️  Building long-format diag_events ...")
    diag_events = build_diag_events(df)
    diag_events.to_parquet(OUT_DIA, index=False)

    # ICD plots
    if not diag_events.empty:
        print("▶️  Plotting top ICD groupings ...")
        top_icd_plot(diag_events, RES / "top_icd3.png", level="icd3", topk=30)

    # Prepare encounters table (one row per encounter) with engineered med aggregates
    print("▶️  Building encounters table ...")
    enc = df.copy()

    # Medication aggregates per encounter (how many meds prescribed; sum of change weights)
    if not med_events.empty:
        agg1 = (
            med_events.groupby("encounter_id")["prescribed"]
            .sum()
            .rename("n_meds_prescribed")
        )
        agg2 = (
            med_events.groupby("encounter_id")["change_weight"]
            .sum()
            .rename("med_change_weight_sum")
        )
        enc = enc.merge(agg1, on="encounter_id", how="left")
        enc = enc.merge(agg2, on="encounter_id", how="left")
    else:
        enc["n_meds_prescribed"] = 0
        enc["med_change_weight_sum"] = 0.0

    # ICD3 multi-hot width (how many unique ICD3 per encounter)
    if not diag_events.empty:
        icd_count = (
            diag_events.groupby("encounter_id")["icd3"].nunique().rename("n_icd3")
        )
        enc = enc.merge(icd_count, on="encounter_id", how="left")
    else:
        enc["n_icd3"] = 0

    # Save encounters table (this is your analysis-ready main table)
    enc.to_parquet(OUT_ENC, index=False)

    # Small report to console
    print("\n✅ EDA complete.")
    print(f"   Summary saved:        {RES/'tables'/'summary.json'}")
    print(f"   Missingness plot:     {RES/'missingness.png'}")
    print(f"   Med co-occur heatmap: {RES/'med_cooccurrence.png'}")
    print(f"   Top ICD3 plot:        {RES/'top_icd3.png'} (if any)")
    print(f"   Encounters parquet:   {OUT_ENC}")
    print(f"   med_events parquet:   {OUT_MED}")
    print(f"   diag_events parquet:  {OUT_DIA}")


if __name__ == "__main__":
    main()
