# ==============================================================================
# GHI FORECASTING — Feature Engineering & Selection Pipeline
# Dataset: Kaohsiung, 10-min resolution, Year 2020
# Target : GHI (Global Horizontal Irradiation)
# ==============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==============================================================================
# SECTION 0 — CONSTANTS
# ==============================================================================

TARGET = "GHI"

# ── Features EXPLICITLY EXCLUDED (with reasons) ───────────────────────────────
#
# LEAKAGE — derived from / functionally identical to GHI:
#   Transmitted Plane Of Array Irradiance  → direct irradiance sub-component of GHI
#   DHI                                    → diffuse sub-component of GHI
#   Clearsky GHI / Chi GHI / Classy GHI   → modelled GHI upper-bound; leaks target
#   Plan of Array Irradiance               → POA = GHI * tilt factor
#
# SIMULATED / MODEL OUTPUTS (not real observations):
#   All Simulated* columns, Model Temperature, Real Output Power
#
# SENSOR / SITE METADATA (static or near-static — zero predictive variance):
#   Latitude, Longitude, Altitude
#
# REDUNDANT GEOMETRY (collinear with Solar Zenith Angle, r > 0.99):
#   Apparent Zenith, Zenith, Apparent Elevation, Elevation, Airmass Absolute
#   Angle of Incidence, Equation of Time
#
# WIND (no physical mechanism linking wind to broadband irradiance attenuation):
#   Wind Speed, Wind Direction, Wx, Wy
#
# TEMPORAL REDUNDANCY:
#   Year sin / Year cos  — 1-year dataset has no inter-annual variation to learn;
#                          these columns are near-constant seasonal proxies that
#                          add noise rather than signal for a single-year model.
#   Hour Angle           — geometric duplicate of Day sin/cos
#   Week of Year, Day Number, Day Length, Azimuth — superseded by Day sin/cos
#
# FLAGS:
#   All *Flag* columns   — QC metadata, not physical predictors
#
# DCS:
#   All DCS* columns     — system-specific output metrics, not meteorological inputs

# ── Tier 1: Solar geometry (physically irreducible) ───────────────────────────
TIER1 = [
    "Solar Zenith Angle",      # GHI ∝ cos(SZA) — primary geometric driver
    "Airmass Relative",        # optical path length through atmosphere
    "Sun Up Over Horizon",     # binary night/day gate (derived from SZA < 90°)
]

# ── Tier 2: Atmospheric attenuation ───────────────────────────────────────────
TIER2 = [
    "Cloud Type",              # strongest cloud-state discriminator
    "Precipitable Water",      # dominant water-vapour absorber
    "Ozone",                   # UV/visible band absorption
    "Surface Albedo",          # ground reflection contributing to diffuse field
    "Pressure",                # Rayleigh scattering density effect
]

# ── Tier 3: Meteorological state ──────────────────────────────────────────────
TIER3 = [
    "Temperature",             # correlates with clear-sky frequency
    "Relative Humidity",       # proxy for aerosol / cloud loading
]

# ── Tier 4: Intra-day temporal encoding ───────────────────────────────────────
# NOTE: Only Day sin/cos — Year sin/cos excluded (single-year dataset).
TIER4 = [
    "Day sin",                 # time-of-day cycle — MUST keep as a pair
    "Day cos",                 # with Day sin
]

# ── Final curated feature set ─────────────────────────────────────────────────
FEATURES_CURATED = TIER1 + TIER2 + TIER3 + TIER4

# Exhaustive block-list used in load_and_prep to filter the raw DataFrame.
# Matches on exact name OR substring (case-insensitive) where marked with *.
BLOCKLIST_EXACT = {
    # Leakage
    "Transmitted Plane Of Array Irradiance",
    "DHI",
    "Clearsky GHI",
    "Plan of Array Irradiance",
    # GHI variants
    "Chi GHI", "Classy GHI",
    # Simulated / model
    "Real Output Power",
    "Model Temperature",
    # Geometry redundant
    "Apparent Zenith", "Zenith", "Apparent Elevation", "Elevation",
    "Airmass Absolute", "Angle of Incidence", "Equation of Time",
    "Hour Angle", "Azimuth", "Week of Year", "Day Number", "Day Length",
    # Wind
    "Wind Speed", "Wind Direction", "Wx", "Wy",
    # Temporal (excluded for single-year)
    "Year sin", "Year cos",
    # Metadata
    "Latitude", "Longitude", "Altitude",
}

BLOCKLIST_SUBSTRINGS = [
    "simulated",   # catches all Simulated* columns
    "flag",        # catches all *Flag* columns
    "dcs",         # catches all DCS* columns
]


# ==============================================================================
# SECTION 1 — DATA LOADING & BASIC PREP
# ==============================================================================

def _is_blocked(col: str) -> bool:
    """Return True if the column matches any blocklist rule."""
    if col in BLOCKLIST_EXACT:
        return True
    col_lower = col.lower()
    return any(sub in col_lower for sub in BLOCKLIST_SUBSTRINGS)


def load_and_prep(csv_path: str, year: int = 2020) -> pd.DataFrame:
    """
    Load raw CSV, set DatetimeIndex, resample to 10-min, filter to `year`.
    Automatically drops all blocked columns at load time.
    """
    df = pd.read_csv(csv_path)
    df.index = pd.to_datetime(df["Date Time"], format="ISO8601")
    df.drop(columns=["Date Time"], inplace=True)
    df = df.sort_index()
    df = df.select_dtypes(include="number").resample("10min").mean()

    # Drop blocked columns immediately
    blocked_present = [c for c in df.columns if _is_blocked(c)]
    if blocked_present:
        print(f"[load_and_prep] Dropping {len(blocked_present)} blocked columns:")
        for c in blocked_present:
            print(f"   ✗ {c}")
        df.drop(columns=blocked_present, inplace=True)

    start = f"{year}-01-01 00:00:00+08:00"
    end   = f"{year}-12-31 23:50:00+08:00"
    df_year = df.loc[start:end].copy()

    print(f"\n[load_and_prep] Remaining columns : {df_year.shape[1]}")
    print(f"[load_and_prep] Period            : "
          f"{df_year.index.min().date()} → {df_year.index.max().date()}")
    print(f"[load_and_prep] Rows              : {len(df_year):,}")
    return df_year


# ==============================================================================
# SECTION 2 — CYCLIC FEATURE ENGINEERING
# ==============================================================================

def engineer_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Day sin/cos from the DatetimeIndex.
    Overwrites any existing columns to guarantee correctness.

    Why Day sin/cos only (no Year sin/cos):
        With a single year of data there is no inter-annual variation to model.
        Year sin/cos encode the seasonal envelope, which is already captured by
        Solar Zenith Angle (SZA changes monotonically with season). Adding Year
        sin/cos would introduce near-zero-variance columns after standardisation
        and inflate VIF without adding predictive information.

    Encoding:
        θ_day = 2π * (minutes_since_midnight / 1440)
        Continuous across the midnight boundary; preferred over raw hour/minute.
    """
    df = df.copy()

    minutes = df.index.hour * 60 + df.index.minute
    df["Day sin"] = np.sin(2 * np.pi * minutes / 1440)
    df["Day cos"] = np.cos(2 * np.pi * minutes / 1440)

    # Sun Up Over Horizon — binarise from Solar Zenith Angle
    if "Solar Zenith Angle" in df.columns:
        df["Sun Up Over Horizon"] = (df["Solar Zenith Angle"] < 90).astype(int)

    return df


# ==============================================================================
# SECTION 3 — FEATURE AVAILABILITY CHECK & SELECTION
# ==============================================================================

def select_features(df: pd.DataFrame,
                    feature_list: list,
                    target: str = TARGET) -> tuple:
    """
    Returns (model_df, available, missing).
    Drops zero-variance columns and prints a report.
    """
    available = [c for c in feature_list if c in df.columns]
    missing   = [c for c in feature_list if c not in df.columns]

    if missing:
        print(f"⚠  Columns missing in DataFrame — check exact names:\n   {missing}")

    model_df = df[available + [target]].copy().dropna()

    # Drop zero-variance
    const = [c for c in model_df.columns if model_df[c].nunique() < 2]
    if const:
        print(f"   Dropping zero-variance: {const}")
        model_df.drop(columns=const, inplace=True)
        available = [c for c in available if c not in const]

    print(f"✅ Features selected : {len(available)}")
    print(f"   Rows after dropna : {len(model_df):,}")
    return model_df, available, missing


# ==============================================================================
# SECTION 4 — MULTICOLLINEARITY AUDIT (VIF)
# ==============================================================================

def vif_audit(df: pd.DataFrame,
              features: list,
              threshold: float = 10.0) -> pd.DataFrame:
    """
    Compute VIF for every feature, flag those above threshold.
    Diagnostic only — does NOT drop any columns.
    """
    X  = df[features].dropna()
    Xs = StandardScaler().fit_transform(X)

    vifs = pd.Series(
        [variance_inflation_factor(Xs, i) for i in range(len(features))],
        index=features,
        name="VIF",
    ).sort_values(ascending=False)

    vifs_df = vifs.to_frame()
    vifs_df["flag"] = vifs_df["VIF"].apply(
        lambda v: "🔴 HIGH — consider dropping" if v > threshold
        else ("🟡 moderate" if v > 5 else "🟢 OK")
    )
    print("\n=== VIF Audit ===")
    print(vifs_df.to_string())
    return vifs_df


def plot_correlation_matrix(df: pd.DataFrame,
                             features: list,
                             target: str = TARGET,
                             figsize: tuple = (16, 13)) -> None:
    """Lower-triangle heatmap + sorted bar chart of Pearson r vs GHI."""

    corr = df[features + [target]].corr()

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={"width_ratios": [2, 1]})

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=axes[0],
                annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.3,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Feature Correlation Matrix — lower triangle", pad=10)

    ghi_corr = corr[target].drop(target).sort_values(key=abs, ascending=True)
    colors   = ["#d73027" if v > 0 else "#4575b4" for v in ghi_corr]
    axes[1].barh(ghi_corr.index, ghi_corr.values, color=colors)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Pearson r with GHI")
    axes[1].set_title("Feature vs GHI Correlation")

    plt.tight_layout()
    plt.savefig("./correlation_analysis.png", bbox_inches="tight")
    plt.show()
    print("Saved → ./correlation_analysis.png")


# ==============================================================================
# SECTION 5 — TIME-AWARE TRAIN / TEST SPLIT
# ==============================================================================

def time_split(model_df: pd.DataFrame,
               features: list,
               target: str = TARGET,
               test_ratio: float = 0.20):
    """
    Chronological split — NO shuffling (prevents temporal leakage).
    Returns X_train, X_test, y_train, y_test.
    """
    X = model_df[features]
    y = model_df[target]

    idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:idx], X.iloc[idx:]
    y_train, y_test = y.iloc[:idx], y.iloc[idx:]

    print(f"Train: {X_train.shape}  "
          f"{X_train.index.min().date()} → {X_train.index.max().date()}")
    print(f"Test : {X_test.shape}   "
          f"{X_test.index.min().date()} → {X_test.index.max().date()}")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 6 — MODEL TRAINING
# ==============================================================================

def train_random_forest(X_train, y_train,
                        n_estimators: int = 300) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train,
                  n_estimators: int = 500) -> XGBRegressor:
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False)
    return xgb


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    pred = model.predict(X_test)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE" : float(mean_absolute_error(y_test, pred)),
        "R²"  : float(r2_score(y_test, pred)),
    }
    print(f"\n{name} — Test Metrics")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# ==============================================================================
# SECTION 7 — FEATURE IMPORTANCE (RF-MDI, XGB-Gain, Permutation, SHAP)
# ==============================================================================

def compute_all_importances(rf_model, xgb_model,
                             X_train, X_test, y_test,
                             shap_sample: int = 1500) -> pd.DataFrame:
    """
    Returns a DataFrame with 5 importance scores + avg_rank column.

    Columns
    -------
    RF_MDI         — mean decrease in impurity
    XGB_gain       — average gain per split
    RF_permutation — model-agnostic; measures predictive loss on test set
    RF_SHAP        — mean |SHAP| from RF
    XGB_SHAP       — mean |SHAP| from XGBoost
    avg_rank       — average rank across all 5 metrics (lower = more important)
    """
    features = list(X_train.columns)

    rf_mdi   = pd.Series(rf_model.feature_importances_,  index=features)
    xgb_gain = pd.Series(xgb_model.feature_importances_, index=features)

    perm     = permutation_importance(
        rf_model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_perm  = pd.Series(perm.importances_mean, index=features)

    sample_idx = (X_test.sample(min(shap_sample, len(X_test)),
                                random_state=RANDOM_STATE)
                        .sort_index())

    rf_explainer  = shap.TreeExplainer(rf_model)
    xgb_explainer = shap.TreeExplainer(xgb_model)

    rf_shap_vals  = rf_explainer.shap_values(sample_idx)
    xgb_shap_vals = xgb_explainer.shap_values(sample_idx)

    rf_shap  = pd.Series(np.abs(rf_shap_vals).mean(axis=0),  index=features)
    xgb_shap = pd.Series(np.abs(xgb_shap_vals).mean(axis=0), index=features)

    table = pd.DataFrame({
        "RF_MDI"        : rf_mdi,
        "XGB_gain"      : xgb_gain,
        "RF_permutation": rf_perm,
        "RF_SHAP"       : rf_shap,
        "XGB_SHAP"      : xgb_shap,
    })
    ranks = table.rank(ascending=False)
    table["avg_rank"] = ranks.mean(axis=1)
    table = table.sort_values("avg_rank")

    # Cache SHAP values for plotting
    table._rf_shap  = (rf_shap_vals, sample_idx)
    table._xgb_shap = (xgb_shap_vals, sample_idx)

    return table


def plot_importance_dashboard(importance_table: pd.DataFrame,
                               top_n: int = 12) -> None:
    """
    4-panel dashboard:
      [A] RF MDI vs XGB gain
      [B] RF Permutation importance
      [C] Combined avg_rank bar
      [D] RF SHAP beeswarm
      [E] XGB SHAP beeswarm
    """
    top          = importance_table.head(top_n)
    features_top = top.index.tolist()

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.45)

    # Panel A
    ax_bar = fig.add_subplot(gs[0, 0])
    x, w   = np.arange(len(features_top)), 0.38
    ax_bar.barh(x - w/2, top["RF_MDI"],   w, label="RF MDI",   color="#4393c3")
    ax_bar.barh(x + w/2, top["XGB_gain"], w, label="XGB gain", color="#d6604d")
    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels(features_top, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Importance")
    ax_bar.set_title(f"RF MDI vs XGB Gain — Top {top_n}")
    ax_bar.legend(fontsize=8)

    # Panel B
    ax_perm = fig.add_subplot(gs[0, 1])
    ax_perm.barh(features_top[::-1], top["RF_permutation"][::-1], color="#74c476")
    ax_perm.set_xlabel("Mean accuracy decrease")
    ax_perm.set_title(f"RF Permutation Importance — Top {top_n}")

    # Panel C
    ax_rank     = fig.add_subplot(gs[0, 2])
    rank_sorted = importance_table["avg_rank"].sort_values()
    cmap_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(rank_sorted)))
    ax_rank.barh(rank_sorted.index[::-1], rank_sorted.values[::-1],
                  color=cmap_colors[:len(rank_sorted)])
    ax_rank.set_xlabel("Average Rank (lower = more important)")
    ax_rank.set_title("Combined Rank (all 5 metrics)")

    # Panel D — RF SHAP beeswarm
    ax_rf_shap = fig.add_subplot(gs[1, :2])
    plt.sca(ax_rf_shap)
    rf_shap_vals, X_shap = importance_table._rf_shap
    shap.summary_plot(rf_shap_vals, X_shap, max_display=top_n,
                       show=False, plot_size=None)
    ax_rf_shap.set_title("RF SHAP — Beeswarm", pad=8)

    # Panel E — XGB SHAP beeswarm
    ax_xgb_shap = fig.add_subplot(gs[1, 2])
    plt.sca(ax_xgb_shap)
    xgb_shap_vals, X_shap = importance_table._xgb_shap
    shap.summary_plot(xgb_shap_vals, X_shap, max_display=top_n,
                       show=False, plot_size=None)
    ax_xgb_shap.set_title("XGB SHAP — Beeswarm", pad=8)

    plt.savefig("./feature_importance_dashboard.png", bbox_inches="tight")
    plt.show()
    print("Saved → ./feature_importance_dashboard.png")


# ==============================================================================
# SECTION 8 — FINAL FEATURE SELECTION & EXPORT
# ==============================================================================

def select_final_features(importance_table: pd.DataFrame,
                           top_n: int = 10,
                           df_columns: list = None) -> list:
    """
    Pick top-N by avg_rank.
    Enforces Day sin/cos pair completeness.
    """
    sin_cos_pairs = {"Day sin": "Day cos", "Day cos": "Day sin"}

    top    = importance_table.head(top_n).index.tolist()
    extras = []
    for f in top:
        partner = sin_cos_pairs.get(f)
        if partner and partner not in top:
            extras.append(partner)
            print(f"ℹ  Added '{partner}' to preserve Day sin/cos pair")
    top = top + extras

    if df_columns:
        top = [f for f in top if f in df_columns]

    print(f"\nFinal {len(top)} features selected:")
    for i, f in enumerate(top, 1):
        rank = importance_table.loc[f, "avg_rank"] if f in importance_table.index else "—"
        print(f"  {i:2d}. {f:<45} avg_rank={rank}")
    return top


def export_forecast_dataset(df_year: pd.DataFrame,
                             top_features: list,
                             target: str = TARGET,
                             output_path: str = "./ghi_forecast_ready.csv") -> pd.DataFrame:
    """
    Build and export the production forecast CSV.
    Column layout: date | feature_1 … feature_N | OT
    'OT' is the standard target column for TimeXer / TSLib.
    """
    out = df_year[top_features + [target]].dropna().copy()

    out.insert(0, "date",
        [f"{ts.year}/{ts.month}/{ts.day} {ts.hour}:{ts.minute:02d}"
         for ts in out.index])

    out = out.rename(columns={target: "OT"})
    out.to_csv(output_path, index=False)

    print(f"\n✅ Exported: {output_path}")
    print(f"   Shape  : {out.shape}")
    print(f"   Columns: {list(out.columns)}")
    return out


# ==============================================================================
# SECTION 9 — PIPELINE ORCHESTRATOR
# ==============================================================================

def run_pipeline(csv_path: str,
                 year: int = 2020,
                 top_n: int = 10,
                 output_path: str = "./ghi_forecast_ready.csv") -> dict:
    """
    End-to-end pipeline. Call this from a single notebook cell.

    Returns a results dict with keys:
        model_df, importance_table, top_features,
        rf_model, xgb_model, metrics, forecast_df, vif_df
    """
    print("=" * 70)
    print("  GHI FORECASTING — FEATURE SELECTION PIPELINE")
    print("=" * 70)

    # 0. Load & auto-drop blocked columns
    df_year = load_and_prep(csv_path, year=year)

    # 1. Engineer Day sin/cos and Sun Up Over Horizon
    df_year = engineer_cyclic_features(df_year)

    # 2. Select curated features
    model_df, available, _ = select_features(df_year, FEATURES_CURATED)

    # 3. VIF audit (diagnostic — does not modify feature set)
    vif_df = vif_audit(model_df, available)

    # 4. Correlation plots
    plot_correlation_matrix(model_df, available)

    # 5. Chronological train / test split
    X_train, X_test, y_train, y_test = time_split(model_df, available)

    # 6. Train models
    print("\n[Training Random Forest …]")
    rf_model  = train_random_forest(X_train, y_train)

    print("[Training XGBoost …]")
    xgb_model = train_xgboost(X_train, y_train)

    # 7. Evaluate
    metrics = {
        "RandomForest": evaluate_model(rf_model,  X_test, y_test, "Random Forest"),
        "XGBoost"     : evaluate_model(xgb_model, X_test, y_test, "XGBoost"),
    }

    # 8. Feature importance + SHAP
    print("\n[Computing feature importances + SHAP …]")
    importance_table = compute_all_importances(
        rf_model, xgb_model, X_train, X_test, y_test
    )
    print("\n=== Combined Importance Table ===")
    print(importance_table.drop(columns=["avg_rank"])
                          .to_string(float_format="{:.4f}".format))
    print("\n=== Average Rank (top 10) ===")
    print(importance_table["avg_rank"].head(10)
                                      .to_string(float_format="{:.2f}".format))

    # 9. Dashboard
    plot_importance_dashboard(importance_table,
                               top_n=min(top_n + 2, len(available)))

    # 10. Final feature selection
    top_features = select_final_features(
        importance_table, top_n=top_n, df_columns=list(df_year.columns)
    )

    # 11. Export
    forecast_df = export_forecast_dataset(df_year, top_features,
                                           output_path=output_path)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    return {
        "model_df"        : model_df,
        "importance_table": importance_table,
        "top_features"    : top_features,
        "rf_model"        : rf_model,
        "xgb_model"       : xgb_model,
        "metrics"         : metrics,
        "forecast_df"     : forecast_df,
        "vif_df"          : vif_df,
    }


# ==============================================================================
# USAGE — paste into a notebook cell
# ==============================================================================
# results = run_pipeline(
#     csv_path    = ".././dataset/forPaper/Kaohsiung_DCS002_All_10_Minutely.csv",
#     year        = 2020,
#     top_n       = 10,
#     output_path = "./ghi_kaohsiung_2020_final.csv",
# )
# importance_table = results["importance_table"]
# forecast_df      = results["forecast_df"]