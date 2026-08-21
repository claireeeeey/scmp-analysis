"""
SCMP churn assignment - data preparation
----------------------------------------
Reads the 4 raw CSVs and writes clean, analysis-ready tables for Google Sheets.

Outputs (into OUTPUT_DIR):
  1. user_level.csv       - one row per user (main analysis table, ~11k rows)
  2. weekly_summary.csv   - one row per week (churn events, funnel entries, engagement)
  3. funnel_summary.csv   - one row per cancellation funnel step
  4. cancel_reasons_clean.csv - one row per user who gave a reason (for the qual coding)

Run:  python scmp_prep.py
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG - change these two paths to match your machine
# ---------------------------------------------------------------------------
INPUT_DIR = "./data"
OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def strip_prefix(df):
    """Raw exports carry warehouse prefixes like 'user_g_a__'. Drop them."""
    df.columns = [
        c.replace("user_g_a_week__", "").replace("user_g_a__", "")
        for c in df.columns
    ]
    return df


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
print("Loading raw files...")

sub = strip_prefix(pd.read_csv(f"{INPUT_DIR}/churners_fy25.csv"))
met = strip_prefix(pd.read_csv(f"{INPUT_DIR}/metrics.csv"))
can = strip_prefix(pd.read_csv(f"{INPUT_DIR}/cancel_reasons.csv"))
pro = strip_prefix(pd.read_csv(f"{INPUT_DIR}/promo_code.csv"))

for df in (sub, met, can):
    df["date_week"] = pd.to_datetime(df["date_week"])
sub["first_subscription_datetime"] = pd.to_datetime(
    sub["first_subscription_datetime"], errors="coerce"
)

print(f"  churners_fy25 : {len(sub):>7,} rows / {sub.user_id.nunique():,} users")
print(f"  metrics       : {len(met):>7,} rows / {met.user_id.nunique():,} users")
print(f"  cancel_reasons: {len(can):>7,} rows / {can.user_id.nunique():,} users")
print(f"  promo_code    : {len(pro):>7,} rows / {pro.user_id.nunique():,} users")

# ---------------------------------------------------------------------------
# 2. SPLIT THE SUBSCRIPTION FILE
#    The raw file mixes two different things in one table:
#      - state rows  (action is null) -> the user's status that week
#      - event rows  (action is set)  -> something happened that week
#    Analysing them together double-counts. Split first.
# ---------------------------------------------------------------------------
print("\nSplitting state rows from event rows...")

state = (
    sub[sub["action"].isna()]
    .drop_duplicates(subset=["user_id", "date_week"])
    .loc[:, ["user_id", "date_week", "subscription_status",
             "audience_segment", "first_subscription_datetime"]]
    .copy()
)

events = sub[sub["action"].notna()].copy()

print(f"  state rows : {len(state):>7,} (one per user-week)")
print(f"  event rows : {len(events):>7,}")
print("  event types:")
print(events.groupby(["action", "action_detail"]).size().to_string())

# ---------------------------------------------------------------------------
# 3. USER SPINE - one row per user, with the attributes that never change
# ---------------------------------------------------------------------------
print("\nBuilding user spine...")

spine = (
    sub.sort_values("date_week")
    .groupby("user_id")
    .agg(
        first_subscription_datetime=("first_subscription_datetime", "first"),
        audience_segment=("audience_segment", "first"),
        first_week_seen=("date_week", "min"),
        last_week_seen=("date_week", "max"),
    )
    .reset_index()
)

# how many weeks the user shows up as Active vs Churn in the state table
status_wk = (
    state.pivot_table(
        index="user_id", columns="subscription_status",
        values="date_week", aggfunc="count",
    )
    .rename(columns={"Active": "weeks_active", "Churn": "weeks_churned"})
    .reset_index()
)
spine = spine.merge(status_wk, on="user_id", how="left")

# ---------------------------------------------------------------------------
# 4. CHURN EVENTS
#    'action == Churn' is the real churn event. Use the FIRST one per user:
#    later ones belong to a second subscription lifecycle after a winback.
# ---------------------------------------------------------------------------
print("Extracting churn events...")

churn_ev = events[events["action"] == "Churn"].copy()

first_churn = (
    churn_ev.sort_values("date_week")
    .groupby("user_id")
    .agg(
        churn_week=("date_week", "first"),
        churn_type=("action_detail", "first"),
        n_churn_events=("date_week", "size"),
    )
    .reset_index()
)

spine = spine.merge(first_churn, on="user_id", how="left")
spine["is_churned"] = spine["churn_week"].notna()

# tenure at the moment of churn
spine["tenure_days_at_churn"] = (
    spine["churn_week"] - spine["first_subscription_datetime"]
).dt.days
spine["tenure_months_at_churn"] = (spine["tenure_days_at_churn"] / 30.44).round(1)

# data-quality flag: churn recorded before the subscription started
spine["dq_negative_tenure"] = spine["tenure_days_at_churn"] < 0

# tenure buckets for pivoting in Sheets
bins = [-np.inf, 90, 180, 270, 330, 360, 375, 400, 550, 730, 1100, np.inf]
labels = ["0-3m", "3-6m", "6-9m", "9-11m", "11-12m", "~12m (renewal)",
          "12-13m", "13-18m", "18-24m", "24-36m", "36m+"]
spine["tenure_bucket_at_churn"] = pd.cut(
    spine["tenure_days_at_churn"], bins=bins, labels=labels
)

# ---------------------------------------------------------------------------
# 5. OTHER LIFECYCLE EVENTS
# ---------------------------------------------------------------------------
ev_counts = (
    events.pivot_table(index="user_id", columns="action",
                       values="date_week", aggfunc="count")
    .add_prefix("n_")
    .reset_index()
)
spine = spine.merge(ev_counts, on="user_id", how="left")

# last known auto-renew state
ar = events[events["action_detail"].isin(["Auto renew on", "Auto renew off"])]
ar_last = (
    ar.sort_values("date_week")
    .groupby("user_id")["action_detail"].last()
    .rename("last_autorenew_state").reset_index()
)
spine = spine.merge(ar_last, on="user_id", how="left")

# ---------------------------------------------------------------------------
# 6. PROMO CODE
# ---------------------------------------------------------------------------
print("Joining promo codes...")

pro_u = pro.drop_duplicates("user_id").copy()
pro_u["promo_code_lower"] = pro_u["promo_code"].str.lower()


def promo_family(code):
    """Rough grouping so you can pivot on campaign TYPE, not 400 raw codes."""
    if pd.isna(code):
        return "No promo recorded"
    c = str(code).lower()
    if "winback" in c or c.startswith("wb") or "return2scmp" in c:
        return "Winback"
    if "jan" in c:
        return "January campaign"
    if any(k in c for k in ["summersale", "singlesday", "cybermonday", "happy"]):
        return "Seasonal sale"
    if "renew" in c or "retain" in c or "preorder" in c:
        return "Renewal / retention"
    return "Other campaign"


pro_u["promo_family"] = pro_u["promo_code_lower"].apply(promo_family)
spine = spine.merge(
    pro_u[["user_id", "promo_code", "promo_family"]], on="user_id", how="left"
)
spine["has_promo"] = spine["promo_code"].notna()
spine["promo_family"] = spine["promo_family"].fillna("No promo recorded")

# ---------------------------------------------------------------------------
# 7. CANCELLATION FUNNEL + REASONS
# ---------------------------------------------------------------------------
print("Summarising cancellation funnel...")

fun = can[can["step_number"].notna()].copy()

fun_u = (
    fun.groupby("user_id")
    .agg(
        max_funnel_step=("step_number", "max"),
        n_funnel_entries=("step_number", "size"),
        first_funnel_week=("date_week", "min"),
        last_funnel_week=("date_week", "max"),
    )
    .reset_index()
)

# last stated reason per user
reason_u = (
    can[can["cancel_reason"].notna()]
    .sort_values("date_week")
    .groupby("user_id")
    .agg(cancel_reason=("cancel_reason", "last"),
         reason_step=("step_number", "last"))
    .reset_index()
)

spine = spine.merge(fun_u, on="user_id", how="left")
spine = spine.merge(reason_u[["user_id", "cancel_reason"]], on="user_id", how="left")
spine["entered_cancel_funnel"] = spine["max_funnel_step"].notna()
spine["gave_reason"] = spine["cancel_reason"].notna()

# the interesting group: started cancelling but never actually churned
spine["funnel_abandoner"] = spine["entered_cancel_funnel"] & ~spine["is_churned"]

# ---------------------------------------------------------------------------
# 8. ENGAGEMENT BEFORE THE EXIT
#    Reference week = churn week for churners, last observed week for the rest.
#    Compare the 4 weeks just before it against weeks 9-12 before it.
# ---------------------------------------------------------------------------
print("Computing pre-churn engagement windows...")

spine["ref_week"] = spine["churn_week"].fillna(spine["last_week_seen"])

met_j = met.merge(spine[["user_id", "ref_week"]], on="user_id", how="inner")
met_j["weeks_before_ref"] = (
    (met_j["ref_week"] - met_j["date_week"]).dt.days / 7
).round().astype(int)


def window_stats(df, lo, hi, suffix):
    """Mean engagement in the window [lo, hi] weeks before the reference week."""
    w = df[(df["weeks_before_ref"] >= lo) & (df["weeks_before_ref"] <= hi)]
    out = (
        w.groupby("user_id")
        .agg(
            **{
                f"engaged_time_{suffix}": ("engaged_time_per_user", "mean"),
                f"pageviews_{suffix}": ("pageviews", "mean"),
                f"quality_read_{suffix}": ("quality_reads_per_article_pageviews", "mean"),
                f"weeks_with_data_{suffix}": ("date_week", "count"),
            }
        )
        .reset_index()
    )
    return out


last4 = window_stats(met_j, 1, 4, "last4w")
prior4 = window_stats(met_j, 9, 12, "prior4w")

spine = spine.merge(last4, on="user_id", how="left")
spine = spine.merge(prior4, on="user_id", how="left")

# overall engagement across the whole observation window
overall = (
    met.groupby("user_id")
    .agg(
        engaged_time_avg=("engaged_time_per_user", "mean"),
        pageviews_avg=("pageviews", "mean"),
        quality_read_avg=("quality_reads_per_article_pageviews", "mean"),
        weeks_with_any_pageview=("pageviews", lambda s: (s > 0).sum()),
        weeks_observed=("date_week", "count"),
    )
    .reset_index()
)
spine = spine.merge(overall, on="user_id", how="left")

spine["active_week_rate"] = (
    spine["weeks_with_any_pageview"] / spine["weeks_observed"]
).round(3)

# engagement trend: negative = cooling off before the exit
spine["engaged_time_delta"] = (
    spine["engaged_time_last4w"] - spine["engaged_time_prior4w"]
).round(1)
spine["engaged_time_change_pct"] = np.where(
    spine["engaged_time_prior4w"] > 0,
    ((spine["engaged_time_last4w"] / spine["engaged_time_prior4w"]) - 1).round(3),
    np.nan,
)

# simple engagement tier for cross-tabs in Sheets
spine["engagement_tier"] = pd.qcut(
    spine["engaged_time_avg"], q=4,
    labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop",
)

# ---------------------------------------------------------------------------
# 9. TIDY AND WRITE user_level.csv
# ---------------------------------------------------------------------------
cols = [
    "user_id", "audience_segment",
    "first_subscription_datetime", "first_week_seen", "last_week_seen",
    "weeks_active", "weeks_churned",
    # churn
    "is_churned", "churn_week", "churn_type", "n_churn_events",
    "tenure_days_at_churn", "tenure_months_at_churn", "tenure_bucket_at_churn",
    # lifecycle
    "n_New", "n_Renew", "n_Auto-renew", "n_Winback", "n_Reactivation",
    "n_Redeem", "n_Refund", "n_Migration", "last_autorenew_state",
    # marketing
    "has_promo", "promo_code", "promo_family",
    # cancellation funnel
    "entered_cancel_funnel", "max_funnel_step", "n_funnel_entries",
    "funnel_abandoner", "gave_reason", "cancel_reason",
    # engagement
    "engaged_time_avg", "pageviews_avg", "quality_read_avg",
    "weeks_with_any_pageview", "weeks_observed", "active_week_rate",
    "engagement_tier",
    "engaged_time_last4w", "engaged_time_prior4w",
    "engaged_time_delta", "engaged_time_change_pct",
    "pageviews_last4w", "pageviews_prior4w",
    "quality_read_last4w", "quality_read_prior4w",
    # data quality
    "dq_negative_tenure",
]
cols = [c for c in cols if c in spine.columns]
user_level = spine[cols].copy()

for c in ["n_New", "n_Renew", "n_Auto-renew", "n_Winback", "n_Reactivation",
          "n_Redeem", "n_Refund", "n_Migration", "n_churn_events",
          "n_funnel_entries", "weeks_active", "weeks_churned"]:
    if c in user_level.columns:
        user_level[c] = user_level[c].fillna(0).astype(int)

for c in user_level.select_dtypes(include="datetime64[ns]").columns:
    user_level[c] = user_level[c].dt.strftime("%Y-%m-%d")

user_level.to_csv(f"{OUTPUT_DIR}/user_level.csv", index=False)

# ---------------------------------------------------------------------------
# 10. WEEKLY SUMMARY
# ---------------------------------------------------------------------------
print("Building weekly summary...")

wk_state = (
    state.groupby(["date_week", "subscription_status"]).size()
    .unstack(fill_value=0)
    .rename(columns={"Active": "users_active", "Churn": "users_in_churned_state"})
)

wk_events = (
    events.groupby(["date_week", "action"]).size()
    .unstack(fill_value=0).add_prefix("ev_")
)

wk_churn_type = (
    churn_ev.groupby(["date_week", "action_detail"]).size()
    .unstack(fill_value=0)
)

wk_funnel = (
    fun.groupby("date_week")["user_id"].nunique().rename("funnel_users")
)

wk_met = met.groupby("date_week").agg(
    engaged_time_avg=("engaged_time_per_user", "mean"),
    pageviews_avg=("pageviews", "mean"),
    quality_read_avg=("quality_reads_per_article_pageviews", "mean"),
)

weekly = (
    wk_state.join(wk_events, how="outer")
    .join(wk_churn_type, how="outer")
    .join(wk_funnel, how="outer")
    .join(wk_met, how="outer")
    .fillna(0)
    .reset_index()
)

weekly["weekly_churn_rate"] = (
    weekly.get("ev_Churn", 0) / weekly["users_active"].replace(0, np.nan)
).round(5)
weekly["date_week"] = weekly["date_week"].dt.strftime("%Y-%m-%d")
weekly.to_csv(f"{OUTPUT_DIR}/weekly_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 11. FUNNEL SUMMARY
# ---------------------------------------------------------------------------
funnel_summary = (
    fun.groupby("step_number")
    .agg(users_reaching_step=("user_id", "nunique"),
         total_entries=("user_id", "size"))
    .reset_index()
)
top = funnel_summary["users_reaching_step"].iloc[0]
funnel_summary["pct_of_step1"] = (
    funnel_summary["users_reaching_step"] / top
).round(3)
funnel_summary["step_to_step_dropoff"] = (
    -funnel_summary["users_reaching_step"].pct_change()
).round(3)
funnel_summary.to_csv(f"{OUTPUT_DIR}/funnel_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 12. CANCEL REASONS (for manual thematic coding in Sheets)
# ---------------------------------------------------------------------------
reasons = (
    can[can["cancel_reason"].notna()]
    .groupby("cancel_reason")
    .agg(n_mentions=("user_id", "size"), n_users=("user_id", "nunique"))
    .sort_values("n_users", ascending=False)
    .reset_index()
)
reasons["theme"] = ""      # <- you fill this column in by hand in Sheets
reasons.to_csv(f"{OUTPUT_DIR}/cancel_reasons_clean.csv", index=False)

# ---------------------------------------------------------------------------
# 13. QA PRINTOUT - sanity-check these before you trust anything
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("QA CHECKS")
print("=" * 60)
print(f"Users in user_level          : {len(user_level):,}")
print(f"Churned users (event-based)  : {int(spine.is_churned.sum()):,} "
      f"({spine.is_churned.mean():.1%})")
print(f"  Active churn               : {(spine.churn_type == 'Active Churn').sum():,}")
print(f"  Passive churn              : {(spine.churn_type == 'Passive Churn').sum():,}")
print(f"Entered cancel funnel        : {int(spine.entered_cancel_funnel.sum()):,}")
print(f"  ...but did NOT churn       : {int(spine.funnel_abandoner.sum()):,}")
print(f"Users with promo code        : {int(spine.has_promo.sum()):,}")
print(f"Negative tenure (data issue) : {int(spine.dq_negative_tenure.sum()):,}")
print(f"Missing first_subscription   : {spine.first_subscription_datetime.isna().sum():,}")
print(f"Segment = TBD                : {(spine.audience_segment == 'TBD').sum():,} "
      f"({(spine.audience_segment == 'TBD').mean():.1%})")
print("\nChurn rate by promo family:")
print(spine.groupby("promo_family").agg(
    users=("user_id", "size"), churn_rate=("is_churned", "mean")
).round(3).sort_values("users", ascending=False).to_string())
print("\nTenure bucket at churn:")
print(spine[spine.is_churned].tenure_bucket_at_churn
      .value_counts().sort_index().to_string())

print(f"\nDone. Files written to {os.path.abspath(OUTPUT_DIR)}/")
