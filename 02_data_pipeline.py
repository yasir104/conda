"""
BRI & Child Human Capital: open-API data pipeline
Steps: (1) API download -> (2) outlier cleaning -> (3) linear interpolation
       -> (4) multiple imputation (MICE) -> (5) KNN (k=5) -> (6) treatment vars -> export for Stata

Install:  pip install requests pandas numpy scikit-learn
NOTE: written without live API access; run it and check the printed diagnostics.
      Indicator codes marked VERIFY should be checked against the DHS /indicators endpoint.
"""
import time
import numpy as np
import pandas as pd
import requests
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

START, END = 2000, 2024
OUT = "bri_panel"          # output file prefix
DHS_API_KEY = ""           # free key from api.dhsprogram.com (optional but recommended)
M_IMPUTATIONS = 5

# ----------------------------------------------------------------------------
# 1. WORLD BANK API
# ----------------------------------------------------------------------------
WB_OUTCOMES = {
    "SE.PRM.NENR": "prim_netenr",
    "SE.SEC.NENR": "sec_netenr",
    "SE.PRM.CMPT.ZS": "prim_compl",
    "SE.ADT.LITR.ZS": "literacy_adult",
    "SE.ENR.PRIM.FM.ZS": "gpi_prim",
    "SH.DYN.MORT": "u5mr",
    "SH.STA.STNT.ZS": "stunting",
    "SH.STA.WAST.ZS": "wasting",
    "SH.IMM.MEAS": "imm_measles",
    "SH.IMM.IDPT": "imm_dpt",
}
WB_MEDIATORS = {
    "EG.ELC.ACCS.ZS": "electricity",
    "SH.STA.BRTC.ZS": "skilled_birth",
    "SH.XPD.CHEX.GD.ZS": "health_exp_gdp",
    "SL.EMP.TOTL.SP.ZS": "emp_ratio",
    "NY.GDP.PCAP.KD": "gdp_pc",
}
WB_CONTROLS = {
    "SP.URB.TOTL.IN.ZS": "urban_share",
    "NE.TRD.GNFS.ZS": "trade_gdp",
    "SP.POP.TOTL": "pop",
}
WB_ALL = {**WB_OUTCOMES, **WB_MEDIATORS, **WB_CONTROLS}


def get_json(url, params=None, retries=4):
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {k+1} ({e})")
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"Failed: {url}")


def wb_countries():
    j = get_json("https://api.worldbank.org/v2/country",
                 {"format": "json", "per_page": 400})
    rows = [{"iso3": c["id"], "country": c["name"],
             "region": c["region"]["value"], "income": c["incomeLevel"]["id"]}
            for c in j[1] if c["region"]["value"] != "Aggregates"]
    return pd.DataFrame(rows)


def wb_indicator(code, name):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    params = {"format": "json", "per_page": 20000, "date": f"{START}:{END}"}
    j = get_json(url, params)
    if len(j) < 2 or j[1] is None:
        print(f"  no data for {code}")
        return pd.DataFrame(columns=["iso3", "year", name])
    df = pd.DataFrame([{"iso3": d["countryiso3code"], "year": int(d["date"]),
                        name: d["value"]} for d in j[1]])
    return df


def build_wb_panel():
    ctry = wb_countries()
    ctry = ctry[ctry["income"].isin(["LIC", "LMC"])]   # current WB classification
    panel = (ctry.assign(key=1)
                 .merge(pd.DataFrame({"year": range(START, END + 1), "key": 1}))
                 .drop(columns="key"))
    for code, name in WB_ALL.items():
        print("WB:", code, name)
        panel = panel.merge(wb_indicator(code, name), on=["iso3", "year"], how="left")
    panel["ln_gdp_pc"] = np.log(panel["gdp_pc"])
    panel["ln_pop"] = np.log(panel["pop"])
    return panel.drop(columns=["gdp_pc", "pop"])


# ----------------------------------------------------------------------------
# 2. DHS API (subnational child outcomes, survey-wave panel)
# ----------------------------------------------------------------------------
DHS_INDICATORS = {                       # VERIFY IDs: api.dhsprogram.com/rest/dhs/indicators?f=json
    "CN_NUTS_C_HA2": "dhs_stunting",
    "CM_ECMR_C_U5M": "dhs_u5m",
    "ED_NARP_B_BTH": "dhs_prim_nar",
    "ED_NARS_B_BTH": "dhs_sec_nar",
    "HC_ELEC_H_ELC": "dhs_electricity",
}


def fetch_dhs():
    base = "https://api.dhsprogram.com/rest/dhs"
    cmap = pd.DataFrame(get_json(f"{base}/countries", {"f": "json"})["Data"])
    cmap = cmap[["DHS_CountryCode", "ISO3_CountryCode"]].rename(
        columns={"ISO3_CountryCode": "iso3"})
    out = []
    for ind, name in DHS_INDICATORS.items():
        print("DHS:", ind, name)
        params = {"indicatorIds": ind, "breakdown": "subnational", "f": "json",
                  "perpage": 5000, "page": 1}
        if DHS_API_KEY:
            params["apiKey"] = DHS_API_KEY
        while True:
            j = get_json(f"{base}/data", params)
            out += [{"DHS_CountryCode": d["DHS_CountryCode"], "year": int(d["SurveyYear"]),
                     "region": d["CharacteristicLabel"], "var": name, "value": d["Value"]}
                    for d in j["Data"]]
            if params["page"] >= j.get("TotalPages", 1):
                break
            params["page"] += 1
    df = pd.DataFrame(out).merge(cmap, on="DHS_CountryCode", how="left")
    wide = df.pivot_table(index=["iso3", "region", "year"], columns="var",
                          values="value").reset_index()
    wide.to_csv(f"{OUT}_dhs_subnational.csv", index=False)
    return wide


# ----------------------------------------------------------------------------
# 3. TREATMENT: BRI MoU year  (ILLUSTRATIVE -- VERIFY EVERY YEAR against the
#    official Belt and Road Portal / Green Finance & Development Center list)
# ----------------------------------------------------------------------------
BRI_MOU_YEAR = {
    "PAK": 2013, "KGZ": 2013, "TJK": 2015, "UZB": 2015, "KHM": 2016, "LAO": 2016,
    "BGD": 2016, "EGY": 2016, "NPL": 2017, "MMR": 2017, "LKA": 2017, "KEN": 2017,
    "ETH": 2018, "GHA": 2018, "NGA": 2018, "SEN": 2018, "TZA": 2018, "UGA": 2018,
    "ZMB": 2018,
}
# Countries absent from the dict are coded never-treated (g = 0). CHECK that each
# absent country truly never signed; add the rest of your sample to the dict.


def add_treatment(df):
    df["first_treat"] = df["iso3"].map(BRI_MOU_YEAR).fillna(0).astype(int)
    df["bri"] = ((df["first_treat"] > 0) & (df["year"] >= df["first_treat"])).astype(int)
    df["rel_time"] = np.where(df["first_treat"] > 0, df["year"] - df["first_treat"], np.nan)
    # Optional: continuous exposure from AidData GCDF v3 -> set path/URL below
    # gcdf = pd.read_csv("<AIDDATA_GCDF_URL_OR_PATH>")
    # (filter sectors Transport/Energy, commitment year >= 2013, sum per iso3-year,
    #  divide by population, merge as 'china_infra_pc')
    return df


# ----------------------------------------------------------------------------
# 4. CLEANING: outliers -> linear interp -> MICE -> KNN(5)
# ----------------------------------------------------------------------------
PCT_VARS = ["prim_netenr", "sec_netenr", "prim_compl", "literacy_adult", "stunting",
            "wasting", "imm_measles", "imm_dpt", "electricity", "skilled_birth",
            "emp_ratio", "urban_share"]


def flag_outliers(df, cols, thresh=3.5):
    """Within-country modified z-score (median/MAD); outliers set to NaN."""
    df = df.copy()
    for c in cols:
        if c in PCT_VARS:                                  # impossible values
            df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan
        g = df.groupby("iso3")[c]
        med = g.transform("median")
        mad = g.transform(lambda s: np.nanmedian(np.abs(s - np.nanmedian(s))))
        n = g.transform("count")
        z = 0.6745 * (df[c] - med) / mad.replace(0, np.nan)
        bad = (z.abs() > thresh) & (n >= 6)
        print(f"  outliers {c}: {int(bad.sum())}")
        df.loc[bad, c] = np.nan
    return df


def impute_pipeline(df, cols, max_gap=5, mice_max_miss=0.40, knn_max_miss=0.75):
    df = df.sort_values(["iso3", "year"]).copy()
    stage = pd.DataFrame(0, index=df.index, columns=cols)   # 0 obs,1 interp,2 MICE,3 KNN
    orig_na = df[cols].isna()

    # (a) linear interpolation within country, interior gaps only (no extrapolation)
    for c in cols:
        df[c] = df.groupby("iso3")[c].transform(
            lambda s: s.interpolate(method="linear", limit=max_gap, limit_area="inside"))
    stage[orig_na & df[cols].notna()] = 1

    miss = df[cols].isna().mean()
    print("Missing share after interpolation:\n", miss.round(2).to_string())
    drop = miss[miss > knn_max_miss].index.tolist()
    if drop:
        print("Dropped (too sparse):", drop)
    cols_ok = [c for c in cols if c not in drop]
    mice_cols = [c for c in cols_ok if miss[c] <= mice_max_miss]

    # (b) multiple imputation (MICE-style, posterior draws), m datasets + mean
    aux = ["year"]
    X = df[mice_cols + aux].copy()
    na_before = df[mice_cols].isna()
    imps = []
    for m in range(M_IMPUTATIONS):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                               max_iter=20, random_state=100 + m, skip_complete=True)
        arr = imp.fit_transform(X)
        imps.append(pd.DataFrame(arr[:, :len(mice_cols)], index=df.index, columns=mice_cols))
    mice_mean = sum(imps) / M_IMPUTATIONS
    for c in mice_cols:
        df[c] = df[c].fillna(mice_mean[c])
    for c in mice_cols:
        stage.loc[na_before[c] & df[c].notna(), c] = 2
    for m, d in enumerate(imps, 1):                          # save for Stata -mi-
        tmp = df[["iso3", "year"]].join(d.where(na_before, df[mice_cols]))
        tmp.to_csv(f"{OUT}_mice_m{m}.csv", index=False)

    # (c) KNN (k = 5) on standardized data for whatever is still missing
    remaining = df[cols_ok].isna()
    if remaining.values.any():
        sc = StandardScaler()
        Z = sc.fit_transform(df[cols_ok + aux])
        Zk = KNNImputer(n_neighbors=5, weights="distance").fit_transform(Z)
        back = pd.DataFrame(sc.inverse_transform(Zk)[:, :len(cols_ok)],
                            index=df.index, columns=cols_ok)
        for c in cols_ok:
            df[c] = df[c].fillna(back[c])
        for c in cols_ok:
            stage.loc[remaining[c] & df[c].notna(), c] = 3

    # bounds after imputation
    for c in cols_ok:
        if c in PCT_VARS:
            df[c] = df[c].clip(0, 100)
    for c in cols:
        df[f"imp_{c}"] = stage[c]
    print("\nShare of values filled by stage (1=interp, 2=MICE, 3=KNN):")
    print(stage.apply(lambda s: s.value_counts(normalize=True)).fillna(0).round(3))
    return df.drop(columns=drop), cols_ok


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    panel = build_wb_panel()
    panel = add_treatment(panel)
    panel.to_csv(f"{OUT}_raw.csv", index=False)

    value_cols = [v for v in {**WB_OUTCOMES, **WB_MEDIATORS, **WB_CONTROLS}.values()
                  if v in panel.columns and v not in ("gdp_pc", "pop")] + ["ln_gdp_pc", "ln_pop"]
    value_cols = list(dict.fromkeys(value_cols))

    panel = flag_outliers(panel, value_cols)
    clean, kept = impute_pipeline(panel, value_cols)
    clean.to_csv(f"{OUT}_clean.csv", index=False)          # <- Stata input
    print("Saved", f"{OUT}_clean.csv", clean.shape)

    try:
        fetch_dhs()
    except Exception as e:
        print("DHS step failed (check key/indicator IDs):", e)
