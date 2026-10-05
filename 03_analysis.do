*==============================================================================
* BRI and Child Human Capital -- Stata analysis
* Input: bri_panel_clean.csv (from 02_data_pipeline.py)
* Packages (run once):
*   ssc install reghdfe, replace
*   ssc install ftools, replace
*   ssc install csdid, replace
*   ssc install drdid, replace
*   ssc install eventstudyinteract, replace
*   ssc install avar, replace
*   ssc install bacondecomp, replace
*   ssc install estout, replace
*   ssc install coefplot, replace
*   ssc install honestdid, replace
*==============================================================================
clear all
set more off
cd "."                                  // <- set your working folder
cap mkdir results

import delimited "bri_panel_clean.csv", clear
encode iso3, gen(id)
xtset id year

* ---- Treatment ---------------------------------------------------------------
* first_treat = MoU year (0 = never); bri = post indicator
gen g = first_treat
gen byte ever = g > 0
gen rel = year - g if ever
* Event-window dummies for Sun-Abraham (cohort-relative time, never-treated = control)
gen byte never = (g == 0)

* ---- Baseline (pre-treatment) controls, fixed at 2012 -------------------------
foreach v in ln_gdp_pc urban_share trade_gdp health_exp_gdp {
    gen b_`v'_t = `v' if year == 2012
    bysort id: egen b_`v' = max(b_`v'_t)
    drop b_`v'_t
}
gen byte lic = income == "LIC"

* ---- Outcome groups ------------------------------------------------------------
global edu   prim_netenr sec_netenr prim_compl literacy_adult gpi_prim
global health u5mr stunting wasting imm_measles imm_dpt
global mech  electricity skilled_birth emp_ratio ln_gdp_pc health_exp_gdp
global ctrl  b_ln_gdp_pc b_urban_share b_trade_gdp

* ---- 0. Descriptives -----------------------------------------------------------
estpost summarize $edu $health $mech if year <= 2012
esttab using "results/T1_summary_pre.rtf", cells("count mean sd min max") replace
tab first_treat if year == 2012

*==============================================================================
* 1. TWFE benchmark (biased with staggered timing; shown for comparison)
*==============================================================================
eststo clear
foreach y in $edu $health {
    eststo twfe_`y': reghdfe `y' bri ln_gdp_pc urban_share trade_gdp, ///
        absorb(id year) vce(cluster id)
}
esttab twfe_* using "results/T2_twfe.rtf", b(3) se(3) star(* .10 ** .05 *** .01) ///
    keep(bri) replace

* Goodman-Bacon decomposition (needs balanced panel; use one outcome)
preserve
    keep if inrange(year, 2005, 2023)
    keep if !missing(prim_netenr)
    bysort id: gen nobs = _N
    keep if nobs == 19
    cap bacondecomp prim_netenr bri, ddetail
restore

*==============================================================================
* 2. Callaway-Sant'Anna (doubly robust, not-yet-treated comparison)
*==============================================================================
foreach y in $edu $health {
    di as txt "=== CS-DID: `y' ==="
    cap noi csdid `y' $ctrl, ivar(id) time(year) gvar(g) ///
        method(dripw) notyet agg(simple)
    cap noi estimates store cs_`y'

    cap noi csdid `y' $ctrl, ivar(id) time(year) gvar(g) ///
        method(dripw) notyet agg(event)
    cap noi csdid_plot, title("Event study: `y'") ///
        name(es_`y', replace)
    cap noi graph export "results/F_es_`y'.png", replace width(1600)
}
cap esttab cs_* using "results/T3_csdid.rtf", b(3) se(3) star(* .10 ** .05 *** .01) replace

* Pre-trend (joint) test
csdid prim_netenr $ctrl, ivar(id) time(year) gvar(g) method(dripw) notyet agg(event)
cap estat pretrend

*==============================================================================
* 3. Sun-Abraham interaction-weighted event study
*==============================================================================
forvalues k = 1/6 {
    gen L`k'_ev = (rel == `k')
    gen F`k'_ev = (rel == -`k')
}
gen L0_ev = (rel == 0)
drop F1_ev                                    // reference period t = -1
foreach y in prim_netenr u5mr stunting {
    cap noi eventstudyinteract `y' L*_ev F*_ev, cohort(g) control_cohort(never) ///
        absorb(id year) vce(cluster id)
    cap noi matrix C = e(b_iw)
    cap noi coefplot, vertical keep(F* L*) yline(0) ///
        title("Sun-Abraham: `y'") name(sa_`y', replace)
    cap noi graph export "results/F_sa_`y'.png", replace width(1600)
}

*==============================================================================
* 4. Mechanisms
*==============================================================================
* 4a. First stage: BRI -> infrastructure / income / health access
foreach m in $mech {
    cap noi csdid `m' $ctrl, ivar(id) time(year) gvar(g) method(dripw) notyet agg(simple)
}

* 4b. Mediation (sequential TWFE): does adding mediators shrink the BRI effect?
foreach y in prim_netenr u5mr stunting {
    eststo m0_`y': reghdfe `y' bri, absorb(id year) vce(cluster id)
    eststo m1_`y': reghdfe `y' bri electricity, absorb(id year) vce(cluster id)
    eststo m2_`y': reghdfe `y' bri electricity skilled_birth ln_gdp_pc emp_ratio, ///
        absorb(id year) vce(cluster id)
}
esttab m0_* m1_* m2_* using "results/T4_mediation.rtf", b(3) se(3) ///
    keep(bri electricity skilled_birth ln_gdp_pc emp_ratio) replace
* Interpretation: suggestive evidence only (mediators are not randomly assigned).

*==============================================================================
* 5. Heterogeneity
*==============================================================================
gen byte low_base = b_ln_gdp_pc < .
summ b_ln_gdp_pc, detail
replace low_base = (b_ln_gdp_pc < r(p50)) if !missing(b_ln_gdp_pc)

foreach y in prim_netenr sec_netenr u5mr stunting {
    eststo h_lic_`y':  reghdfe `y' c.bri##i.lic,      absorb(id year) vce(cluster id)
    eststo h_base_`y': reghdfe `y' c.bri##i.low_base, absorb(id year) vce(cluster id)
}
* Gender: use GPI and (if available) sex-specific outcomes
eststo h_gpi: reghdfe gpi_prim bri ln_gdp_pc, absorb(id year) vce(cluster id)
esttab h_* using "results/T5_heterogeneity.rtf", b(3) se(3) replace

*==============================================================================
* 6. Robustness
*==============================================================================
* 6a. Imputation robustness: drop country-years with imputed outcome
foreach y in prim_netenr u5mr stunting {
    cap noi csdid `y' $ctrl if imp_`y' == 0, ivar(id) time(year) gvar(g) ///
        method(dripw) notyet agg(simple)
}
* 6b. Placebo: shift treatment 3 years earlier (pre-period only)
gen g_placebo = g - 3 if g > 0
replace g_placebo = 0 if g == 0
cap noi csdid prim_netenr $ctrl if year < 2013, ivar(id) time(year) gvar(g_placebo) ///
    method(dripw) notyet agg(simple)
* 6c. Continuous exposure (needs china_infra_pc merged from AidData in Python)
cap confirm variable china_infra_pc
if _rc == 0 {
    gen exposure_post = china_infra_pc * (year >= 2013)
    reghdfe prim_netenr exposure_post ln_gdp_pc, absorb(id year) vce(cluster id)
    reghdfe u5mr        exposure_post ln_gdp_pc, absorb(id year) vce(cluster id)
}
* 6d. HonestDiD (sensitivity to parallel-trends violations): run after csdid event
*     study with:  honestdid, pre(...) post(...) mvec(0(0.5)2)

di "Done. Outputs in /results"
