"""
CareerLens — Comprehensive Model Test Suite
Tests: artifact loading, prediction logic, feature engineering,
       domain distribution, edge cases, and flask app startup.
"""
import json, sys, os, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).parent)   # ensure relative paths work

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = []

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((status, label, detail))
    print(f"  {status}  {label}" + (f"  →  {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ─────────────────────────────────────────────────────────────────────────────
section("1. ARTIFACT INTEGRITY")
OUT_DIR = Path("artifacts_layer")

required_artifacts = [
    "label_encoder.joblib", "preprocess.joblib", "stacking_ensemble.joblib",
    "feature_order.json", "dropped_correlated_features.json",
    "role_to_domain.json", "role_merge_map.json", "model_meta.json",
    "test_metrics.json",
]
for art in required_artifacts:
    p = OUT_DIR / art
    check(f"artifact exists: {art}", p.exists(), f"{p.stat().st_size:,} bytes" if p.exists() else "MISSING")

# ─────────────────────────────────────────────────────────────────────────────
section("2. MODEL LOADING")

try:
    le          = joblib.load(OUT_DIR / "label_encoder.joblib")
    check("LabelEncoder loads", True, f"{len(le.classes_)} classes")
except Exception as e:
    check("LabelEncoder loads", False, str(e)); sys.exit(1)

try:
    preprocess  = joblib.load(OUT_DIR / "preprocess.joblib")
    check("Preprocessor loads", True, type(preprocess).__name__)
except Exception as e:
    check("Preprocessor loads", False, str(e)); sys.exit(1)

try:
    ensemble    = joblib.load(OUT_DIR / "stacking_ensemble.joblib")
    base_model_names = list(ensemble["base_models"].keys())
    check("Stacking ensemble loads", True, f"base models: {base_model_names}")
    check("Meta model present", "meta_model" in ensemble, type(ensemble.get("meta_model")).__name__)
except Exception as e:
    check("Stacking ensemble loads", False, str(e)); sys.exit(1)

feature_order    = json.loads((OUT_DIR / "feature_order.json").read_text())
drop_corr        = json.loads((OUT_DIR / "dropped_correlated_features.json").read_text())["dropped_features"]
role_to_domain   = json.loads((OUT_DIR / "role_to_domain.json").read_text())
model_meta       = json.loads((OUT_DIR / "model_meta.json").read_text())
test_metrics_ref = json.loads((OUT_DIR / "test_metrics.json").read_text())

check("feature_order loaded", len(feature_order) > 0, f"{len(feature_order)} features")
check("drop_corr loaded", True, f"{len(drop_corr)} correlated features dropped")
check("role_to_domain loaded", len(role_to_domain) > 0, f"{len(role_to_domain)} roles")

# ─────────────────────────────────────────────────────────────────────────────
section("3. SAVED METRIC SANITY CHECK")

acc  = test_metrics_ref["accuracy"]
f1   = test_metrics_ref["macro_f1"]
top3 = test_metrics_ref["top3_acc"]
ece  = test_metrics_ref["ece"]

check("Test accuracy ≥ 0.80", acc >= 0.80, f"{acc:.4f}")
check("Macro F1 ≥ 0.80",      f1  >= 0.80, f"{f1:.4f}")
check("Top-3 accuracy ≥ 0.95",top3>= 0.95, f"{top3:.4f}")
check("ECE ≤ 0.05  (calibration)", ece <= 0.05, f"{ece:.4f}")
check("Model version present", "version" in model_meta, model_meta.get("version"))
check("n_classes == 30", model_meta["n_classes"] == 30, str(model_meta.get("n_classes")))

# ─────────────────────────────────────────────────────────────────────────────
section("4. FEATURE ENGINEERING SMOKE TEST")

# Replicate app.py add_features() inline for isolation
def add_features(X_in):
    X = X_in.copy()
    for sc, ic, p in [
        ("programming_skill","interest_dev_overall","prog"),
        ("data_analytics_skill","interest_data_overall","data_analytics"),
        ("data_science_ml_skill","interest_data_overall","data_science"),
        ("cloud_devops_skill","interest_cloud_infra_overall","cloud"),
        ("cybersecurity_skill","interest_cybersecurity","cyber"),
        ("ui_ux_design_skill","interest_ui_ux_design","uiux"),
        ("business_analysis_skill","interest_business_and_management","biz"),
    ]:
        if sc in X and ic in X:
            X[f"{p}_alignment"] = X[sc] * X[ic]
            X[f"{p}_skill_gap"]  = X[sc] - X[ic]
    for cols, name in [
        (["programming_skill","web_dev_skill","mobile_dev_skill"],"dev_composite"),
        (["data_analytics_skill","data_science_ml_skill","data_modeling_skill"],"data_composite"),
        (["cloud_devops_skill","cloud_aws","cloud_azure"],"cloud_composite"),
        (["cybersecurity_skill","siem_experience_score","vuln_assessments_done","pentest_tools_known_count"],"security_composite"),
        (["devops_docker","devops_kubernetes","devops_terraform"],"devops_composite"),
        (["embedded_c_cpp_skill","microcontroller_projects_count","rtos_experience_score","firmware_debugging_skill"],"embedded_composite"),
    ]:
        present = [c for c in cols if c in X]
        X[name] = X[present].mean(axis=1) if present else 0
    cc = [c for c in ["dev_composite","data_composite","cloud_composite","security_composite","devops_composite","embedded_composite"] if c in X]
    if cc:
        X["dominant_domain_score"] = X[cc].max(axis=1)
        X["dominant_domain_idx"]   = X[cc].values.argmax(axis=1).astype(int)
    sc2 = [c for c in X if c.endswith("_skill")]
    if len(sc2) >= 3:
        X["skill_max"]  = X[sc2].max(axis=1); X["skill_min"] = X[sc2].min(axis=1)
        X["skill_range"]= X["skill_max"]-X["skill_min"]; X["skill_mean"] = X[sc2].mean(axis=1)
        X["skill_std"]  = X[sc2].std(axis=1)
        X["skills_above_mean"] = X[sc2].gt(X["skill_mean"],axis=0).sum(axis=1)
        X["skill_focus_ratio"] = X["skill_max"]/(X["skill_mean"]+1e-6)
    ac = [c for c in ["project_count","github_commits_90d","internship_experience_count"] if c in X]
    if ac:
        X["activity_total"] = X[ac].sum(axis=1); X["activity_max"] = X[ac].max(axis=1)
        X["is_active"] = (X["activity_total"] > 0).astype(int)
    if "internship_experience_count" in X: X["has_internship"] = (X["internship_experience_count"] > 0).astype(int)
    if "certifications_total" in X:
        X["cert_level_bin"] = pd.cut(X["certifications_total"],bins=[-1,0,1,3,float("inf")],labels=[0,1,2,3]).astype(int)
        X["is_certified"] = (X["certifications_total"] > 0).astype(int)
    acad = [c for c in ["math_scores","cs_fundamentals_scores","cognitive_ability_score"] if c in X]
    if acad: X["academic_composite"] = X[acad].mean(axis=1)
    if "cgpa" in X:
        X["cgpa_normalized"] = X["cgpa"]/10.0; X["is_top_performer"] = (X["cgpa"] >= 8.5).astype(int)
    for cols, name in [
        (["projects_backend","projects_frontend","projects_fullstack"],"project_dev_total"),
        (["projects_data_analytics","projects_data_engineering","projects_ml_ai"],"project_data_total"),
        (["projects_security_defense","projects_security_offense"],"project_security_total"),
        (["projects_cloud","projects_devops"],"project_cloud_ops"),
        (["projects_mobile_android","projects_mobile_ios","projects_mobile_flutter"],"project_mobile_total"),
    ]:
        present = [c for c in cols if c in X]; X[name] = X[present].sum(axis=1) if present else 0
    for cols, name in [
        (["frontend_react","frontend_angular"],"frontend_stack"),
        (["backend_node","backend_django","backend_spring"],"backend_stack"),
        (["data_tool_spark","data_tool_airflow","data_tool_kafka"],"data_stack"),
        (["security_tool_siem","security_tool_wireshark","security_tool_burpsuite"],"security_stack"),
        (["mobile_kotlin","mobile_flutter"],"mobile_stack"),
        (["observability_prometheus","observability_grafana"],"observability_stack"),
    ]:
        present = [c for c in cols if c in X]; X[name] = X[present].sum(axis=1) if present else 0
    asec = [c for c in ["cybersecurity_skill","siem_experience_score","vuln_assessments_done","pentest_tools_known_count","incident_response_cases","compliance_frameworks_known_count"] if c in X]
    if asec: X["security_total"] = X[asec].sum(axis=1); X["is_security_focused"] = (X["security_total"] >= 5).astype(int)
    pc = [c for c in ["pentest_tools_known_count","security_tool_burpsuite","projects_security_offense"] if c in X]
    if pc: X["pentest_focused"] = X[pc].sum(axis=1)
    dc = [c for c in ["siem_experience_score","incident_response_cases","security_tool_siem","projects_security_defense"] if c in X]
    if dc: X["defensive_focused"] = X[dc].sum(axis=1)
    soft = [c for c in ["teamwork_behavior","communication_skill","learning_motivation","professional_discipline_score"] if c in X]
    if soft:
        X["soft_skills_total"] = X[soft].sum(axis=1); X["soft_skills_mean"] = X[soft].mean(axis=1)
        X["is_strong_soft"] = (X["soft_skills_mean"] >= 7).astype(int)
    return X

def aligned_proba(model, X, nc):
    p = model.predict_proba(X)
    if hasattr(model, "classes_"):
        a = np.zeros((p.shape[0], nc), dtype=float); a[:, model.classes_] = p; return a
    return p

def preprocess_input(d):
    df   = pd.DataFrame([d])
    dfe  = add_features(df)
    dfs  = dfe.drop(columns=drop_corr, errors="ignore").reindex(columns=feature_order, fill_value=0)
    return preprocess.transform(dfs)

def predict_single(input_dict):
    nc    = len(le.classes_)
    X     = preprocess_input(input_dict)
    parts = [aligned_proba(m, X, nc) for m in ensemble["base_models"].values()]
    prob  = ensemble["meta_model"].predict_proba(np.hstack(parts))[0]
    top3  = np.argsort(prob)[::-1][:3]
    return [{"role": le.classes_[i], "probability": round(float(prob[i])*100, 1),
              "domain": role_to_domain.get(le.classes_[i], "Unknown")}
            for i in top3]

# Minimal all-zero smoke test
dummy = {f: 0 for f in feature_order}
try:
    out = predict_single(dummy)
    check("Predict on all-zeros profile", len(out) == 3, f"top role: {out[0]['role']} ({out[0]['probability']}%)")
except Exception as e:
    check("Predict on all-zeros profile", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
section("5. DOMAIN DISTRIBUTION TESTS — 8 DISTINCT PROFILES")

DOMAIN_COLORS = {
    "Software Engineering":"#3b82f6",
    "Data & Artificial Intelligence":"#8b5cf6",
    "Cybersecurity":"#ef4444",
    "Cloud, DevOps & Platform Engineering":"#06b6d4",
    "UI/UX & Product":"#f59e0b",
    "Quality Assurance & Testing":"#10b981",
    "Systems & Infrastructure":"#6366f1",
}

def base_profile():
    return {f: 0 for f in feature_order}

profiles = {
    "ML Engineer": {**base_profile(), **dict(
        programming_skill=9, data_science_ml_skill=9, data_analytics_skill=8,
        interest_data_overall=9, interest_dev_overall=7,
        math_scores=90, cs_fundamentals_scores=85, cgpa=8.8,
        projects_ml_ai=5, project_count=8, github_commits_90d=200,
        data_tool_spark=1, certifications_total=3,
    )},
    "Cybersecurity Analyst": {**base_profile(), **dict(
        cybersecurity_skill=9, siem_experience_score=8, vuln_assessments_done=7,
        pentest_tools_known_count=6, incident_response_cases=5,
        interest_cybersecurity=9, compliance_frameworks_known_count=4,
        security_tool_siem=1, security_tool_wireshark=1, security_tool_burpsuite=1,
        projects_security_defense=4, projects_security_offense=3,
        certifications_total=3, cgpa=7.5,
    )},
    "Cloud/DevOps Engineer": {**base_profile(), **dict(
        cloud_devops_skill=9, cloud_aws=1, cloud_azure=1,
        devops_docker=1, devops_kubernetes=1, devops_terraform=1,
        interest_cloud_infra_overall=9,
        observability_prometheus=1, observability_grafana=1,
        projects_cloud=5, projects_devops=4, project_count=9,
        certifications_total=4, cgpa=8.0,
    )},
    "Frontend / UI Developer": {**base_profile(), **dict(
        ui_ux_design_skill=9, web_dev_skill=9, programming_skill=7,
        interest_ui_ux_design=9, frontend_react=1, frontend_angular=1,
        projects_frontend=5, projects_fullstack=3, project_count=8,
        cgpa=7.8, github_commits_90d=150,
    )},
    "QA Engineer": {**base_profile(), **dict(
        qa_testing_skill=9, programming_skill=6, web_dev_skill=5,
        testing_tool_selenium=1, testing_tool_jmeter=1,
        projects_backend=2, project_count=5, certifications_total=2,
        cgpa=7.2,
    )},
    "Embedded Systems Engineer": {**base_profile(), **dict(
        embedded_c_cpp_skill=9, rtos_experience_score=8,
        microcontroller_projects_count=6, firmware_debugging_skill=7,
        lang_c_cpp=1, projects_embedded=4, project_count=6, cgpa=8.0,
    )},
    "Data Analyst": {**base_profile(), **dict(
        data_analytics_skill=9, data_modeling_skill=8, db_sql_skill=8,
        interest_data_overall=9, programming_skill=6,
        projects_data_analytics=5, project_count=7,
        certifications_total=2, cgpa=7.9,
    )},
    "Backend Developer": {**base_profile(), **dict(
        programming_skill=9, web_dev_skill=8, api_design_skill=8,
        system_design_score=7, db_sql_skill=8,
        interest_dev_overall=9,
        backend_node=1, backend_django=1,
        projects_backend=6, project_count=10,
        github_commits_90d=250, cgpa=8.2,
    )},
}

all_predicted_domains = []
for profile_name, profile_data in profiles.items():
    try:
        preds = predict_single(profile_data)
        top   = preds[0]
        top3_roles   = [p["role"] for p in preds]
        top3_domains = [p["domain"] for p in preds]
        all_predicted_domains.append(top["domain"])
        print(f"\n  Profile: {profile_name}")
        print(f"    #1 → {top['role']} ({top['probability']}%) | Domain: {top['domain']}")
        for i, p in enumerate(preds[1:], 2):
            print(f"    #{i} → {p['role']} ({p['probability']}%) | Domain: {p['domain']}")
        check(f"{profile_name}: confidence ≥ 20%", top["probability"] >= 20, f"{top['probability']}%")
        check(f"{profile_name}: has valid domain", top["domain"] != "Unknown", top["domain"])
    except Exception as e:
        check(f"{profile_name}: prediction succeeds", False, str(e))

unique_domains = set(all_predicted_domains)
check("At least 4 distinct domains predicted", len(unique_domains) >= 4,
      f"{len(unique_domains)} unique: {sorted(unique_domains)}")

# Check Cybersecurity is NOT the top result for non-cyber profiles
non_cyber = [d for pname, d in zip(profiles.keys(), all_predicted_domains) if "Cyber" not in pname]
cyber_bias = sum(1 for d in non_cyber if "Cyber" in d)
check("No Cybersecurity bias on non-cyber profiles",
      cyber_bias == 0, f"{cyber_bias}/{len(non_cyber)} non-cyber profiles mapped to Cyber")

# ─────────────────────────────────────────────────────────────────────────────
section("6. PROBABILITY SANITY CHECKS")

for profile_name, profile_data in list(profiles.items())[:3]:
    try:
        nc   = len(le.classes_)
        X    = preprocess_input(profile_data)
        parts = [aligned_proba(m, X, nc) for m in ensemble["base_models"].values()]
        prob = ensemble["meta_model"].predict_proba(np.hstack(parts))[0]
        check(f"{profile_name}: probs sum ≈ 1.0", abs(prob.sum() - 1.0) < 1e-4, f"sum={prob.sum():.6f}")
        check(f"{profile_name}: no negative probs", (prob >= 0).all(), f"min={prob.min():.6f}")
        check(f"{profile_name}: no prob > 1.0",     (prob <= 1.0).all(), f"max={prob.max():.6f}")
    except Exception as e:
        check(f"{profile_name}: proba validation", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
section("7. EDGE CASES")

# All-max profile
max_profile = {**base_profile()}
for f in feature_order:
    if "skill" in f or "score" in f or "count" in f or "interest" in f:
        max_profile[f] = 9
try:
    preds = predict_single(max_profile)
    check("All-max profile predicts without error", True, f"top: {preds[0]['role']}")
except Exception as e:
    check("All-max profile predicts without error", False, str(e))

# Single high-signal feature
single_signal = {**base_profile(), "cybersecurity_skill": 9, "interest_cybersecurity": 9,
                  "pentest_tools_known_count": 5, "projects_security_defense": 3}
try:
    preds = predict_single(single_signal)
    check("Single-signal cyber profile returns valid output", True, f"top: {preds[0]['role']} | domain: {preds[0]['domain']}")
except Exception as e:
    check("Single-signal cyber profile returns valid output", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
section("8. FLASK APP IMPORT TEST")

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    check("Flask app.py imports without error", True, "all routes registered")
    # Check key routes exist
    flask_app = app_module.app
    rules = [r.rule for r in flask_app.url_map.iter_rules()]
    check("Route / exists",          "/" in rules)
    check("Route /predict_v2 exists","/predict_v2" in rules)
    check("Route /dashboard exists", "/dashboard" in rules)
except Exception as e:
    import traceback
    check("Flask app.py imports without error", False, str(e))
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
section("9. SUMMARY")

total  = len(results)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
warned = sum(1 for s, _, _ in results if s == WARN)

print(f"\n  Total checks : {total}")
print(f"  {PASS}  : {passed}")
print(f"  {FAIL}  : {failed}")
print(f"  {WARN} : {warned}")

if failed:
    print("\n  Failed checks:")
    for s, label, detail in results:
        if s == FAIL:
            print(f"    ❌  {label}  →  {detail}")
    sys.exit(1)
else:
    print(f"\n  🎉  All {passed} checks passed! Model is healthy.")
    sys.exit(0)
