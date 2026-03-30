"""
CareerAI — Flask Application
All features: Predict, SHAP, PDF Export, GitHub, PDF Resume, Profile History, Admin
"""
import io, json, os, re, sqlite3, warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

warnings.filterwarnings("ignore")

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import requests as req_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import shap as shap_lib
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                     Spacer, Table, TableStyle)
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ── App & artifact loading ─────────────────────────────────────────────────────
app     = Flask(__name__)
OUT_DIR = Path("artifacts_layer")

le               = joblib.load(OUT_DIR / "label_encoder.joblib")
preprocess       = joblib.load(OUT_DIR / "preprocess.joblib")
ensemble         = joblib.load(OUT_DIR / "stacking_ensemble.joblib")
feature_order    = json.load(open(OUT_DIR / "feature_order.json"))
drop_corr        = json.load(open(OUT_DIR / "dropped_correlated_features.json"))["dropped_features"]
role_to_domain   = json.load(open(OUT_DIR / "role_to_domain.json"))
merge_map        = json.load(open(OUT_DIR / "role_merge_map.json"))
model_meta       = json.load(open(OUT_DIR / "model_meta.json"))
test_metrics     = json.load(open(OUT_DIR / "test_metrics.json"))
analytics_data   = json.load(open(OUT_DIR / "analytics_data.json"))
feature_imp      = json.load(open(OUT_DIR / "feature_importance.json"))
role_profiles    = json.load(open(OUT_DIR / "role_skill_profiles.json"))
role_percentiles = json.load(open(OUT_DIR / "role_percentiles.json"))
roadmaps         = json.load(open(OUT_DIR / "roadmaps.json"))
salary_data      = json.load(open(OUT_DIR / "salary_data.json"))
radar_profiles    = json.load(open(OUT_DIR / "radar_profiles.json"))
companies_data    = json.load(open(OUT_DIR / "companies.json"))
action_resources  = json.load(open(OUT_DIR / "action_plan_resources.json"))

nc = len(le.classes_)

# SHAP explainer
shap_explainer = None
if SHAP_AVAILABLE and (OUT_DIR / "shap_explainer.joblib").exists():
    try:
        shap_explainer = joblib.load(OUT_DIR / "shap_explainer.joblib")
    except Exception:
        pass

DOMAIN_COLORS = {
    "Software Engineering":                   "#3b82f6",
    "Data & Artificial Intelligence":         "#8b5cf6",
    "Cybersecurity":                          "#ef4444",
    "Cloud, DevOps & Platform Engineering":  "#06b6d4",
    "UI/UX & Product":                        "#f59e0b",
    "Quality Assurance & Testing":            "#10b981",
    "Systems & Infrastructure":               "#6366f1",
}
DOMAIN_ICONS = {
    "Software Engineering":                   "💻",
    "Data & Artificial Intelligence":         "🤖",
    "Cybersecurity":                          "🔐",
    "Cloud, DevOps & Platform Engineering":  "☁️",
    "UI/UX & Product":                        "🎨",
    "Quality Assurance & Testing":            "✅",
    "Systems & Infrastructure":               "🖥️",
}

# ── SQLite DB ──────────────────────────────────────────────────────────────────
DB_PATH = Path("career_app.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, name TEXT DEFAULT 'Anonymous',
        top_role TEXT, domain TEXT, confidence REAL, top3 TEXT, profile TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, name TEXT, data TEXT, result TEXT)""")
    con.commit(); con.close()

init_db()

def log_prediction(name, results, profile):
    top = results[0]
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO predictions(ts,name,top_role,domain,confidence,top3,profile) VALUES(?,?,?,?,?,?,?)",
        (datetime.now().isoformat(), name or "Anonymous", top["role"], top["domain"],
         top["probability"], json.dumps([r["role"] for r in results]), json.dumps(profile)))
    con.commit(); con.close()

def db_save_profile(name, data, result):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO profiles(ts,name,data,result) VALUES(?,?,?,?)",
        (datetime.now().isoformat(), name, json.dumps(data), json.dumps(result)))
    con.commit(); con.close()

# ── Feature engineering ────────────────────────────────────────────────────────
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
        (["programming_skill","web_dev_skill","mobile_dev_skill"],                "dev_composite"),
        (["data_analytics_skill","data_science_ml_skill","data_modeling_skill"],  "data_composite"),
        (["cloud_devops_skill","cloud_aws","cloud_azure"],                         "cloud_composite"),
        (["cybersecurity_skill","siem_experience_score","vuln_assessments_done","pentest_tools_known_count"], "security_composite"),
        (["devops_docker","devops_kubernetes","devops_terraform"],                 "devops_composite"),
        (["embedded_c_cpp_skill","microcontroller_projects_count","rtos_experience_score","firmware_debugging_skill"], "embedded_composite"),
    ]:
        present = [c for c in cols if c in X]
        X[name] = X[present].mean(axis=1) if present else 0
    cc = [c for c in ["dev_composite","data_composite","cloud_composite","security_composite","devops_composite","embedded_composite"] if c in X]
    if cc:
        X["dominant_domain_score"] = X[cc].max(axis=1)
        X["dominant_domain_idx"]   = X[cc].values.argmax(axis=1).astype(int)
    sc2 = [c for c in X if c.endswith("_skill")]
    if len(sc2) >= 3:
        X["skill_max"]  = X[sc2].max(axis=1); X["skill_min"]  = X[sc2].min(axis=1)
        X["skill_range"]= X["skill_max"] - X["skill_min"]; X["skill_mean"] = X[sc2].mean(axis=1)
        X["skill_std"]  = X[sc2].std(axis=1)
        X["skills_above_mean"] = X[sc2].gt(X["skill_mean"], axis=0).sum(axis=1)
        X["skill_focus_ratio"] = X["skill_max"] / (X["skill_mean"] + 1e-6)
    ac = [c for c in ["project_count","github_commits_90d","internship_experience_count"] if c in X]
    if ac:
        X["activity_total"] = X[ac].sum(axis=1); X["activity_max"] = X[ac].max(axis=1)
        X["is_active"] = (X["activity_total"] > 0).astype(int)
    if "internship_experience_count" in X: X["has_internship"] = (X["internship_experience_count"] > 0).astype(int)
    if "certifications_total" in X:
        X["cert_level_bin"] = pd.cut(X["certifications_total"], bins=[-1,0,1,3,float("inf")], labels=[0,1,2,3]).astype(int)
        X["is_certified"] = (X["certifications_total"] > 0).astype(int)
    acad = [c for c in ["math_scores","cs_fundamentals_scores","cognitive_ability_score"] if c in X]
    if acad: X["academic_composite"] = X[acad].mean(axis=1)
    if "cgpa" in X:
        X["cgpa_normalized"] = X["cgpa"] / 10.0; X["is_top_performer"] = (X["cgpa"] >= 8.5).astype(int)
    for cols, name in [
        (["projects_backend","projects_frontend","projects_fullstack"], "project_dev_total"),
        (["projects_data_analytics","projects_data_engineering","projects_ml_ai"], "project_data_total"),
        (["projects_security_defense","projects_security_offense"], "project_security_total"),
        (["projects_cloud","projects_devops"], "project_cloud_ops"),
        (["projects_mobile_android","projects_mobile_ios","projects_mobile_flutter"], "project_mobile_total"),
    ]:
        present = [c for c in cols if c in X]; X[name] = X[present].sum(axis=1) if present else 0
    for cols, name in [
        (["frontend_react","frontend_angular"], "frontend_stack"),
        (["backend_node","backend_django","backend_spring"], "backend_stack"),
        (["data_tool_spark","data_tool_airflow","data_tool_kafka"], "data_stack"),
        (["security_tool_siem","security_tool_wireshark","security_tool_burpsuite"], "security_stack"),
        (["mobile_kotlin","mobile_flutter"], "mobile_stack"),
        (["observability_prometheus","observability_grafana"], "observability_stack"),
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
    X     = preprocess_input(input_dict)
    parts = [aligned_proba(m, X, nc) for m in ensemble["base_models"].values()]
    prob  = ensemble["meta_model"].predict_proba(np.hstack(parts))[0]
    top3  = np.argsort(prob)[::-1][:3]
    return [{"role": le.classes_[i], "probability": round(float(prob[i])*100,1),
              "domain": role_to_domain.get(le.classes_[i],"Unknown"),
              "color": DOMAIN_COLORS.get(role_to_domain.get(le.classes_[i],""),"#6b7280"),
              "icon":  DOMAIN_ICONS.get(role_to_domain.get(le.classes_[i],""),"💼")}
             for i in top3]

# ── Skill gap ──────────────────────────────────────────────────────────────────
SGAP_FIELDS = ["programming_skill","data_analytics_skill","data_science_ml_skill","cloud_devops_skill",
               "cybersecurity_skill","ui_ux_design_skill","qa_testing_skill","web_dev_skill",
               "mobile_dev_skill","db_sql_skill","system_design_score","data_modeling_skill",
               "api_design_skill","embedded_c_cpp_skill","siem_experience_score",
               "cgpa","github_commits_90d","project_count","certifications_total","internship_experience_count"]
SGAP_LABELS = {"programming_skill":"Programming","data_analytics_skill":"Data Analytics","data_science_ml_skill":"Data Science/ML","cloud_devops_skill":"Cloud/DevOps","cybersecurity_skill":"Cybersecurity","ui_ux_design_skill":"UI/UX Design","qa_testing_skill":"QA/Testing","web_dev_skill":"Web Dev","mobile_dev_skill":"Mobile Dev","db_sql_skill":"Database/SQL","system_design_score":"System Design","data_modeling_skill":"Data Modeling","api_design_skill":"API Design","embedded_c_cpp_skill":"Embedded C/C++","siem_experience_score":"SIEM Experience","cgpa":"CGPA (norm)","github_commits_90d":"GitHub Commits (norm)","project_count":"Projects (norm)","certifications_total":"Certifications (norm)","internship_experience_count":"Internships (norm)"}
SGAP_SCALE  = {"cgpa":10.0,"github_commits_90d":30.0,"project_count":2.0,"certifications_total":2.0,"internship_experience_count":2.0}

def compute_skill_gap(user, role):
    profile = role_profiles.get(role, {})
    gaps = []
    for f in SGAP_FIELDS:
        if f not in profile: continue
        u = min(float(user.get(f,0)) * SGAP_SCALE.get(f,1.0), 10)
        i = min(float(profile[f])    * SGAP_SCALE.get(f,1.0), 10)
        gaps.append({"field":f,"label":SGAP_LABELS.get(f,f),"user":round(u,1),"ideal":round(i,1),"gap":round(i-u,2)})
    gaps.sort(key=lambda x: -x["gap"])
    tot_u = sum(g["user"]  for g in gaps)
    tot_i = sum(g["ideal"] for g in gaps)
    pct   = round(min(tot_u / max(tot_i,1) * 100, 100), 1)
    return gaps[:12], pct

def compute_peer_compare(user, role):
    pcts = role_percentiles.get(role, {})
    out  = []
    for f, label in [("cgpa","CGPA"),("programming_skill","Programming"),("data_analytics_skill","Data Analytics"),
                     ("data_science_ml_skill","DS/ML"),("cloud_devops_skill","Cloud/DevOps"),
                     ("cybersecurity_skill","Cybersecurity"),("project_count","Projects"),
                     ("github_commits_90d","GitHub Commits"),("certifications_total","Certifications")]:
        if f not in pcts: continue
        uv = float(user.get(f,0)); p = pcts[f]
        rank, rc = (("Top 25%","#10b981") if uv>=p["p75"] else
                    ("Top 50%","#3b82f6") if uv>=p["p50"] else
                    ("Top 75%","#f59e0b") if uv>=p["p25"] else ("Bottom 25%","#ef4444"))
        out.append({"field":f,"label":label,"user":uv,"p25":p["p25"],"p50":p["p50"],"p75":p["p75"],"mean":p["mean"],"rank":rank,"rank_color":rc})
    return out

# ── SHAP ───────────────────────────────────────────────────────────────────────
SHAP_LABELS = {"programming_skill":"Programming Skill","data_science_ml_skill":"Data Science/ML","data_analytics_skill":"Data Analytics","cloud_devops_skill":"Cloud & DevOps","cybersecurity_skill":"Cybersecurity","web_dev_skill":"Web Development","mobile_dev_skill":"Mobile Dev","db_sql_skill":"Database/SQL","ui_ux_design_skill":"UI/UX Design","qa_testing_skill":"QA & Testing","system_design_score":"System Design","api_design_skill":"API Design","data_modeling_skill":"Data Modeling","embedded_c_cpp_skill":"Embedded C/C++","siem_experience_score":"SIEM Experience","interest_dev_overall":"Interest: Dev","interest_data_overall":"Interest: Data","interest_cloud_infra_overall":"Interest: Cloud","interest_cybersecurity":"Interest: Security","interest_ui_ux_design":"Interest: UI/UX","cgpa":"CGPA","math_scores":"Math Score","cs_fundamentals_scores":"CS Fundamentals","cognitive_ability_score":"Cognitive Ability","project_count":"Project Count","github_commits_90d":"GitHub Commits","certifications_total":"Certifications","internship_experience_count":"Internships","dev_composite":"Dev Composite Score","data_composite":"Data Composite Score","cloud_composite":"Cloud Composite Score","security_composite":"Security Composite","prog_alignment":"Programming Alignment","data_science_alignment":"DS/ML Alignment","cloud_alignment":"Cloud Alignment","cyber_alignment":"Cyber Alignment","skill_mean":"Overall Skill Avg","academic_composite":"Academic Score","projects_ml_ai":"ML/AI Projects","projects_data_analytics":"Data Projects","projects_backend":"Backend Projects","projects_cloud":"Cloud Projects"}

def get_shap_explanation(input_dict, top_role):
    if not shap_explainer: return []
    try:
        X = preprocess_input(input_dict)
        sv = shap_explainer.shap_values(X)   # (1, n_feat, n_classes)
        ci = list(le.classes_).index(top_role)
        cs = sv[0, :, ci]
        seen, out = set(), []
        for idx in list(np.argsort(cs)[::-1][:6]) + list(np.argsort(cs)[:6]):
            if idx in seen: continue
            seen.add(idx)
            f = feature_order[idx]; v = float(cs[idx])
            out.append({"feature":f,"label":SHAP_LABELS.get(f,f.replace("_"," ").title()),"shap":round(v,4),"direction":"positive" if v>0 else "negative"})
        out.sort(key=lambda x: -abs(x["shap"]))
        return out[:10]
    except Exception: return []

# ── PDF report ─────────────────────────────────────────────────────────────────
def build_pdf_report(name, results, skill_gap, peers, roadmap, shap_exp):
    if not REPORTLAB_AVAILABLE: return None
    buf = io.BytesIO()
    c   = rl_colors
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
    SS  = getSampleStyleSheet()
    H1  = ParagraphStyle("H1", parent=SS["Title"],   fontSize=20, textColor=c.HexColor("#0d1117"), spaceAfter=4, alignment=TA_CENTER)
    H2  = ParagraphStyle("H2", parent=SS["Heading2"],fontSize=13, textColor=c.HexColor("#4f8ef7"), spaceBefore=12, spaceAfter=4)
    H3  = ParagraphStyle("H3", parent=SS["Heading3"],fontSize=11, textColor=c.HexColor("#0d1117"), spaceBefore=6, spaceAfter=3)
    NRM = ParagraphStyle("NRM",parent=SS["Normal"],  fontSize=10, textColor=c.HexColor("#0d1117"), spaceAfter=3)
    MUT = ParagraphStyle("MUT",parent=SS["Normal"],  fontSize=9,  textColor=c.HexColor("#7d8997"), spaceAfter=2)
    ACCENT=c.HexColor("#4f8ef7"); GREEN=c.HexColor("#10b981"); WHITE=c.white; LIGHT=c.HexColor("#f0f4f8")
    story = []

    def tbl(data, cols, hdr_color=ACCENT):
        t = Table(data, colWidths=cols)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),hdr_color),("TEXTCOLOR",(0,0),(-1,0),WHITE),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.4,c.HexColor("#d0d7e0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LIGHT]),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),
        ])); return t

    story += [Paragraph("CareerAI — Career Prediction Report", H1),
               HRFlowable(width="100%",thickness=1,color=ACCENT), Spacer(1,0.3*cm)]
    story.append(tbl([["Student","Generated","Model","Dataset"],
                       [name or "—", datetime.now().strftime("%d %b %Y %I:%M %p"),
                        "Stacking Ensemble (LGBM+XGB+CatBoost+LR)",
                        "30,000 records · 30 roles · 7 domains"]],
                      [3.5*cm,4.5*cm,7*cm,3*cm]))
    story += [Spacer(1,0.4*cm), Paragraph("Top Predictions", H2)]
    story.append(tbl([["#","Role","Domain","Confidence"]]+[[f"#{i+1}",r["role"],r["domain"],f"{r['probability']}%"] for i,r in enumerate(results[:3])],[1*cm,7*cm,6*cm,3*cm]))

    if shap_exp:
        story += [Spacer(1,0.3*cm), Paragraph(f"Why '{results[0]['role']}'? — SHAP Feature Explanation", H2),
                   Paragraph("SHAP values: positive = pushed toward this role, negative = pushed away.", MUT)]
        story.append(tbl([["Feature","SHAP Value","Direction"]]+[[e["label"],f"{e['shap']:+.4f}","▲ Supports" if e["direction"]=="positive" else "▼ Reduces"] for e in shap_exp[:8]],[8*cm,4*cm,5*cm],c.HexColor("#a855f7")))

    story += [Spacer(1,0.3*cm), Paragraph("Skill Gap Analysis", H2),
               Paragraph(f"Overall profile match: {skill_gap['overall_pct']}%", MUT)]
    story.append(tbl([["Skill","You","Ideal","Gap"]]+[[g["label"],f"{g['user']:.1f}/10",f"{g['ideal']:.1f}/10",f"{g['gap']:+.1f}"] for g in skill_gap["gaps"][:10]],[7*cm,3*cm,3*cm,4*cm],GREEN))

    if peers:
        story += [Spacer(1,0.3*cm), Paragraph("Peer Comparison", H2)]
        story.append(tbl([["Metric","You","P25","Median","P75","Rank"]]+[[p["label"],str(p["user"]),str(p["p25"]),str(p["p50"]),str(p["p75"]),p["rank"]] for p in peers[:8]],[4*cm,2*cm,2*cm,2*cm,2*cm,5*cm],c.HexColor("#06b6d4")))

    if roadmap and roadmap.get("skills"):
        story += [Spacer(1,0.3*cm), Paragraph("Learning Roadmap", H2),
                   Paragraph("Skills:", H3)]
        story += [Paragraph(f"• {s}", NRM) for s in roadmap.get("skills",[])[:6]]
        story += [Paragraph("Certifications:", H3)]
        story += [Paragraph(f"• {s}", NRM) for s in roadmap.get("certs",[])[:4]]
        tl = roadmap.get("timeline",{})
        if tl:
            story += [Paragraph("Timeline:", H3)]
            story.append(tbl([["3 Months",tl.get("3_months","—")],["6 Months",tl.get("6_months","—")],["12 Months",tl.get("12_months","—")]],[3*cm,14*cm],c.HexColor("#0d1117")))

    story += [Spacer(1,0.5*cm), HRFlowable(width="100%",thickness=0.5,color=c.HexColor("#7d8997")),
               Paragraph("Generated by CareerAI · B.Tech Final Year Capstone · Test Accuracy 85.2% · Top-3 Accuracy 99.3%", MUT)]
    doc.build(story); buf.seek(0); return buf

# ── Resume text parser ─────────────────────────────────────────────────────────
SKILL_KW = {"lang_python":["python"],"lang_java":["java ","java\n","java,"],"lang_javascript":["javascript","js ","node.js","nodejs","react","angular","vue"],"lang_c_cpp":["c++","c/c++","cpp"," c,"],"lang_sql":["sql","mysql","postgresql","postgres","sqlite"],"frontend_react":["react"],"frontend_angular":["angular"],"backend_node":["node.js","nodejs","express"],"backend_django":["django"],"backend_spring":["spring boot","springboot"],"db_postgres":["postgresql","postgres"],"data_tool_spark":["apache spark","pyspark","spark"],"data_tool_airflow":["airflow"],"data_tool_kafka":["kafka"],"cloud_aws":["aws","amazon web services"],"cloud_azure":["azure"],"devops_docker":["docker"],"devops_kubernetes":["kubernetes","k8s"],"devops_terraform":["terraform"],"observability_prometheus":["prometheus"],"observability_grafana":["grafana"],"security_tool_siem":["siem","splunk","qradar"],"security_tool_wireshark":["wireshark"],"security_tool_burpsuite":["burp suite","burpsuite"],"testing_tool_selenium":["selenium"],"testing_tool_jmeter":["jmeter"],"mobile_kotlin":["kotlin"],"mobile_flutter":["flutter"]}
SCORE_KW  = {"programming_skill":["programming","software development","coding"],"data_analytics_skill":["data analytics","analytics","tableau","power bi","looker"],"data_science_ml_skill":["machine learning","deep learning","tensorflow","pytorch","scikit","nlp","ml engineer"],"cloud_devops_skill":["cloud","devops","ci/cd","pipeline","infrastructure"],"cybersecurity_skill":["security","penetration testing","ethical hacking","soc","vulnerability"],"ui_ux_design_skill":["ui/ux","user experience","figma","adobe xd","wireframe"],"qa_testing_skill":["testing","quality assurance","qa","test automation"],"web_dev_skill":["web development","frontend","html","css","react","angular"],"mobile_dev_skill":["mobile","android","ios","flutter","kotlin","swift"],"db_sql_skill":["database","sql","nosql","mongodb","redis"],"system_design_score":["system design","architecture","distributed systems","microservices"],"networking_sysadmin_skill":["networking","linux","sysadmin","tcp/ip"],"embedded_c_cpp_skill":["embedded","firmware","microcontroller","rtos","arduino"],"business_analysis_skill":["business analysis","requirements","stakeholder","agile","scrum"]}

def parse_resume_text(text):
    t = text.lower()
    ex = {}

    # ── Binary tool/language flags ─────────────────────────────────────────
    for f, kws in SKILL_KW.items():
        ex[f] = 1 if any(k in t for k in kws) else 0

    # ── Skill scores — smarter multi-signal scoring ────────────────────────
    # programming_skill: python/java/JS alone should count strongly
    prog_hits = sum(1 for k in ["python","java","c++","javascript","typescript","golang","rust","programming","software development","coding","developer","engineer"] if k in t)
    ex["programming_skill"] = min(prog_hits * 1.5, 9.0)

    data_hits = sum(1 for k in ["data analytics","analytics","tableau","power bi","looker","data analysis","business intelligence","bi analyst"] if k in t)
    ex["data_analytics_skill"] = min(data_hits * 2.0, 9.0)

    ml_hits = sum(1 for k in ["machine learning","deep learning","tensorflow","pytorch","scikit","nlp","llm","neural","generative ai","mlops","ml engineer","data scientist","ai engineer"] if k in t)
    ex["data_science_ml_skill"] = min(ml_hits * 1.8, 9.5)

    cloud_hits = sum(1 for k in ["aws","azure","gcp","google cloud","cloud","devops","ci/cd","jenkins","github actions","pipeline","infrastructure as code"] if k in t)
    ex["cloud_devops_skill"] = min(cloud_hits * 1.2, 9.0)

    sec_hits = sum(1 for k in ["cybersecurity","security","penetration","ethical hacking","soc analyst","vulnerability","ctf","owasp","firewall","incident response"] if k in t)
    ex["cybersecurity_skill"] = min(sec_hits * 2.5, 9.0)

    web_hits = sum(1 for k in ["web development","frontend","html","css","react","angular","vue","next.js","nuxt","bootstrap","tailwind","web developer"] if k in t)
    ex["web_dev_skill"] = min(web_hits * 1.5, 9.0)

    mob_hits = sum(1 for k in ["mobile","android","ios","flutter","kotlin","swift","react native","xamarin"] if k in t)
    ex["mobile_dev_skill"] = min(mob_hits * 2.0, 9.0)

    db_hits = sum(1 for k in ["database","sql","mysql","postgresql","mongodb","redis","nosql","oracle","sqlite","db2"] if k in t)
    ex["db_sql_skill"] = min(db_hits * 1.5, 9.0)

    sys_hits = sum(1 for k in ["system design","architecture","distributed systems","microservices","kafka","grpc","scalab"] if k in t)
    ex["system_design_score"] = min(sys_hits * 2.0, 9.0)

    net_hits = sum(1 for k in ["networking","linux","sysadmin","tcp/ip","dns","vpn","cisco","network engineer"] if k in t)
    ex["networking_sysadmin_skill"] = min(net_hits * 2.0, 9.0)

    emb_hits = sum(1 for k in ["embedded","firmware","microcontroller","rtos","arduino","raspberry pi","stm32","vhdl","fpga"] if k in t)
    ex["embedded_c_cpp_skill"] = min(emb_hits * 2.5, 9.0)

    uiux_hits = sum(1 for k in ["ui/ux","ux design","user experience","figma","adobe xd","wireframe","prototype","usability"] if k in t)
    ex["ui_ux_design_skill"] = min(uiux_hits * 2.5, 9.0)

    qa_hits = sum(1 for k in ["testing","quality assurance","qa","test automation","selenium","pytest","unit test","integration test","jmeter"] if k in t)
    ex["qa_testing_skill"] = min(qa_hits * 2.0, 9.0)

    biz_hits = sum(1 for k in ["business analysis","requirements","stakeholder","product","agile","scrum","jira","confluence","business analyst"] if k in t)
    ex["business_analysis_skill"] = min(biz_hits * 2.0, 9.0)

    api_hits = sum(1 for k in ["api","rest","graphql","fastapi","swagger","openapi","postman"] if k in t)
    ex["api_design_skill"] = min(api_hits * 1.5, 9.0)

    dm_hits = sum(1 for k in ["data modeling","data warehouse","star schema","snowflake","dbt","dimensional modeling","etl"] if k in t)
    ex["data_modeling_skill"] = min(dm_hits * 2.5, 9.0)

    cloud_arch_hits = sum(1 for k in ["cloud architecture","well-architected","serverless","lambda","eks","ecs","solution architect"] if k in t)
    ex["cloud_arch_patterns_score"] = min(cloud_arch_hits * 3.0, 9.0)

    ds_hits = sum(1 for k in ["distributed systems","distributed","consistency","cap theorem","consensus","replication"] if k in t)
    ex["distributed_systems_knowledge_score"] = min(ds_hits * 3.0, 9.0)

    # ── Numeric extractions ────────────────────────────────────────────────
    # CGPA
    for pat in [r'cgpa[:\s]*([0-9]+\.?[0-9]*)', r'gpa[:\s]*([0-9]+\.?[0-9]*)', r'([89]\.[0-9])\s*(?:cgpa|gpa|/10)']:
        m = re.search(pat, t)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 10: ex["cgpa"] = v; break
            except: pass

    # Project count
    for pat in [r'(\d+)\+?\s*projects?', r'projects?[:\s]*(\d+)']:
        m = re.search(pat, t)
        if m:
            try: ex["project_count"] = min(int(m.group(1)), 25); break
            except: pass

    # GitHub commits
    for pat in [r'(\d+)\+?\s*commits?', r'commits?[:\s]*(\d+)']:
        m = re.search(pat, t)
        if m:
            try: ex["github_commits_90d"] = min(int(m.group(1)), 500); break
            except: pass

    # Internships
    # Extract number of internships
    _im = re.search(r'(\d+)\s+internships?', t)
    if _im:
        ex['internship_experience_count'] = min(int(_im.group(1)), 5)
    else:
        intern_count = len(re.findall(r'intern', t))
        if intern_count: ex['internship_experience_count'] = min(intern_count, 5)

    # Certifications
    cert_count = len(re.findall(r'certif|certified|comptia|cisco|oscp', t))
    if cert_count: ex["certifications_total"] = min(cert_count, 8)

    # GitHub repos
    m = re.search(r'(\d+)\s*(?:public\s*)?repos?', t)
    if m:
        try: ex["github_repos_count"] = min(int(m.group(1)), 100)
        except: pass

    # Soft skills — learning motivation and teamwork hints
    if any(k in t for k in ["passionate","enthusiastic","motivated","driven","dedicated"]): ex.setdefault("learning_motivation", 8.0)
    if any(k in t for k in ["team","collaboration","cross-functional","led","managed"]): ex.setdefault("teamwork_behavior", 7.5)
    if any(k in t for k in ["communic","present","stakeholder","client"]): ex.setdefault("communication_skill", 7.5)

    # ── Infer interest fields from detected skills ─────────────────────────
    # CRITICAL: the model heavily uses skill×interest alignment features.
    # Without interest fields the alignment is always 0 and predictions skew
    # toward UI Designer. Map each skill score → a proportional interest value.
    skill_to_interest = [
        ("programming_skill",        "interest_dev_overall"),
        ("web_dev_skill",            "interest_dev_overall"),
        ("mobile_dev_skill",         "interest_dev_overall"),
        ("data_analytics_skill",     "interest_data_overall"),
        ("data_science_ml_skill",    "interest_data_overall"),
        ("data_modeling_skill",      "interest_data_overall"),
        ("cloud_devops_skill",       "interest_cloud_infra_overall"),
        ("cybersecurity_skill",      "interest_cybersecurity"),
        ("ui_ux_design_skill",       "interest_ui_ux_design"),
        ("business_analysis_skill",  "interest_business_and_management"),
    ]
    for skill_field, interest_field in skill_to_interest:
        skill_val = ex.get(skill_field, 0)
        if skill_val > 0:
            # Interest is set to ~85-90% of skill score (present but slightly
            # conservative so the model doesn't over-boost niche roles)
            inferred = round(min(skill_val * 0.9, 9.0), 1)
            # Only set if not already present or if this score is higher
            ex[interest_field] = max(ex.get(interest_field, 0), inferred)

    return ex

# ── GitHub analyser ────────────────────────────────────────────────────────────
GH_TOOL_KW = {"python":"lang_python","java":"lang_java","javascript":"lang_javascript","typescript":"lang_javascript","c++":"lang_c_cpp","c":"lang_c_cpp","kotlin":"mobile_kotlin","dart":"mobile_flutter","sql":"lang_sql","react":"frontend_react","angular":"frontend_angular","nodejs":"backend_node","node":"backend_node","django":"backend_django","spring":"backend_spring","postgresql":"db_postgres","postgres":"db_postgres","spark":"data_tool_spark","airflow":"data_tool_airflow","kafka":"data_tool_kafka","aws":"cloud_aws","azure":"cloud_azure","docker":"devops_docker","kubernetes":"devops_kubernetes","terraform":"devops_terraform","prometheus":"observability_prometheus","grafana":"observability_grafana","selenium":"testing_tool_selenium","jmeter":"testing_tool_jmeter","wireshark":"security_tool_wireshark","burpsuite":"security_tool_burpsuite","kotlin":"mobile_kotlin","flutter":"mobile_flutter"}
GH_SCORE_KW= {"machine-learning":"data_science_ml_skill","deep-learning":"data_science_ml_skill","tensorflow":"data_science_ml_skill","pytorch":"data_science_ml_skill","data-science":"data_analytics_skill","data-analysis":"data_analytics_skill","web-development":"web_dev_skill","api":"api_design_skill","microservices":"system_design_score","cloud":"cloud_devops_skill","devops":"cloud_devops_skill","cybersecurity":"cybersecurity_skill","security":"cybersecurity_skill","mobile":"mobile_dev_skill","android":"mobile_dev_skill","embedded":"embedded_c_cpp_skill","firmware":"embedded_c_cpp_skill","database":"db_sql_skill","ui":"ui_ux_design_skill","testing":"qa_testing_skill","automation":"qa_testing_skill"}

def fetch_github_profile(username):
    username = username.strip().lstrip("@").split("/")[-1]
    if not username: return None, "Please enter a valid GitHub username."
    headers  = {"Accept":"application/vnd.github.v3+json","User-Agent":"CareerAI-v4"}
    tok = os.environ.get("GITHUB_TOKEN","")
    if tok: headers["Authorization"] = f"token {tok}"
    base = "https://api.github.com"
    try:
        ur = req_lib.get(f"{base}/users/{username}", headers=headers, timeout=10)
    except Exception as conn_err:
        err_str = str(conn_err).lower()
        if "proxy" in err_str or "tunnel" in err_str or "connection" in err_str or "timeout" in err_str:
            return None, "Cannot reach GitHub API. Make sure you have internet access and try again."
        return None, f"Connection error: {str(conn_err)[:100]}"
    if ur.status_code == 404: return None, f"GitHub user '{username}' not found. Check the spelling."
    if ur.status_code == 403: return None, "GitHub API rate limit reached. Wait 1 minute and try again."
    if ur.status_code == 401: return None, "GitHub authentication failed. Check GITHUB_TOKEN if set."
    if ur.status_code != 200: return None, f"GitHub API returned error {ur.status_code}. Try again."
    user = ur.json()
    try:
        rr = req_lib.get(f"{base}/users/{username}/repos", headers=headers,
                         params={"per_page":100,"sort":"pushed","type":"owner"}, timeout=10)
        repos = rr.json() if rr.status_code == 200 else []
    except Exception:
        repos = []
    ex = {}
    ex["github_repos_count"] = int(user.get("public_repos",0))
    ex["project_count"]      = min(int(user.get("public_repos",0)), 20)
    lang_ctr, flags, score_hints = {}, set(), {}
    pt = {"projects_ml_ai":0,"projects_data_analytics":0,"projects_backend":0,"projects_frontend":0,"projects_fullstack":0,"projects_mobile_android":0,"projects_security_defense":0,"projects_cloud":0,"projects_devops":0,"projects_embedded":0}
    for repo in repos:
        if repo.get("fork"): continue
        lang = (repo.get("language") or "").lower()
        if lang: lang_ctr[lang] = lang_ctr.get(lang,0)+1
        topics = repo.get("topics",[])
        text   = " ".join(topics)+" "+(repo.get("description") or "").lower()+" "+repo.get("name","").lower()
        for kw,f in GH_TOOL_KW.items():
            if kw in text or kw in lang: flags.add(f)
        for kw,f in GH_SCORE_KW.items():
            if kw in text: score_hints[f] = score_hints.get(f,0)+1
        if any(w in text for w in ["machine-learning","deep-learning","ml","ai","nlp","pytorch","tensorflow"]): pt["projects_ml_ai"]+=1
        elif any(w in text for w in ["data","analytics","dashboard","pandas","etl","airflow"]): pt["projects_data_analytics"]+=1
        elif any(w in text for w in ["fullstack","full-stack","mern","django","react","next"]): pt["projects_fullstack"]+=1
        elif any(w in text for w in ["frontend","css","html","vue","angular","tailwind"]): pt["projects_frontend"]+=1
        elif any(w in text for w in ["backend","api","rest","graphql","server","microservice","node","spring","flask"]): pt["projects_backend"]+=1
        elif any(w in text for w in ["android","mobile","ios","flutter","kotlin","swift"]): pt["projects_mobile_android"]+=1
        elif any(w in text for w in ["security","ctf","pentest","exploit","hack"]): pt["projects_security_defense"]+=1
        elif any(w in text for w in ["cloud","aws","azure","gcp","terraform","kubernetes"]): pt["projects_devops"]+=1
        elif any(w in text for w in ["embedded","firmware","arduino","stm32","rtos"]): pt["projects_embedded"]+=1
    for f in flags: ex[f] = 1
    LANG_MAP = {"python":"lang_python","java":"lang_java","javascript":"lang_javascript","typescript":"lang_javascript","c++":"lang_c_cpp","c":"lang_c_cpp","kotlin":"mobile_kotlin","dart":"mobile_flutter"}
    for lang,f in LANG_MAP.items():
        if lang in lang_ctr: ex[f] = 1
    top_langs = sorted(lang_ctr.items(), key=lambda x: -x[1])
    for lang,_ in top_langs[:3]:
        if lang in ["python"]:      ex.setdefault("programming_skill",7.0); ex.setdefault("data_analytics_skill",5.0)
        elif lang in ["javascript","typescript"]: ex.setdefault("programming_skill",6.5); ex.setdefault("web_dev_skill",7.0)
        elif lang in ["java","kotlin"]:           ex.setdefault("programming_skill",6.5)
        elif lang in ["c","c++"]:                 ex.setdefault("programming_skill",6.0); ex.setdefault("embedded_c_cpp_skill",5.0)
    for f,cnt in score_hints.items(): ex[f] = max(ex.get(f,0), min(2.5+cnt*1.2, 8.5))
    for k,v in pt.items():
        if v: ex[k] = v
    total_stars = sum(r.get("stargazers_count",0) for r in repos if not r.get("fork"))
    ex["github_commits_90d"] = min(150+total_stars*2, 400) if total_stars>50 else max(50,len(repos)*8)
    bio = (user.get("bio") or "").lower()
    # Fix: use `bio` (the GitHub bio string), not `t` which is undefined here
    cert_count = len(re.findall(r'certif|certified|comptia|cisco|oscp', bio))
    if cert_count: ex["certifications_total"] = min(cert_count*2, 6)
    summary = {"username":user.get("login"),"name":user.get("name") or user.get("login"),"avatar_url":user.get("avatar_url"),"bio":user.get("bio") or "","public_repos":user.get("public_repos",0),"followers":user.get("followers",0),"following":user.get("following",0),"location":user.get("location") or "","total_stars":total_stars,"non_fork_repos":sum(1 for r in repos if not r.get("fork")),"top_languages":[l for l,_ in top_langs[:5]],"project_types":{k:v for k,v in pt.items() if v},"detected_tools":sorted(flags)}
    return {"extracted":ex,"summary":summary}, None


# ── Career Readiness Score ────────────────────────────────────────────────────
def compute_readiness_score(user, top_role, skill_gap_pct, peers, confidence):
    """
    0-100 score across 4 dimensions:
      40% skill match (from skill gap)
      25% peer rank  (from peer percentiles)
      20% academic   (CGPA, math, CS fundamentals)
      15% experience (projects, internships, certs, github)
    """
    # 1. Skill match (40 pts)
    skill_pts = round(skill_gap_pct * 0.40, 1)

    # 2. Peer rank (25 pts) — average rank score across metrics
    rank_map = {"Top 25%": 100, "Top 50%": 75, "Top 75%": 50, "Bottom 25%": 25}
    rank_scores = [rank_map.get(p.get("rank","Bottom 25%"), 25) for p in peers]
    peer_avg = sum(rank_scores) / max(len(rank_scores), 1)
    peer_pts = round(peer_avg * 0.25, 1)

    # 3. Academic (20 pts)
    cgpa_norm   = min(float(user.get("cgpa", 0)) / 10.0, 1.0)
    math_norm   = min(float(user.get("math_scores", 0)) / 100.0, 1.0)
    cs_norm     = min(float(user.get("cs_fundamentals_scores", 0)) / 100.0, 1.0)
    acad_avg    = (cgpa_norm * 0.5 + math_norm * 0.25 + cs_norm * 0.25)
    acad_pts    = round(acad_avg * 100 * 0.20, 1)

    # 4. Experience (15 pts)
    proj_score  = min(float(user.get("project_count", 0)) / 12.0, 1.0)
    intern_score= min(float(user.get("internship_experience_count", 0)) / 3.0, 1.0)
    cert_score  = min(float(user.get("certifications_total", 0)) / 5.0, 1.0)
    gh_score    = min(float(user.get("github_commits_90d", 0)) / 250.0, 1.0)
    exp_avg     = (proj_score * 0.35 + intern_score * 0.30 + cert_score * 0.20 + gh_score * 0.15)
    exp_pts     = round(exp_avg * 100 * 0.15, 1)

    total = round(skill_pts + peer_pts + acad_pts + exp_pts, 1)
    total = min(max(total, 0), 100)

    # Band classification
    if total >= 85:
        band, band_color = "Standout",   "#10b981"
    elif total >= 70:
        band, band_color = "Job-Ready",  "#3b82f6"
    elif total >= 50:
        band, band_color = "Developing", "#f59e0b"
    else:
        band, band_color = "Not Ready",  "#ef4444"

    return {
        "score":      total,
        "band":       band,
        "band_color": band_color,
        "breakdown": {
            "skill_match": {"pts": skill_pts, "max": 40, "label": "Skill match"},
            "peer_rank":   {"pts": peer_pts,  "max": 25, "label": "Peer ranking"},
            "academic":    {"pts": acad_pts,  "max": 20, "label": "Academic"},
            "experience":  {"pts": exp_pts,   "max": 15, "label": "Experience"},
        }
    }


# ── Plain-English SHAP Explanation ───────────────────────────────────────────
PLAIN_ENGLISH_TEMPLATES = {
    # Positive drivers
    "data_science_alignment":   "Your strong Data Science/ML skill combined with high interest in AI — a powerful combination for this role.",
    "cloud_alignment":          "Your cloud skills align well with your interest in infrastructure, a key signal for cloud roles.",
    "cyber_alignment":          "Your cybersecurity skill matches your interest in security — exactly what this role needs.",
    "prog_alignment":           "Your programming ability combined with development interest strongly fits this role.",
    "uiux_alignment":           "Your UI/UX design skill and interest in design make you a natural fit here.",
    "data_analytics_alignment": "Your data analytics skill paired with your data interest is a strong match.",
    "biz_alignment":            "Your business analysis skill combined with management interest signals a good fit.",
    "data_science_ml_skill":    "Your Data Science / ML skill level is above average for this role.",
    "programming_skill":        "Your programming skill is a key driver — this role needs strong coders.",
    "cloud_devops_skill":       "Your Cloud & DevOps skills are a major positive signal here.",
    "cybersecurity_skill":      "Your cybersecurity skill is one of the top reasons for this prediction.",
    "web_dev_skill":            "Your web development skill is contributing to this prediction.",
    "system_design_score":      "Your system design knowledge is valued highly in this role.",
    "api_design_skill":         "Your API design skill is a positive signal for this engineering role.",
    "data_modeling_skill":      "Your data modeling skill is relevant and adding to the prediction.",
    "projects_ml_ai":           "Your ML/AI projects demonstrate hands-on experience the model rewarded.",
    "projects_backend":         "Your backend projects show practical engineering experience.",
    "projects_data_analytics":  "Your data analytics projects are a strong practical signal.",
    "projects_cloud":           "Your cloud projects demonstrate real infrastructure experience.",
    "dominant_domain_score":    "You have a clear dominant domain strength that matches this role.",
    "skill_focus_ratio":        "You are a specialist rather than a generalist — this role rewards depth.",
    "embedded_composite":       "Your embedded systems skills are the primary driver for this role.",
    "security_composite":       "Your combined security skills form a strong match.",
    "data_composite":           "Your combined data skills are the main reason for this prediction.",
    "dev_composite":            "Your combined development skills form a strong match.",
    "cloud_composite":          "Your combined cloud skills are highly relevant here.",
    "cgpa":                     "Your academic performance (CGPA) is above the average for this role.",
    "internship_experience_count": "Your internship experience is a positive differentiator.",
    "certifications_total":     "Your certifications show commitment to professional development.",
    "github_commits_90d":       "Your recent GitHub activity shows active coding — the model rewarded this.",
    "project_count":            "Your number of projects shows practical hands-on experience.",
    "activity_total":           "Your overall activity level (projects + commits + internships) is strong.",
    "soft_skills_mean":         "Your soft skills (teamwork, communication, motivation) are above average.",
    "communication_skill":      "Your communication skill is a positive factor for this role.",
    "learning_motivation":      "Your high learning motivation is a positive behavioral signal.",
    # Negative drivers (push away)
    "networking_sysadmin_skill":   "Your networking/sysadmin skill is lower than typical for this role.",
    "qa_testing_skill":            "Your QA/testing skill is below average for this prediction — less relevant here.",
    "ui_ux_design_skill":          "Your UI/UX design skill is lower, which reduces fit for design-heavy roles.",
    "mobile_dev_skill":            "Your mobile development skill is lower — this role doesn't prioritise it.",
    "embedded_c_cpp_skill":        "Your embedded C/C++ skill is low — this role doesn't require it.",
    "business_analysis_skill":     "Your business analysis skill is lower, which is less critical here.",
}

def build_plain_english_explanation(shap_factors, top_role, user_input):
    """Convert SHAP factors into 3-4 natural language sentences."""
    if not shap_factors:
        skill_map = [
            ("data_science_ml_skill","data_science_ml_skill"),
            ("cloud_devops_skill","cloud_devops_skill"),
            ("cybersecurity_skill","cybersecurity_skill"),
            ("programming_skill","programming_skill"),
            ("ui_ux_design_skill","ui_ux_design_skill"),
        ]
        synthetic = []
        for raw_field, skill_field in skill_map:
            val = float(user_input.get(raw_field, 0))
            if val >= 6:
                synthetic.append({"feature":skill_field,"direction":"positive","shap":val,
                                   "label":skill_field.replace("_skill","").replace("_"," ").title()})
            elif val < 3:
                synthetic.append({"feature":skill_field,"direction":"negative","shap":-val,
                                   "label":skill_field.replace("_skill","").replace("_"," ").title()})
        if not synthetic:
            return {"summary":"","sentences":[],"top_positive":[],"top_negative":[]}
        shap_factors = sorted(synthetic, key=lambda x: -abs(x["shap"]))[:5]

    positives = [f for f in shap_factors if f["direction"] == "positive"][:3]
    negatives = [f for f in shap_factors if f["direction"] == "negative"][:2]

    sentences = []

    # Opening sentence about the role
    conf_word = "strongly" if len(positives) >= 3 else "moderately"
    sentences.append(
        f"The model {conf_word} predicts <b>{top_role}</b> as your best-fit career based on your profile."
    )

    # Main positive drivers
    if positives:
        pos_labels = []
        for f in positives[:2]:
            tmpl = PLAIN_ENGLISH_TEMPLATES.get(f["feature"])
            if tmpl:
                pos_labels.append(tmpl)
            else:
                friendly = f["label"].replace(" (norm)","").replace(" Alignment","").replace(" ×"," and")
                pos_labels.append(f"Your {friendly} is a strong positive signal for this role.")
        sentences.extend(pos_labels)

    # Gap / improvement sentence from top negative driver
    if negatives:
        neg = negatives[0]
        tmpl = PLAIN_ENGLISH_TEMPLATES.get(neg["feature"])
        if tmpl:
            sentences.append(tmpl)

    # Closing actionable sentence
    top_gap_field = None
    if shap_factors:
        neg_factors = [f for f in shap_factors if f["direction"] == "negative"]
        if neg_factors:
            top_gap_field = neg_factors[0].get("label","")
    if top_gap_field:
        sentences.append(
            f"To increase your confidence further, focus on improving <b>{top_gap_field}</b>."
        )

    return {
        "summary": " ".join(sentences[:1]),
        "sentences": sentences,
        "top_positive": [PLAIN_ENGLISH_TEMPLATES.get(f["feature"], f["label"]) for f in positives],
        "top_negative": [PLAIN_ENGLISH_TEMPLATES.get(f["feature"], f["label"]) for f in negatives],
    }


# ── Action Plan ───────────────────────────────────────────────────────────────
def compute_action_plan(skill_gaps, user_input):
    """Pick top 3 actionable steps from the biggest skill gaps."""
    plan = []
    seen = set()
    # Sort gaps descending by gap size
    sorted_gaps = sorted(skill_gaps, key=lambda g: g["gap"], reverse=True)
    for gap in sorted_gaps:
        field = gap["field"]
        if field in action_resources and field not in seen:
            r = action_resources[field]
            plan.append({
                "skill":    gap["label"],
                "gap":      gap["gap"],
                "title":    r["title"],
                "resource": r["resource"],
                "url":      r["url"],
                "time":     r["time"],
                "tag":      r["tag"],
            })
            seen.add(field)
        if len(plan) == 3:
            break
    return plan

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_meta=model_meta, test_metrics=test_metrics)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", model_meta=model_meta, test_metrics=test_metrics,
                           analytics=analytics_data, feature_imp=feature_imp)

@app.route("/admin_disabled")
def admin():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT ts,name,top_role,domain,confidence,top3 FROM predictions ORDER BY ts DESC LIMIT 50").fetchall()
    total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    domain_dist = con.execute("SELECT domain,COUNT(*) FROM predictions GROUP BY domain ORDER BY 2 DESC").fetchall()
    top_roles   = con.execute("SELECT top_role,COUNT(*) FROM predictions GROUP BY top_role ORDER BY 2 DESC LIMIT 10").fetchall()
    con.close()
    logs = [{"ts":r[0][:16].replace("T"," "),"name":r[1],"top_role":r[2],"domain":r[3],"confidence":r[4],"top3":json.loads(r[5]) if r[5] else []} for r in rows]
    return render_template("admin.html", logs=logs, total=total,
                           domain_dist=[{"domain":d[0],"count":d[1]} for d in domain_dist],
                           top_roles=[{"role":r[0],"count":r[1]} for r in top_roles])

@app.route("/profiles_disabled")
def profiles_page():
    return render_template("profiles.html")

@app.route("/predict_v2", methods=["POST"])
def predict_v2():
    try:
        data     = request.get_json()
        name     = data.pop("student_name","Anonymous")
        results  = predict_single(data)
        top_role = results[0]["role"]
        gaps, pct = compute_skill_gap(data, top_role)
        peers    = compute_peer_compare(data, top_role)
        roadmap  = roadmaps.get(top_role, {})
        shap_exp   = get_shap_explanation(data, top_role)
        readiness  = compute_readiness_score(data, top_role, pct, peers, results[0]["probability"])
        plain_exp  = build_plain_english_explanation(shap_exp, top_role, data)
        log_prediction(name, results, data)
        salary    = salary_data.get(top_role, {})
        radar     = radar_profiles.get(top_role, {})
        companies = companies_data.get(top_role, [])
        action    = compute_action_plan(gaps, data)
        return jsonify({"success":True,"predictions":results,"skill_gap":{"gaps":gaps,"overall_pct":pct},"peer_compare":peers,"roadmap":roadmap,"shap":shap_exp,"salary":salary,"readiness":readiness,"plain_explanation":plain_exp,"radar":radar,"companies":companies,"action_plan":action,"top_role":top_role})
    except Exception as e:
        import traceback
        return jsonify({"success":False,"error":str(e),"trace":traceback.format_exc()}), 500

@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    try:
        if not REPORTLAB_AVAILABLE:
            return jsonify({"success":False,"error":"reportlab not installed"}), 500
        d  = request.get_json()
        buf = build_pdf_report(d.get("student_name","Student"),d.get("predictions",[]),d.get("skill_gap",{}),d.get("peer_compare",[]),d.get("roadmap",{}),d.get("shap",[]))
        if not buf: return jsonify({"success":False,"error":"PDF generation failed"}), 500
        name = (d.get("student_name","Report") or "Report").replace(" ","_")
        return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=f"CareerAI_{name}.pdf")
    except Exception as e:
        import traceback
        return jsonify({"success":False,"error":str(e),"trace":traceback.format_exc()}), 500

@app.route("/parse_resume", methods=["POST"])
def parse_resume():
    try:
        data = request.get_json()
        text = data.get("text","")
        if not text.strip(): return jsonify({"success":False,"error":"Empty text"})
        return jsonify({"success":True,"extracted":parse_resume_text(text)})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/parse_resume_pdf", methods=["POST"])
def parse_resume_pdf():
    try:
        if not PDF_AVAILABLE: return jsonify({"success":False,"error":"pdfplumber not installed. Run: pip install pdfplumber"}), 500
        if "file" not in request.files: return jsonify({"success":False,"error":"No file uploaded"}), 400
        f = request.files["file"]
        if not f.filename.lower().endswith(".pdf"): return jsonify({"success":False,"error":"Only PDF files supported"}), 400
        raw = f.read()
        if len(raw) > 5*1024*1024: return jsonify({"success":False,"error":"File too large (max 5MB)"}), 400
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text_parts.append(t)
        raw_text = "\n".join(text_parts)
        if not raw_text.strip(): return jsonify({"success":False,"error":"No text found in PDF. Use a text-based PDF (not scanned image)."}), 400
        return jsonify({"success":True,"extracted":parse_resume_text(raw_text),"raw_text":raw_text[:1500],"page_count":len(text_parts),"char_count":len(raw_text)})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/github_profile", methods=["POST"])
def github_profile():
    try:
        if not REQUESTS_AVAILABLE: return jsonify({"success":False,"error":"requests not installed"}), 500
        username = (request.get_json() or {}).get("username","").strip()
        if not username: return jsonify({"success":False,"error":"Username required"}), 400
        result, err = fetch_github_profile(username)
        if err: return jsonify({"success":False,"error":err}), 400
        return jsonify({"success":True,**result})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/save_profile", methods=["POST"])
def save_profile_route():
    try:
        d = request.get_json()
        name = (d.get("name","") or "").strip()
        if not name: return jsonify({"success":False,"error":"Name required"})
        db_save_profile(name, d.get("profile",{}), d.get("result",{}))
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/get_profiles")
def get_profiles():
    try:
        con  = sqlite3.connect(DB_PATH)
        rows = con.execute("SELECT id,ts,name,data,result FROM profiles ORDER BY ts DESC LIMIT 20").fetchall()
        con.close()
        out = []
        for r in rows:
            res = json.loads(r[4]) if r[4] else {}
            out.append({"id":r[0],"ts":r[1][:16].replace("T"," "),"name":r[2],"top_role":res.get("top_role","—"),"confidence":res.get("confidence",0),"profile":json.loads(r[3]),"result":res})
        return jsonify({"success":True,"profiles":out})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/api/analytics")
def get_analytics():
    return jsonify({"analytics":analytics_data,"feature_importance":feature_imp})


# ── Compare Roles page ───────────────────────────────────────────────────────
@app.route("/compare")
def compare_page():
    roles = sorted(le.classes_.tolist())
    return render_template("compare.html",
                           roles=roles,
                           profiles=role_profiles,
                           salaries=salary_data)


# ── Batch Prediction ──────────────────────────────────────────────────────────
@app.route("/batch_disabled")
def batch_page():
    return render_template("batch.html", model_meta=model_meta, test_metrics=test_metrics)

@app.route("/batch_predict_disabled", methods=["POST"])
def batch_predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        f = request.files["file"]
        if not f.filename.lower().endswith(".csv"):
            return jsonify({"success": False, "error": "Only CSV files accepted"}), 400
        df = pd.read_csv(f)
        if len(df) > 500:
            return jsonify({"success": False, "error": "Max 500 rows per batch"}), 400
        CBF = ['lang_python','lang_java','lang_javascript','lang_c_cpp','lang_sql',
               'frontend_react','frontend_angular','backend_node','backend_django','backend_spring',
               'db_postgres','data_tool_spark','data_tool_airflow','data_tool_kafka',
               'cloud_aws','cloud_azure','devops_docker','devops_kubernetes','devops_terraform',
               'observability_prometheus','observability_grafana','security_tool_siem',
               'security_tool_wireshark','security_tool_burpsuite','testing_tool_selenium',
               'testing_tool_jmeter','mobile_kotlin','mobile_flutter']
        results_out = []
        domain_counts = {}
        role_counts   = {}
        band_counts   = {"Standout":0,"Job-Ready":0,"Developing":0,"Not Ready":0}
        errors = 0
        for idx, row in df.iterrows():
            try:
                d = {}
                for col in df.columns:
                    if col in CBF:
                        d[col] = 1 if str(row[col]).strip() in ['1','True','true','yes','Yes'] else 0
                    else:
                        try: d[col] = float(row[col])
                        except: d[col] = 0
                preds = predict_single(d)
                top   = preds[0]
                gaps, pct = compute_skill_gap(d, top["role"])
                peers     = compute_peer_compare(d, top["role"])
                rs        = compute_readiness_score(d, top["role"], pct, peers, top["probability"])
                name = str(row.get("student_name", row.get("name", f"Student {idx+1}"))).strip()
                results_out.append({
                    "name":           name,
                    "top_role":       top["role"],
                    "domain":         top["domain"],
                    "confidence":     top["probability"],
                    "alt1":           preds[1]["role"] if len(preds)>1 else "",
                    "alt2":           preds[2]["role"] if len(preds)>2 else "",
                    "readiness_score":rs["score"],
                    "readiness_band": rs["band"],
                    "skill_match_pct":pct,
                })
                domain_counts[top["domain"]] = domain_counts.get(top["domain"],0)+1
                role_counts[top["role"]]     = role_counts.get(top["role"],0)+1
                band_counts[rs["band"]]      = band_counts.get(rs["band"],0)+1
            except Exception:
                errors += 1
        total = len(results_out)
        summary = {
            "total":          total,
            "errors":         errors,
            "domain_counts":  dict(sorted(domain_counts.items(),key=lambda x:-x[1])),
            "top_roles":      dict(sorted(role_counts.items(),key=lambda x:-x[1])[:8]),
            "band_counts":    band_counts,
            "avg_confidence": round(sum(r["confidence"] for r in results_out)/max(total,1),1),
            "avg_readiness":  round(sum(r["readiness_score"] for r in results_out)/max(total,1),1),
        }
        return jsonify({"success":True,"results":results_out,"summary":summary})
    except Exception as e:
        import traceback
        return jsonify({"success":False,"error":str(e),"trace":traceback.format_exc()}), 500

if __name__ == "__main__":
    print("CareerAI starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)


# ── SHAP Explanation route ───────────────────────────────────────────────────
FRIENDLY_LABELS = {
    "data_science_ml_skill":"Data Science / ML skill","programming_skill":"Programming skill",
    "data_analytics_skill":"Data Analytics skill","cloud_devops_skill":"Cloud & DevOps skill",
    "cybersecurity_skill":"Cybersecurity skill","ui_ux_design_skill":"UI/UX Design skill",
    "web_dev_skill":"Web Dev skill","mobile_dev_skill":"Mobile Dev skill",
    "db_sql_skill":"Database / SQL skill","system_design_score":"System Design score",
    "cloud_arch_patterns_score":"Cloud Architecture score","api_design_skill":"API Design skill",
    "data_modeling_skill":"Data Modeling skill","embedded_c_cpp_skill":"Embedded C/C++ skill",
    "interest_data_overall":"Interest in Data & AI","interest_dev_overall":"Interest in Dev",
    "interest_cloud_infra_overall":"Interest in Cloud","interest_cybersecurity":"Interest in Cybersecurity",
    "interest_ui_ux_design":"Interest in UI/UX","cgpa":"CGPA","math_scores":"Math score",
    "cs_fundamentals_scores":"CS Fundamentals","cognitive_ability_score":"Cognitive ability",
    "github_commits_90d":"GitHub commits (90d)","project_count":"Project count",
    "internship_experience_count":"Internships","certifications_total":"Certifications",
    "communication_skill":"Communication skill","teamwork_behavior":"Teamwork",
    "learning_motivation":"Learning motivation","siem_experience_score":"SIEM experience",
    "lang_python":"Python","lang_javascript":"JavaScript","lang_java":"Java",
    "devops_docker":"Docker","devops_kubernetes":"Kubernetes","cloud_aws":"AWS",
    "data_tool_spark":"Apache Spark","data_tool_airflow":"Airflow",
    "prog_alignment":"Prog skill × interest","data_analytics_alignment":"Analytics × interest",
    "data_science_alignment":"ML skill × interest","cloud_alignment":"Cloud skill × interest",
    "cyber_alignment":"Cyber skill × interest","skill_mean":"Avg skill level",
    "skill_focus_ratio":"Skill focus (specialist)","dominant_domain_score":"Dominant domain score",
    "dev_composite":"Dev composite","data_composite":"Data composite",
    "cloud_composite":"Cloud composite","security_composite":"Security composite",
}

@app.route("/explain", methods=["POST"])
def explain():
    """Return SHAP-based top-10 factors driving the prediction."""
    try:
        data     = request.get_json()
        top_role = data.get("top_role", "")
        input_d  = {k: v for k, v in data.items() if k != "top_role"}

        df_in  = pd.DataFrame([input_d])
        df_fe  = add_features(df_in)
        df_sel = df_fe.drop(columns=drop_corr, errors="ignore").reindex(columns=feature_order, fill_value=0)
        X_proc = preprocess.transform(df_sel)

        if SHAP_AVAILABLE and shap_explainer is not None:
            shap_vals  = shap_explainer.shap_values(X_proc)     # (n_samples, n_features, n_classes)
            class_idx  = list(le.classes_).index(top_role) if top_role in list(le.classes_) else 0
            row_shap   = shap_vals[0, :, class_idx]            # shape: (n_features,)
            feat_names = feature_order

            # Top 10 by absolute SHAP value
            pairs = sorted(zip(feat_names, row_shap.tolist()), key=lambda x: -abs(x[1]))[:10]
            factors = [{
                "feature": f,
                "label":   FRIENDLY_LABELS.get(f, f.replace("_"," ").title()),
                "shap":    round(v, 4),
                "direction": "positive" if v >= 0 else "negative",
                "user_val": round(float(df_fe[f].iloc[0]) if f in df_fe.columns else 0, 2),
            } for f, v in pairs]
        else:
            # Fallback: use raw feature importance ranked by user's values
            fi = json.load(open(OUT_DIR / "feature_importance.json"))
            factors = [{"feature": f, "label": FRIENDLY_LABELS.get(f, f.replace("_"," ").title()),
                        "shap": round(fi.get(f,0)/10000, 4), "direction": "positive",
                        "user_val": round(float(df_fe[f].iloc[0]) if f in df_fe.columns else 0, 2)}
                       for f in list(fi.keys())[:10]]

        return jsonify({"success": True, "factors": factors, "top_role": top_role})
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500


# ── What-if Simulator route ──────────────────────────────────────────────────
@app.route("/whatif", methods=["POST"])
def whatif():
    """Simulate prediction when one field is changed."""
    try:
        data      = request.get_json()
        field     = data.get("field")
        new_value = float(data.get("value", 0))
        base_data = {k: v for k, v in data.items() if k not in ("field", "value")}
        base_data[field] = new_value
        results = predict_single(base_data)
        return jsonify({
            "success":     True,
            "predictions": results[:3],
            "changed_field": field,
            "new_value":  new_value,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Salary data route ────────────────────────────────────────────────────────
@app.route("/api/salary", methods=["POST", "GET"])
def get_salary():
    if request.method == 'POST':
        role = (request.get_json() or {}).get('role', '')
    else:
        role = request.args.get('role', '')
    data = salary_data.get(role)
    if not data:
        for k in salary_data:
            if role.lower() in k.lower() or k.lower() in role.lower():
                data = salary_data[k]; break
    if data:
        return jsonify({"success": True, "salary": data, "role": role})
    return jsonify({"success": False, "error": "Role not found"}), 404


# ── Predict route already updated — inject salary into response ──────────────
