"""Build the IAA hackathon notebook from parts.

    python3 build_notebook.py [--execute] [--data DIR]

Reads ai_log.json and prompt_library.md from this folder, writes iaa_notebook.ipynb.
With --execute it runs the notebook top to bottom (nbclient), then fills the
{{tokens}} in the summary from results_summary.json written by the last cell.
"""
import json, sys, os, re
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
LOG = json.loads((HERE / "ai_log.json").read_text())
PROMPTS = (HERE / "prompt_library.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s, hidden=False):
    c = nbf.v4.new_code_cell(s.strip("\n"))
    if hidden: c.metadata.update({"_kg_hide-input": True, "jupyter": {"source_hidden": True}})
    cells.append(c)

# =====================================================================================
md(r"""
# Pricing a house the way an actuary prices a risk
### IAA AI Assisted Hackathon · model documented in the structure of a Canadian rate filing

**How to read this notebook.** It follows the IAA template (sections 1 to 5). Inside it, the model is documented the way a Canadian auto rate filing justifies a rating model: what the model is for, what data went in, why each choice was made, how it was validated, what changes against the current model, and which decisions an actuary must sign. Grey boxes state what a reviewer would ask; green boxes give a plain reading of the technical terms; exchange cards show where AI changed a decision and how each claim was checked.
""")

md(r"""
## 1. Summary

| | |
|---|---|
| **Team / participant** | {{team}} |
| **Public leaderboard (RMSLE)** | {{public_lb}} |
| **One-line highlight** | An appointed actuary working with an AI team (data steward, modeller, independent reviewer, filing writer, explainer): every AI claim is tested, every judgment is signed by the human. |

**Proposal.** Replace the current model (a standard penalised GLM, the kind of rating model already in force) with a blend of {{n_models}} models from three families, documented with a glass-box rating plan.

| Measure | Current model | Proposed blend |
|---|---|---|
| Cross-validated error (RMSLE, mean of {{n_seeds}} split(s)) | {{bench_cv}} | {{blend_cv}} |
| Locked hold-out error ({{n_hold}} houses never touched) | {{bench_ho}} | {{blend_ho}} |
| Test houses whose price moves by 5% or less | | {{within5}} |

<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px;margin:8px 0">
<b>How to read this.</b> RMSLE is the typical size of a miss in log terms, read as a percentage: {{blend_cv}} means predictions are typically about {{blend_cv_pct}} off, roughly ±{{blend_cv_dollars}} on a $300k house. Lower is better.
</div>
""")

# =====================================================================================
md(r"""
## 2. Generative AI integration

We did not use AI as one prompt and one answer. We used it the way a small actuarial department uses its team: each AI role has a defined job, every AI claim is tested before it is used, and every decision a number cannot settle is recorded for the appointed actuary to sign. The diagram shows the loop; the ledger lists every decision it produced; the cards below show the exchanges that changed the analysis, with the check that settled each one. The full prompt transcript and the role set-up prompts are in section 5.
""")

code(r"""
# ---------- configuration: the only cell to change for a new data set ----------
COMPETITION = "ai-assisted-hackathon-1"     # Kaggle slug; data is read from /kaggle/input/<slug>/
TARGET, ID = "SalePrice", "Id"
FAST = True               # True while iterating (minutes); False for the final submission run
SEEDS = [42] if FAST else [42, 1, 2]
HOLDOUT_SHARE, HOLDOUT_SEED = 0.20, 2026
RUN_MID = True            # glass-box surrogate (needs the midlearn package and R; see the install cell)
TEAM = "Your name (FCAS, FCIA)"

import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display, Markdown
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

_cands = [f"/kaggle/input/{COMPETITION}", os.environ.get("IAA_DATA", "data")]
DATA_DIR = next(p for p in _cands if os.path.exists(os.path.join(p, "train.csv")))
T0 = time.time()
BLUE, ORANGE, AQUA, YELLOW, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#0f141a", "#7a838f", "#e3e6ea"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.edgecolor": "#c3c8cf", "axes.labelcolor": "#48515d", "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})
rmsle = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))   # inputs already in log1p space
print("data:", DATA_DIR, "| fast mode:", FAST, "| split seeds:", SEEDS)
""")

code(r"""
# ---------- the AI decision log (also saved as ai_log.json); input hidden on Kaggle to keep the page readable ----------
AI_LOG = __AI_LOG__
print(f"{len(AI_LOG)} logged exchanges")
""", hidden=True)
code(r"""
# ---------- render the log as a ledger and as exchange cards ----------

def _esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def card(i):
    e = next(x for x in AI_LOG if x["id"] == i)
    rows = [("Question", e["question"]), ("AI answer", e["ai_answer"]), ("Challenge", e["challenge"]), ("Test", e["test"]),
            ("Result", e["result"]), ("Decision", e["decision"]), ("Effect", e["effect"])]
    body = "".join(f'<tr><td style="color:#7a838f;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap;text-align:left">{k}</td>'
                   f'<td style="padding:3px 0;text-align:left">{_esc(v)}</td></tr>' for k, v in rows if v)
    return HTML(f'<div style="border:1px solid #d8dde3;border-radius:8px;padding:10px 14px;margin:10px 0;background:#fcfcfb;max-width:900px">'
                f'<div style="font-family:monospace;font-size:12px;color:#1c5cab">EXCHANGE {e["id"]} · {_esc(e["role"])} · raised by {_esc(e["proposed_by"])} · {_esc(e["status"])}</div>'
                f'<table style="font-size:13.5px;line-height:1.45;margin-top:6px">{body}</table></div>')

def ledger():
    df = pd.DataFrame(AI_LOG)[["id", "stage", "role", "proposed_by", "decision", "status"]]
    df.columns = ["Ref", "Stage", "AI role", "Raised by", "Decision", "Status"]
    return df.style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "12.5px"})

# the working loop and the team
fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.2)
steps = [("Question", "actuary or AI"), ("Proposal", "AI role"), ("Challenge", "actuary"), ("Test", "AI runs it"), ("Evidence", "numbers, exhibits"), ("Decision", "actuary signs")]
for k, (t, s) in enumerate(steps):
    x = 0.3 + k * 1.8
    ax.add_patch(plt.Rectangle((x, 1.6), 1.45, 0.9, color=BLUE if t != "Decision" else ORANGE, alpha=.12 if t != "Decision" else .22, lw=0))
    ax.text(x + .72, 2.13, t, ha="center", fontsize=11, weight="bold", color=INK); ax.text(x + .72, 1.82, s, ha="center", fontsize=8.5, color=MUTED)
    if k < len(steps) - 1: ax.annotate("", (x + 1.75, 2.05), (x + 1.5, 2.05), arrowprops=dict(arrowstyle="->", color=MUTED))
ax.annotate("", (1.0, 1.55), (9.9, 1.55), arrowprops=dict(arrowstyle="->", color=MUTED, connectionstyle="arc3,rad=-0.07"))
ax.text(5.5, 0.45, "logged as an exchange card; a failed test sends the question back round the loop", ha="center", fontsize=9, color=MUTED)
ax.text(5.5, 0.05, "AI roles: data steward · modeller · independent reviewer · filing writer · plain-language explainer", ha="center", fontsize=9.5, color=INK)
plt.show()
display(ledger())
""")

md(r"""
**AI roles.** The appointed actuary sets each role up once with a standing prompt (section 5) and then gives it tasks.

| Role | What it does here | Judging criterion it serves |
|---|---|---|
| Data steward | Reads the data dictionary, classifies blanks, tests where the data came from | EDA and feature understanding |
| Modeller | Reproduces the incumbent, proposes challengers that make different errors | Prompting for modelling strategy |
| Independent reviewer | A fresh session that did not build the model, asked to break it | Explaining or validating outputs |
| Filing writer | Drafts the filing-style documentation and the judgment register | Automating tedious work, interpretability |
| Plain-language explainer | Writes the green "how to read this" boxes | Explaining outputs |
| Appointed actuary (human) | Challenges every claim, makes and signs the judgments | |
""")

# =====================================================================================
md(r"""
## 3. Step-by-step analysis
### 3.1 Data understanding

<div style="border:1px dashed #c3c8cf;border-radius:8px;padding:8px 14px;color:#48515d"><b>What a reviewer asks.</b> Identify data sources, how data was adjusted, how missing values and duplicates were handled, which records were excluded and why, and material limitations. <i>(NAIC model white paper A.1–A.4; FSRA rating and underwriting guidance, data governance)</i></div>

**Purpose and intended use.** A filing starts with what the number is for, because that decides which inputs are allowed and which error costs more. We document this model as an **insured-value check at homeowners underwriting**: under-valuing a home leaves it underinsured, so the low side of the uncertainty range matters most. *(Judgment J1, section 4.)*
""")

code(r"""
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv")); test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
feat = [c for c in train.columns if c not in (ID, TARGET)]
num = train[feat].select_dtypes("number").columns.tolist(); txt = [c for c in feat if c not in num]
dups = int(train[feat].duplicated().sum())
display(pd.DataFrame({"": ["Training rows", "Test rows", "Columns", "Numeric / text", "Duplicate rows", "Median price", "Skewness: raw, log"],
                      "Value": [f"{len(train):,}", f"{len(test):,}", len(feat), f"{len(num)} / {len(txt)}", dups,
                                f"${train[TARGET].median():,.0f}", f"{train[TARGET].skew():.2f}, {np.log1p(train[TARGET]).skew():.2f}"]}).style.hide(axis="index"))
fig, axs = plt.subplots(1, 2, figsize=(11, 3))
axs[0].hist(train[TARGET] / 1e3, bins=30, color=BLUE, rwidth=.9); axs[0].set_title("Sale price, $000"); axs[0].grid(axis="x")
axs[1].hist(np.log1p(train[TARGET]), bins=30, color=BLUE, rwidth=.9); axs[1].set_title("log(1 + price): the scale every model uses"); axs[1].grid(axis="x")
plt.tight_layout(); plt.show()
""")
md(r"""
<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px">
<b>How to read this.</b> Skewness measures lopsidedness; 0 is symmetric. Raw prices are stretched by a few expensive houses. On the log scale, equal steps are equal percentage changes ($100k to $200k is the same distance as $200k to $400k), which is how prices and claim sizes behave. It is the same reason severity is modelled with a log link.
</div>
""")
code(r"""display(card("D2"))""")

# =====================================================================================
md(r"""
### 3.2 Data cleaning and preprocessing

<div style="border:1px dashed #c3c8cf;border-radius:8px;padding:8px 14px;color:#48515d"><b>What a reviewer asks.</b> Provide a data dictionary, the rationale linking each variable to the risk, variables considered and rejected, and how inputs were transformed. <i>(NAIC B.3, C.2, C.5; FSRA guidance, proxy and point-of-use)</i></div>
""")
code(r"""
# ---------- data steward's rules, read from data_description.txt (exchange D1) ----------
NA_NONE = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']      # blank = feature absent
NA_ZERO = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
NA_GROUP_MEDIAN = {"LotFrontage": "Neighborhood"}                                                          # estimated within neighbourhood
NA_FROM_COLUMN = {"GarageYrBlt": "YearBuilt"}
QUAL = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ORDINAL = {**{c: QUAL for c in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']},
           'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
           'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
           'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
           'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
           'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
           'CentralAir': {'N': 0, 'Y': 1}, 'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}}
SALE_TIME_VARS = ["MoSold", "YrSold", "SaleType", "SaleCondition"]   # known only after sale: dropped (judgment J2)
SEGMENTS = ["Neighborhood", "OverallQual"]                            # for segment testing in 3.4

def exclusion_rule(d):                                               # non-market sales removed from fitting (judgment J3)
    return (d["Neighborhood"] == "Edwards") & (d["GrLivArea"] > 4000)

def engineer(d):                                                     # engineered rating variables, rationale in the table below
    d["TotalSF"] = d["TotalBsmtSF"] + d["1stFlrSF"] + d["2ndFlrSF"]
    d["TotalBath"] = d["FullBath"] + .5 * d["HalfBath"] + d["BsmtFullBath"] + .5 * d["BsmtHalfBath"]
    d["TotalPorchSF"] = d[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']].sum(axis=1)
    d["HouseAge"] = (d["YrSold"] - d["YearBuilt"]).clip(lower=0)          # age at valuation date
    d["RemodelAge"] = (d["YrSold"] - d["YearRemodAdd"]).clip(lower=0)
    d["QualSF"] = d["OverallQual"] * d["GrLivArea"]
    d["HasGarage"] = (d["GarageArea"] > 0).astype(int); d["HasBsmt"] = (d["TotalBsmtSF"] > 0).astype(int)
    d["MSSubClass"] = d["MSSubClass"].astype(str)
    return d

def clean(d, ref):
    d = d.copy()
    d[NA_NONE] = d[NA_NONE].fillna("None"); d[NA_ZERO] = d[NA_ZERO].fillna(0)
    for c, g in NA_GROUP_MEDIAN.items(): d[c] = d[c].fillna(d[g].map(ref.groupby(g)[c].median())).fillna(ref[c].median())
    for c, src in NA_FROM_COLUMN.items(): d[c] = d[c].fillna(d[src])
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype) in ("str", "string"): d[c] = d[c].fillna(ref[c].mode().iloc[0]).astype(str)
    for c, m in ORDINAL.items(): d[c + "_o"] = d[c].map(m).fillna(0)
    d = engineer(d)
    return d.drop(columns=[c for c in SALE_TIME_VARS if c in d.columns])

miss = (train[feat].isna().sum() + test[feat].isna().sum()); miss = miss[miss > 0].sort_values(ascending=False)
rule = {**{c: "Means none" for c in NA_NONE}, **{c: "Means zero" for c in NA_ZERO}, **{c: "Estimated" for c in list(NA_GROUP_MEDIAN) + list(NA_FROM_COLUMN)}}
colors = {"Means none": BLUE, "Means zero": ORANGE, "Estimated": AQUA, "Truly missing": YELLOW}
fig, ax = plt.subplots(figsize=(11, 0.22 * len(miss) + 0.8))
ax.barh(miss.index[::-1], miss.values[::-1], color=[colors[rule.get(c, "Truly missing")] for c in miss.index[::-1]], height=.6)
for k, v in colors.items(): ax.barh([], [], color=v, label=k)
ax.legend(frameon=False, loc="lower right"); ax.set_title(f"Missing values by column ({len(miss)} columns, {int(miss.sum()):,} cells): treatment from the data dictionary")
ax.grid(axis="y"); plt.tight_layout(); plt.show()

ytr_all = np.log1p(train[TARGET].values)
trc_all, tec = clean(train.drop(columns=[TARGET]), train), clean(test, train)
excl = exclusion_rule(trc_all).values
trc = trc_all[~excl].drop(columns=[ID]).reset_index(drop=True); y = ytr_all[~excl]; ids_te = tec[ID].values; tec = tec.drop(columns=[ID])
print(f"excluded from fitting: {int(excl.sum())} rows; fitting rows: {len(trc):,}; sale-time variables dropped: {SALE_TIME_VARS}")
""")
code(r"""display(card("D1")); display(card("D7"))""")
md(r"""
| Engineered variable | Definition | Why it relates to value |
|---|---|---|
| TotalSF | Basement + first + second floor area | Buyers pay for usable space wherever it sits |
| TotalBath | Full baths + ½ × half baths, above and below grade | Utility per occupant |
| HouseAge, RemodelAge | Years from build or remodel to valuation | Depreciation of the structure |
| QualSF | Overall quality × living area | Quality is priced per square foot |
| Quality codes as numbers | Po, Fa, TA, Gd, Ex coded 1 to 5 | Keeps the order the codes carry |
""")
code(r"""
# ---------- two design matrices: linear models need logs and indicators; trees take raw columns ----------
cat_cols = [c for c in trc.columns if trc[c].dtype == object or str(trc[c].dtype) in ("str", "string")]
num_cols = [c for c in trc.columns if c not in cat_cols]
full = pd.concat([trc, tec], keys=["tr", "te"])
lin = full[num_cols].astype(float).copy()
skewed = [c for c in num_cols if lin[c].min() >= 0 and lin[c].skew() > 0.75]
lin[skewed] = np.log1p(lin[skewed])
lin = pd.concat([lin, pd.get_dummies(full[cat_cols], dtype=float)], axis=1); lin = lin.loc[:, lin.loc["tr"].nunique() > 1]
X_lin, X_lin_te = lin.loc["tr"].values, lin.loc["te"].values
tree = full.copy()
for c in cat_cols: tree[c] = pd.Categorical(tree[c])
X_tree, X_tree_te = tree.loc["tr"].reset_index(drop=True), tree.loc["te"].reset_index(drop=True)
X_cb, X_cb_te = full.loc["tr"].reset_index(drop=True), full.loc["te"].reset_index(drop=True)
print(f"linear design: {X_lin.shape[1]} columns ({len(skewed)} log-transformed); tree design: {X_tree.shape[1]} columns")
""")

# =====================================================================================
md(r"""
### 3.3 Modeling

<div style="border:1px dashed #c3c8cf;border-radius:8px;padding:8px 14px;color:#48515d"><b>What a reviewer asks.</b> Model type and why it suits its role; how data was split for validation; the target; link and distribution; how variables were selected. <i>(NAIC B.1, B.2)</i></div>

The target is log price, the equivalent of a log-link model: effects multiply and errors are percentages. Eight models from three families are fitted; the **current model** is the penalised GLM (Ridge), the kind of rating model already in force; the **proposed model** is a non-negative blend of all eight with weights fitted out of fold.
""")
code(r"""
# ---------- model zoo (FAST uses lighter tree settings) ----------
k = 0.3 if FAST else 1.0
def zoo(seed):
    return {
        "Ridge (current)": lambda: RidgeCV(alphas=np.logspace(-1, 3, 30)),
        "Lasso": lambda: LassoCV(alphas=np.logspace(-4.5, -1.5, 30), max_iter=20000, cv=5, random_state=0),
        "Elastic Net": lambda: ElasticNetCV(l1_ratio=[.3, .6, .9], alphas=np.logspace(-4.5, -1.5, 20), max_iter=20000, cv=5, random_state=0),
        "SVR": lambda: SVR(C=20, epsilon=0.008, gamma=0.0003),
        "Kernel ridge": lambda: KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5),
        "LightGBM": lambda: LGBMRegressor(num_leaves=6, learning_rate=0.01 / k, n_estimators=int(4000 * k), subsample=0.7, subsample_freq=1,
                                          colsample_bytree=0.25, min_child_samples=8, reg_alpha=0.05, reg_lambda=0.5, random_state=seed, verbosity=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=int(3000 * k), learning_rate=0.01 / k, max_depth=4, min_child_weight=2, subsample=0.7,
                                        colsample_bytree=0.3, reg_alpha=0.005, enable_categorical=True, tree_method="hist", random_state=seed),
        "CatBoost": lambda: CatBoostRegressor(iterations=int(2500 * k), learning_rate=0.03 / k, depth=6, l2_leaf_reg=3, random_seed=seed, verbose=0),
    }
LINEAR = {"Ridge (current)", "Lasso", "Elastic Net", "SVR", "Kernel ridge"}
BENCH = "Ridge (current)"

def fit_predict(name, make, rows, pred_rows=None, test=False):
    m = make()
    if name in LINEAR:
        s = StandardScaler().fit(X_lin[rows]); m.fit(s.transform(X_lin[rows]), y[rows])
        return m.predict(s.transform(X_lin_te if test else X_lin[pred_rows]))
    if name == "CatBoost":
        m.fit(X_cb.iloc[rows], y[rows], cat_features=cat_cols); return m.predict(X_cb_te if test else X_cb.iloc[pred_rows])
    m.fit(X_tree.iloc[rows], y[rows]); return m.predict(X_tree_te if test else X_tree.iloc[pred_rows])

def nnls_w(P, t):
    w, _ = nnls(P, t); return w / w.sum()

def run(rows, seed):
    # out-of-fold predictions for every model on `rows`, nested blend weights, and the blend's OOF
    Z = zoo(seed); folds = list(KFold(5, shuffle=True, random_state=seed).split(rows))
    oof = pd.DataFrame(index=range(len(rows)), columns=list(Z), dtype=float)
    for name, make in Z.items():
        for a, b in folds: oof.loc[b, name] = fit_predict(name, make, rows[a], rows[b])
    P, t = oof.values, y[rows]; nested = np.zeros(len(rows))
    for a, b in folds: nested[b] = P[b] @ nnls_w(P[a], t[a])
    return oof, nested, nnls_w(P, t), Z

ALL = np.arange(len(y)); RES = {}
for s in SEEDS:
    oof, nested, w, Z = run(ALL, s)
    RES[s] = dict(oof=oof, blend=nested, w=w)
    print(f"seed {s}: blend {rmsle(nested, y):.4f} | current {rmsle(oof[BENCH], y):.4f}  ({time.time() - T0:.0f}s)")
""")
code(r"""
# ---------- results by model across fold splits ----------
tab = pd.DataFrame({s: {**{m: rmsle(RES[s]["oof"][m], y) for m in Z}, "Proposed blend": rmsle(RES[s]["blend"], y)} for s in SEEDS})
tab["Mean"] = tab.mean(axis=1); tab = tab.sort_values("Mean")
fig, ax = plt.subplots(figsize=(11, 0.42 * len(tab) + 0.8))
for i, (m, r) in enumerate(tab[::-1].iterrows()):
    c = ORANGE if m == "Proposed blend" else (MUTED if m == BENCH else BLUE)
    ax.plot([r[SEEDS].min(), r[SEEDS].max()], [i, i], color=c, alpha=.4, lw=2); ax.scatter(r[SEEDS], [i] * len(SEEDS), color=c, s=40, zorder=3)
    ax.text(r[SEEDS].max() + 0.0004, i, f"{r['Mean']:.4f}", va="center", fontsize=9, color=MUTED)
ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab.index[::-1]); ax.grid(axis="y")
ax.set_title("Out-of-fold error by model (each dot one fold split; lower is better)"); plt.tight_layout(); plt.show()
w = RES[SEEDS[0]]["w"]
display(pd.DataFrame({"Blend weight": [f"{v:.1%}" for v in w]}, index=list(Z)).sort_values("Blend weight", ascending=False).T)
""")
md(r"""
<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px">
<b>How to read this.</b> Cross-validation is a back-test run five times: each house is priced by a model that never saw its sale price, like testing a reserving method on accident years it was not fitted to. Each dot is one shuffle of the houses; the spread along a row is luck of the shuffle, so two models closer than that cannot be told apart. A blend weight of 37% means the final log price is 37% that model's view.
</div>
""")
code(r"""display(card("D3")); display(card("D4"))""")

# =====================================================================================
md(r"""
### 3.4 Results

<div style="border:1px dashed #c3c8cf;border-radius:8px;padding:8px 14px;color:#48515d"><b>What a reviewer asks.</b> Fit on validation data, stability, overfitting, why the new model beats the old (lift and double-lift), interpretability of the rating plan, rating examples, impact on each policy, and outcomes testing by segment. <i>(NAIC B.4, B.5, C.1–C.7; FSRA standard filing Appendix C; FSRA guidance, outcomes testing)</i></div>

#### Paired comparison and double lift
""")
code(r"""
rng = np.random.default_rng(0); rows = []
for s in SEEDS:
    ec, ep = RES[s]["oof"][BENCH].values - y, RES[s]["blend"] - y
    bs = [rmsle(ec[i], 0) - rmsle(ep[i], 0) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    rows.append([s, rmsle(ec, 0), rmsle(ep, 0), rmsle(ec, 0) - rmsle(ep, 0), f"{np.quantile(bs, .025):.4f} to {np.quantile(bs, .975):.4f}"])
display(pd.DataFrame(rows, columns=["Split seed", "Current", "Proposed", "Gain", "95% CI of gain"]).style.hide(axis="index").format({"Current": "{:.4f}", "Proposed": "{:.4f}", "Gain": "{:.4f}"}))

o = RES[SEEDS[0]]; act, pm, pc = np.expm1(y), np.expm1(o["blend"]), np.expm1(o["oof"][BENCH].values)
d = pd.DataFrame({"q": pd.qcut(pm / pc, 10, labels=False), "a": act, "m": pm, "c": pc}).groupby("q").sum()
fig, ax = plt.subplots(figsize=(11, 3.3))
ax.axhline(1, color=INK, lw=1); ax.plot(d.index + 1, d.a / d.m, "-o", color=BLUE, label="Proposed"); ax.plot(d.index + 1, d.a / d.c, "-o", color=ORANGE, label="Current")
ax.set_xticks(range(1, 11)); ax.set_xlabel("decile of proposed ÷ current (left: proposed lower; right: proposed higher)"); ax.set_ylabel("actual ÷ predicted")
ax.set_title("Double lift, out of fold: where the models disagree, which tracks actual prices?"); ax.legend(frameon=False); plt.tight_layout(); plt.show()
""")
md(r"""
<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px">
<b>How to read this.</b> The 95% interval comes from resampling the houses 1,000 times; if it stays above zero, the gain is not luck. In the double lift, a line at 1.00 means that model's predictions equal actual prices on average for that group of houses. Every point uses out-of-fold predictions, so a line near 1.00 is earned, not memorised.
</div>

#### Overfitting check: apparent, cross-validated and locked hold-out error
""")
code(r"""
# lock away HOLDOUT_SHARE of houses before any fitting, weighting or selection; run the whole pipeline on the rest; score once
hold = np.sort(np.random.default_rng(HOLDOUT_SEED).choice(len(y), int(round(HOLDOUT_SHARE * len(y))), replace=False)); fit = np.setdiff1d(ALL, hold)
oof_f, nested_f, w_f, Zf = run(fit, SEEDS[0])
p_fit, p_hold = {}, {}
for name, make in Zf.items():
    p_fit[name] = fit_predict(name, make, fit, fit); p_hold[name] = fit_predict(name, make, fit, hold)
blend_fit = sum(wk * p_fit[m] for m, wk in zip(Zf, w_f)); blend_hold = sum(wk * p_hold[m] for m, wk in zip(Zf, w_f))
OF = pd.DataFrame({"Apparent (fitted houses)": [rmsle(p_fit[BENCH], y[fit]), rmsle(blend_fit, y[fit])],
                   "Cross-validated": [rmsle(oof_f[BENCH], y[fit]), rmsle(nested_f, y[fit])],
                   f"Locked hold-out ({len(hold)} houses)": [rmsle(p_hold[BENCH], y[hold]), rmsle(blend_hold, y[hold])]}, index=["Current", "Proposed"])
display(OF.style.format("{:.4f}"))
ec, ep = p_hold[BENCH] - y[hold], blend_hold - y[hold]
bs = [rmsle(ec[i], 0) - rmsle(ep[i], 0) for i in (rng.integers(0, len(hold), len(hold)) for _ in range(1000))]
print(f"hold-out gain {rmsle(ec, 0) - rmsle(ep, 0):.4f}; 95% interval {np.quantile(bs, .025):.4f} to {np.quantile(bs, .975):.4f}")
""")
code(r"""display(card("D6"))""")
md(r"""
<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px">
<b>How to read this.</b> Overfitting is memorising the training houses rather than learning what drives price, like a student who memorised last year's exam. "Apparent" error, measured on the houses a model was fitted to, shows exactly that: it looks far better than on new houses. No score in this notebook is measured that way. The locked hold-out is a sealed envelope opened once; if its error agrees with the cross-validated error, the reported scores are not an artefact of overfitting.
</div>

#### 90% range and its coverage
""")
code(r"""
r = RES[SEEDS[0]]["blend"] - y; cover = np.zeros(len(r), bool)
for a, b in KFold(5, shuffle=True, random_state=99).split(r): cover[b] = np.abs(r[b]) <= np.quantile(np.abs(r[a]), 0.9)
Q90 = float(np.quantile(np.abs(r), 0.9))
band = pd.qcut(np.expm1(y), 5, labels=["cheapest fifth", "2nd", "3rd", "4th", "dearest fifth"])
cov = pd.Series(cover).groupby(band.codes).mean(); cov.index = band.categories
fig, ax = plt.subplots(figsize=(11, 2.8)); ax.bar(cov.index, cov.values, color=[BLUE if v >= .885 else ORANGE for v in cov.values], width=.35)
ax.axhline(.9, color=INK, lw=1); ax.set_ylim(.6, 1); ax.set_title(f"90% range = prediction × {np.exp(-Q90):.3f} to × {np.exp(Q90):.3f}; coverage {cover.mean():.1%} overall, by price band")
ax.grid(axis="x"); plt.tight_layout(); plt.show()
""")
md(r"""
#### The rating plan: the blend read as base rate × factors (MID glass box)

MID (by the 2025 winner, Ryoichi Asashiba) fits the additive model that best reproduces the blend's predictions: base plus one term per variable plus selected interactions. In log terms the terms add; in dollars they multiply, like a log-link GLM rating plan.
""")
code(r"""
# install on Kaggle if needed:  !pip install -q midlearn  &&  !apt-get install -y -q r-cran-rcppeigen r-cran-midr
MID_OK = False
if RUN_MID:
    try:
        from midlearn import MIDExplainer
        from sklearn.base import BaseEstimator, RegressorMixin
        REN = {c: "X" + c for c in trc.columns if c[0].isdigit()}; BACK = {v: k for k, v in REN.items()}
        Zfull = zoo(SEEDS[0]); FITTED = {}
        for name, make in Zfull.items():
            m = make()
            if name in LINEAR: s_ = StandardScaler().fit(X_lin); m.fit(s_.transform(X_lin), y); FITTED[name] = (m, s_)
            elif name == "CatBoost": m.fit(X_cb, y, cat_features=cat_cols); FITTED[name] = (m, None)
            else: m.fit(X_tree, y); FITTED[name] = (m, None)
        def design(df):
            df = df.rename(columns=BACK)[trc.columns]; l = df[num_cols].astype(float).copy(); l[skewed] = np.log1p(l[skewed].clip(lower=0))
            l = pd.concat([l, pd.get_dummies(df[cat_cols], dtype=float)], axis=1).reindex(columns=lin.columns, fill_value=0.0).values
            t = df.copy()
            for c in cat_cols: t[c] = pd.Categorical(t[c], categories=X_tree[c].cat.categories)
            return l, t.reset_index(drop=True), df.reset_index(drop=True)
        class Blend(RegressorMixin, BaseEstimator):
            def fit(self, X=None, y_=None): return self
            def predict(self, X):
                l, t, cb = design(X); out = np.zeros(len(cb))
                for (name, (m, s_)), wk in zip(FITTED.items(), w):
                    if wk > 0: out += wk * (m.predict(s_.transform(l)) if s_ is not None else m.predict(cb if name == "CatBoost" else t))
                return out
        terms = [REN.get(c, c) for c in trc.columns if not c.endswith("_o") and c != "QualSF"] + ["GrLivArea:Neighborhood"]
        Xte_r = tec.rename(columns=REN)
        mid = MIDExplainer(estimator=Blend(), penalty=0.1, params_main=25, singular_ok=True, terms=terms, verbosity=0).fit(Xte_r)
        MID_OK = True
        print(f"rating plan reproduces {1 - mid.ratio:.2%} of the blend's variation on the test houses")
        q = mid.main_effects("OverallQual"); q["relativity"] = np.exp(q["mid"])
        fig, ax = plt.subplots(figsize=(11, 2.8)); labels = [f"{int(np.ceil(a))}" + ("" if np.ceil(a) == np.floor(b) else f"–{int(np.floor(b))}") for a, b in zip(q.OverallQual_min, q.OverallQual_max)]
        ax.bar(labels, q.relativity - 1, bottom=1, color=[BLUE if v >= 1 else RED for v in q.relativity], width=.4); ax.axhline(1, color=INK, lw=1)
        for i, v in enumerate(q.relativity): ax.text(i, v + (0.01 if v >= 1 else -0.03), f"{v:.2f}", ha="center", fontsize=9, color=MUTED)
        ax.set_title("Overall quality relativities from the rating plan (1.00 = average quality)"); ax.set_xlabel("quality grade"); ax.grid(axis="x"); plt.tight_layout(); plt.show()
    except Exception as ex:
        print("MID skipped:", ex)
""")
md(r"""
<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px">
<b>How to read this.</b> A relativity of 1.09 for grade 8 means a grade-8 house is priced 9% above an otherwise identical house of average quality. "Reproduces 99.9% of the variation" means the rating plan is close to a perfect copy of the blend on these houses; it still loses some accuracy on houses it was not fitted to.
</div>

#### Rating example: one house, priced step by step
""")
code(r"""
pred_te = sum(wk * fit_predict(m, mk, ALL, test=True) for (m, mk), wk in zip(zoo(SEEDS[0]).items(), w)) if not MID_OK else Blend().predict(Xte_r)
row = int(np.argmin(np.abs(np.expm1(pred_te) - np.median(np.expm1(pred_te)))))
print(f"test house {ids_te[row]}: predicted ${np.expm1(pred_te[row]):,.0f}; 90% range ${np.expm1(pred_te[row]) * np.exp(-Q90):,.0f} to ${np.expm1(pred_te[row]) * np.exp(Q90):,.0f}")
if MID_OK:
    br = mid.breakdown(data=Xte_r, row=row).breakdown.head(10); base = float(np.expm1(mid.intercept))
    steps = np.concatenate([[0], np.cumsum(br["mid"].values)])
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(br) + 1))
    for i, (t, v) in enumerate(zip(br["term"], br["mid"])):
        ax.barh(i, v, left=steps[i], color=BLUE if v >= 0 else RED, height=.6); ax.text(max(steps[i], steps[i + 1]) + .003, i, f"{np.exp(v) - 1:+.1%}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(br))); ax.set_yticklabels([t.replace("X1st", "1st").replace("X2nd", "2nd") for t in br["term"]]); ax.invert_yaxis(); ax.grid(axis="y")
    ax.set_title(f"Premium-style build-up: base ${base:,.0f} × each factor (largest ten terms)"); plt.tight_layout(); plt.show()
""")
md(r"""
<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px">
<b>How to read this.</b> Read it like a premium calculation: start at the base rate for the average house and multiply by each factor in turn. A "+9.1%" line multiplies by 1.091; a "−1.9%" line by 0.981. This is the rating example a filing requires.
</div>

#### Impact analysis and segment testing
""")
code(r"""
bench_te = fit_predict(BENCH, zoo(SEEDS[0])[BENCH], ALL, test=True)
chg = np.expm1(pred_te) / np.expm1(bench_te) - 1
bins = [-9, -.15, -.10, -.05, -.02, .02, .05, .10, .15, 9]; lab = ["<−15%", "−15/−10", "−10/−5", "−5/−2", "±2%", "+2/+5", "+5/+10", "+10/+15", ">+15%"]
cnt = pd.cut(chg, bins, labels=lab).value_counts().reindex(lab)
fig, axs = plt.subplots(1, 2, figsize=(11, 3))
axs[0].bar(cnt.index, cnt.values, color=[ORANGE if l in ("<−15%", ">+15%") else BLUE for l in cnt.index], width=.5); axs[0].tick_params(axis="x", labelsize=7.5)
axs[0].set_title(f"Change against current model, {len(chg):,} test houses"); axs[0].grid(axis="x")
seg = pd.DataFrame({"g": trc[SEGMENTS[1]].values, "a": act, "p": pm}).groupby("g").agg(n=("a", "size"), a=("a", "sum"), p=("p", "sum"))
axs[1].bar(seg.index.astype(str), seg.a / seg.p - 1, bottom=1, color=BLUE, width=.5); axs[1].axhline(1, color=INK, lw=1)
axs[1].set_title(f"Actual ÷ predicted by {SEGMENTS[1]}"); axs[1].grid(axis="x"); plt.tight_layout(); plt.show()
ae = pd.DataFrame({"g": trc[SEGMENTS[0]].values, "a": act, "p": pm}).groupby("g").agg(Houses=("a", "size"), a=("a", "sum"), p=("p", "sum"))
ae["A/E"] = (ae.a / ae.p).round(3); display(ae.sort_values("Houses", ascending=False)[["Houses", "A/E"]].head(12).T)
print(f"{np.mean(np.abs(chg) <= .05):.1%} of test houses move by 5% or less; {int(np.sum(np.abs(chg) > .15))} move by more than 15%")
""")
code(r"""display(card("D8"))""")
md(r"""
<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px">
<b>How to read this.</b> The left chart is a dislocation chart: how many houses' prices would move by how much if the proposed model replaced the current one. A/E of 1.00 means actual prices equal predictions on average for that group; 1.07 means the group is under-priced by about 7%. Read A/E with the number of houses, as you would credibility-weight a territory.
</div>
""")

# =====================================================================================
md(r"""
## 4. Findings and insights

**For a stakeholder, in plain language.** *(Write after the final run. Three to five sentences: what drives value, how accurate the model is in dollars, where it should not be trusted, what the business should do with it.)*

**Judgment register.** Every decision a number cannot settle, with the evidence and a recommendation. Items marked "Needs your call" are signed by the appointed actuary, not by the AI.
""")
code(r"""
JUDGMENTS = [
    ("J1", "Intended use and business story", "Insured-value check, collateral value, or portfolio exposure", "Insured-value check at homeowners underwriting", "Needs your call"),
    ("J2", "Variables known only at sale", "Dropping them costs about 0.2% accuracy", "Drop for any pre-sale use (done in 3.2)", "Recommended"),
    ("J3", "Houses resembling the excluded non-market sales", "Thin evidence either way", "Refer to an underwriter rather than price automatically", "Needs your call"),
    ("J4", "File the blend or the rating plan", "The rating plan is less accurate on unseen houses", "File the blend; the plan explains it", "Needs your call"),
    ("J5", "Shape of the 90% range", "Flat range under-covers the cheapest and dearest fifths", "Make it depend on price; widen the low side", "Needs your call"),
    ("J6", "Neighbourhood as a rating variable", "Calibrated within a few percent in large neighbourhoods", "Keep, with periodic proxy testing", "Recommended"),
]
display(pd.DataFrame(JUDGMENTS, columns=["Ref", "Decision", "Evidence", "Recommendation", "Status"]).style.hide(axis="index").set_properties(**{"text-align": "left"}))
""")
code(r"""display(card("D5"))""")

# =====================================================================================
md(r"""
## 5. Reproducibility note

- Every number in this notebook is produced by the code above, top to bottom, from the competition files. Random seeds are fixed (`SEEDS`, `HOLDOUT_SEED`).
- AI prompts and outputs that changed a decision are in `AI_LOG` (rendered as the ledger in section 2); the set-up prompt for each AI role is below; a condensed session transcript is attached as `ai_transcript.md`.
- Planned before submission: an independent review by a fresh AI session that did not build the model (exchange R1).
""")
code(r"""display(card("R1"))""")
md("<details><summary><b>Prompt library: AI role set-ups and challenge prompts</b></summary>\n\n" + PROMPTS + "\n</details>")
code(r"""
# ---------- submission file, checks, environment ----------
import sklearn, lightgbm, xgboost, catboost, platform
sub = pd.DataFrame({ID: ids_te, TARGET: np.expm1(pred_te)})
assert len(sub) == len(test) and sub[TARGET].gt(0).all() and sub[ID].is_unique
sub.to_csv("submission.csv", index=False)
print("submission.csv written:", sub.shape)
print("python", platform.python_version(), "| scikit-learn", sklearn.__version__, "| lightgbm", lightgbm.__version__,
      "| xgboost", xgboost.__version__, "| catboost", catboost.__version__, f"| run time {time.time() - T0:.0f}s")
SUMMARY = dict(team=TEAM, public_lb="(fill after submitting)", n_models=len(Z), n_seeds=len(SEEDS),
               bench_cv=f"{tab.loc[BENCH, 'Mean']:.4f}", blend_cv=f"{tab.loc['Proposed blend', 'Mean']:.4f}",
               blend_cv_pct=f"{np.expm1(tab.loc['Proposed blend', 'Mean']):.0%}", blend_cv_dollars=f"${300000 * np.expm1(tab.loc['Proposed blend', 'Mean']) / 1000:,.0f}k",
               bench_ho=f"{OF.iloc[0, 2]:.4f}", blend_ho=f"{OF.iloc[1, 2]:.4f}", n_hold=len(hold), within5=f"{np.mean(np.abs(chg) <= .05):.0%}")
json.dump(SUMMARY, open("results_summary.json", "w"))
""")

# =====================================================================================
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                    "language_info": {"name": "python"}})
src = json.dumps(LOG, ensure_ascii=False, indent=1)
for c in nb.cells:
    if c.cell_type == "code" and "__AI_LOG__" in c.source:
        c.source = c.source.replace("__AI_LOG__", src)
out = HERE / "iaa_notebook.ipynb"

if "--execute" in sys.argv:
    from nbclient import NotebookClient
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else None
    run_dir = HERE / "run"; run_dir.mkdir(exist_ok=True)
    if data: os.environ["IAA_DATA"] = data
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(run_dir)}}).execute()
    S = json.load(open(run_dir / "results_summary.json"))
    for c in nb.cells:
        if c.cell_type == "markdown":
            c.source = re.sub(r"\{\{(\w+)\}\}", lambda m: str(S.get(m.group(1), m.group(0))), c.source)
nbf.write(nb, out)
print("wrote", out)
