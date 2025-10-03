# ==========================================
# General Curriculum Builder (by Department)
# ==========================================
# - Upstream de-duplication using char-3gram TF-IDF + cosine
# - Dept -> Subtopic clustering with TF-IDF (word/bigram) + KMeans
# - Subtopic -> Top-N journal mapping via cosine to centroids
# - Multi-sheet Excel export for Sales

# Setup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load Clean Matched Data
df = pd.read_excel(r"C:\Users\jwang\OneDrive\Desktop\clean_text.xlsx")

# Tunable Knobs (Centralized)
TOPN_JOURNALS = 5 # how many journals to list per subtopic
TOPN_TERMS = 6 # top TF-IDF terms to show in raw subtopic labels
MAX_FEATURES = 20000 # TF-IDF vocab cap
NGRAM_RANGE = (1, 2) # word unigrams + bigrams (captures "supply chain", etc.)
RANDOM_STATE = 42 # determinism for KMeans
SUBJECT_FILTER = False # if True: journals must match subtopic's dominant subject_area

# =======================================================
# 1. Upstream De-Duplication: char-3gram TF-IDF + cosine
# =======================================================
"""
De-duplicate a string Series using TF-IDF over character n-grams + cosine similarity.
Returns a new Series of the same shape, where each entry is replaced by its
component's canonical string (chosen as the longest string; tie-breaker: earliest).
threshold = 0.80 is a good starting point for near-duplicate detection.
"""
def dedup_series_char_ngrams(series: pd.Series, threshold: float = 0.80,
                             analyzer: str = "char", ngram_range=(3,3),
                             lowercase: bool = True) -> pd.Series:

    s = series.fillna("").astype(str).str.strip()
    if len(s) == 0:
        return s

    # Build TF-IDF on char n-grams (robust to transpositions, punctuation, casing)
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase)
    X = vec.fit_transform(s.tolist())

    # Sparse cosine similarity, thresholded to graph edges (i ~ j if sim > threshold)
    S = cosine_similarity(X, dense_output=False)
    S = S.multiply(S > threshold)

    # Connected components over the similarity graph
    n = S.shape[0]
    visited = np.zeros(n, dtype=bool)
    comp_id = -np.ones(n, dtype=int)
    components = []
    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS from node i
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp.append(u)
            # neighbors with sim > threshold
            nbrs = S[u].indices.tolist()
            for v in nbrs:
                if not visited[v]:
                    stack.append(v)
        components.append(comp)
        cid = len(components) - 1
        for u in comp:
            comp_id[u] = cid

    # Choose canonical string per component:
    # longest string wins; if tie, earliest occurrence (stable)
    comp_to_canon = {}
    for comp in components:
        if len(comp) == 1:
            comp_to_canon[comp_id[comp[0]]] = s.iloc[comp[0]]
        else:
            strings = [(u, s.iloc[u]) for u in comp]
            strings_sorted = sorted(strings, key=lambda t: (-len(t[1]), t[0]))
            comp_to_canon[comp_id[comp[0]]] = strings_sorted[0][1]

    # Map each index to its component canonical
    canon = [comp_to_canon[comp_id[i]] for i in range(n)]
    return pd.Series(canon, index=s.index)

# Apply de-duplication to key string fields that often contain near-duplicates.
df["journal_title_canon"] = dedup_series_char_ngrams(df["journal_title"], threshold=0.80)
df["course_title_canon"] = dedup_series_char_ngrams(df["course_title"], threshold=0.85)

# ============================================
# 2. Prep for Modeling (Matched Records Only)
# ============================================
"""
Keep only the needed columns, drop records missing text,
normalize types/whitespace, and drop accidental duplicates.
"""
def prep_df(df: pd.DataFrame) -> pd.DataFrame:

    cols = ["institution","course_id","course_title","course_title_canon","clean_description",
            "journal_title","journal_title_canon","journal_abstract","subject_area","department"]
    d = df[cols].copy()

    # Require both sides of text for matched modeling
    d = d.dropna(subset=["clean_description","journal_abstract","department"])

    # Normalize types/whitespace
    for c in ["institution","course_id","course_title","course_title_canon",
              "journal_title","journal_title_canon","journal_abstract",
              "subject_area","department","clean_description"]:
        d[c] = d[c].astype(str).str.strip()

    # Drop accidental duplicates (same course matched to the same canonical journal)
    d = d.drop_duplicates(subset=["institution","course_id","journal_title_canon"])
    return d

# ========================
# 3. Helpers for Modeling
# ========================
"""
Heuristic #clusters per department:
K ≈ sqrt(n/2), clamped to [3, 10] and never exceeding n.
"""
def pick_k(n_courses: int, lo: int = 3, hi: int = 10) -> int:

    if n_courses <= lo:
        return max(1, n_courses)
    k = int(round(np.sqrt(n_courses / 2)))
    return max(lo, min(k, hi, n_courses))

"""
Raw/technical label from centroid's top TF-IDF features (word/bigram space).
"""
def label_from_center(center_vec: np.ndarray, vocab: np.ndarray, topn: int = 6) -> str:
 
    idx = np.argsort(center_vec)[-topn:][::-1]
    return ", ".join(vocab[idx])

"""
Pick up to `want` representative courses closest to centroid,
preferring distinct institutions for better storytelling.
"""
def choose_exemplars(X_course, center_row, course_rows: pd.DataFrame, want=3):
   
    sim = cosine_similarity(center_row.reshape(1, -1), X_course).ravel()
    order = np.argsort(-sim)

    picks, seen = [], set()
    for r in order:
        inst = course_rows.iloc[r]["institution"]
        if inst not in seen:
            picks.append(r); seen.add(inst)
        if len(picks) == want:
            break

    out = []
    for r in picks:
        out.append({
            "institution": course_rows.iloc[r]["institution"],
            "course_id": course_rows.iloc[r]["course_id"],
            "course_title": course_rows.iloc[r]["course_title_canon"] or course_rows.iloc[r]["course_title"]
        })
    return out

# =====================================
# 4. Core Modeling for One Department
# =====================================
"""
For a single department:
    - Build shared TF-IDF (word/bigram) over courses + journals
    - KMeans over courses -> subtopics
    - Label subtopics (raw TF-IDF terms) + select exemplars
    - Rank top-N journals per subtopic via cosine to centroids
    - Produce summary + coverage pivot
"""
def run_one_department(df_dept: pd.DataFrame,
                       topN_journals: int = TOPN_JOURNALS,
                       subject_filter: bool = SUBJECT_FILTER):
   
    dept_name = df_dept["department"].iloc[0]

    # Distinct entities inside the department
    courses = (df_dept[["institution","course_id","course_title","course_title_canon",
                        "clean_description","subject_area"]]
               .drop_duplicates(subset=["institution","course_id"])
               .reset_index(drop=True))
    journals = (df_dept[["journal_title","journal_title_canon","journal_abstract","subject_area"]]
                .drop_duplicates(subset=["journal_title_canon"])
                .reset_index(drop=True))

    if courses.empty or journals.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # Shared TF-IDF (word/bigram) so centroids ↔ journals live in same space
    n_docs = len(courses) + len(journals)
    min_df = 1 if n_docs < 200 else 3  # small slices need min_df=1 for stability

    corpus = pd.concat([courses["clean_description"], journals["journal_abstract"]], axis=0).tolist()
    vec = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=min_df, max_features=MAX_FEATURES,
                          stop_words="english", strip_accents="unicode")
    X = vec.fit_transform(corpus)
    vocab = np.array(vec.get_feature_names_out())

    nC = len(courses)
    X_course = X[:nC]
    X_journal = X[nC:]

    # KMeans: courses -> subtopics
    K = pick_k(nC)
    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
    clusters = km.fit_predict(X_course)
    centers = km.cluster_centers_  # dense (K x |V|)

    # Raw technical labels from centroid features
    sub_labels = [label_from_center(centers[k], vocab, topn=TOPN_TERMS) for k in range(K)]
    label_map = dict(enumerate(sub_labels))

    # Attach subtopic to each course
    courses_out = courses.assign(
        department = dept_name,
        sub_id = clusters,
        sub_label = [label_map[c] for c in clusters]
    )

    # Dominant subject_area per subtopic (join by keys; safer than positional iloc)
    course_keys = courses_out[["institution","course_id","sub_id"]]
    # Join back so each course–journal row in df_dept knows its sub_id
    df_join = df_dept.merge(course_keys, on=["institution","course_id"], how="inner")

    dom_subject_by_sub = {}
    for k in range(K):
        sub_rows = df_join[df_join["sub_id"] == k]
        vc = sub_rows["subject_area"].value_counts()
        dom_subject_by_sub[k] = vc.idxmax() if not vc.empty else None

    # Exemplars per subtopic (1–3 courses, different institutions)
    exemplar_rows = []
    for k in range(K):
        idx = np.where(clusters == k)[0]
        ex = choose_exemplars(X_course[idx], centers[k], courses.iloc[idx], want=3)
        for e in ex:
            exemplar_rows.append({
                "department": dept_name,
                "sub_id": k,
                "sub_label": label_map[k],
                **e
            })
    exemplars = pd.DataFrame(exemplar_rows)

    # Rank journals by cosine to subtopic centroid
    sim_sub_jr = cosine_similarity(centers, X_journal)  # (K x J)
    rows = []
    for k in range(K):
        ranked = np.argsort(-sim_sub_jr[k])

        if subject_filter and dom_subject_by_sub[k] is not None:
            mask = (journals["subject_area"].values == dom_subject_by_sub[k])
            cand = [j for j in ranked if mask[j]]
        else:
            cand = ranked.tolist()

        top_idx = cand[:topN_journals]
        for j in top_idx:
            rows.append({
                "department": dept_name,
                "sub_id": k,
                "sub_label": label_map[k],
                "journal_title": journals.loc[j, "journal_title_canon"] or journals.loc[j, "journal_title"],
                "subject_area":  journals.loc[j, "subject_area"],
                "similarity": float(sim_sub_jr[k, j])
            })
    subtopic_journals = pd.DataFrame(rows)

    # Summary + Coverage
    summary = (courses_out
               .groupby(["department","sub_id","sub_label"])
               .agg(n_courses=("course_id","nunique"),
                    n_institutions=("institution","nunique"))
               .reset_index()
               .sort_values(["department","n_institutions","n_courses"], ascending=[True, False, False]))

    coverage = (courses_out.assign(present=1)
                .pivot_table(index=["department","institution"],
                             columns="sub_label", values="present",
                             aggfunc="sum", fill_value=0)
                .reset_index())

    return courses_out, subtopic_journals, summary, coverage, exemplars

# ===================================
# 5. Orchestrate for All Departments
# ===================================
"""
Full pipeline:
    - prep with canonical titles
    - loop departments
    - concatenate deliverables
"""
def run_all_departments(df: pd.DataFrame,
                        topN_journals: int = TOPN_JOURNALS,
                        subject_filter: bool = SUBJECT_FILTER):
    d0 = prep_df(df)
    all_courses, all_jr, all_summ, all_cov, all_ex = [], [], [], [], []

    for dept in sorted(d0["department"].unique()):
        slice_ = d0[d0["department"] == dept].copy()
        c, j, s, v, e = run_one_department(slice_,
                                           topN_journals=topN_journals,
                                           subject_filter=subject_filter)
        if not c.empty: all_courses.append(c)
        if not j.empty: all_jr.append(j)
        if not s.empty: all_summ.append(s)
        if not v.empty: all_cov.append(v)
        if not e.empty: all_ex.append(e)

    COURSES_SUB = pd.concat(all_courses, ignore_index=True) if all_courses else pd.DataFrame()
    SUBTOPIC_JR = pd.concat(all_jr, ignore_index=True) if all_jr else pd.DataFrame()
    SUB_SUMMARY = pd.concat(all_summ, ignore_index=True) if all_summ else pd.DataFrame()
    COVERAGE_PIV = pd.concat(all_cov, ignore_index=True) if all_cov else pd.DataFrame()
    EXEMPLARS = pd.concat(all_ex, ignore_index=True) if all_ex else pd.DataFrame()

    return COURSES_SUB, SUBTOPIC_JR, SUB_SUMMARY, COVERAGE_PIV, EXEMPLARS

# ======================
# 6) Execute and Export
# ======================
COURSES_SUB, SUBTOPIC_JR, SUB_SUMMARY, COVERAGE_PIV, EXEMPLARS = run_all_departments(
    df, topN_journals=TOPN_JOURNALS, subject_filter=SUBJECT_FILTER
)
print("Courses with sub-topics:", COURSES_SUB.shape)
print("Sub-topic → journals:", SUBTOPIC_JR.shape)
print("Sub-topic summary:", SUB_SUMMARY.shape)
print("Coverage pivot:", COVERAGE_PIV.shape)
print("Exemplars:", EXEMPLARS.shape)

# Single Excel with Multiple Sheets
with pd.ExcelWriter(r"C:\Users\jwang\OneDrive - Emerald Group Publishing Ltd\Desktop\General Curriculum\general_curriculum_department.xlsx", engine="openpyxl") as writer:
    COURSES_SUB.to_excel(writer, sheet_name="Courses_Subtopics", index=False)
    SUBTOPIC_JR.to_excel(writer, sheet_name="Subtopic_Journals", index=False)
    SUB_SUMMARY.to_excel(writer, sheet_name="Summary", index=False)
    COVERAGE_PIV.to_excel(writer, sheet_name="Coverage_Pivot", index=False)
    EXEMPLARS.to_excel(writer, sheet_name="Exemplars", index=False)
