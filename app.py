import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import duckdb

st.set_page_config(
    page_title="Sports News RecSys & Propensity Demo",
    page_icon="🏆",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 1. DATA LAYER (Swap these cached loaders with MIND-small TSV files later)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    con = duckdb.connect("data/mind.duckdb", read_only=True)
    articles = con.execute("SELECT * FROM sports_news").df()
    interactions = con.execute("SELECT * FROM sports_interactions").df()
    users = con.execute("SELECT * FROM user_features").df()
    con.close()

    # 1. Fill nulls and add deterministic simulated article age (MIND has no publish time)
    rng = np.random.default_rng(42)
    articles["subcategory"] = articles["subcategory"].fillna("sports").astype(str)
    articles["title"] = articles["title"].fillna("").astype(str)
    articles["abstract"] = articles["abstract"].fillna("").astype(str)
    if "hours_ago" not in articles.columns:
        articles["hours_ago"] = rng.integers(1, 72, size=len(articles))

    # 2. Add derived UI/propensity columns for real MIND users
    if "avg_read_time_min" not in users.columns:
        users["avg_read_time_min"] = np.clip(
            2.0 + 0.35 * users["articles_read_30d"] + rng.normal(0, 0.8, len(users)),
            1.0,
            12.0,
        ).round(1)

    if "persona" not in users.columns:
        users["persona"] = np.where(
            users["sport_diversity"] >= 3,
            "Multi-Sport Fan",
            np.where(users["articles_read_30d"] >= 10, "Avid Single-Sport Reader", "Casual Reader"),
        )

    # Pick 20 random users + 1 synthetic Cold-Start visitor for the sidebar
    top_users = users.sort_values("articles_read_30d", ascending=False).sample(20).copy()
    cold_start = pd.DataFrame(
        [
            {
                "user_id": "COLD_START_USER",
                "articles_read_30d": 0,
                "sessions_7d": 1,
                "sport_diversity": 0,
                "avg_read_time_min": 1.0,
                "persona": "Cold-Start New Visitor",
            }
        ]
    )
    featured_users = pd.concat([top_users, cold_start], ignore_index=True)

    return articles, interactions, featured_users

# ---------------------------------------------------------------------------
# 2. MODELS (Content-Based, Collaborative Filtering, Subscription Propensity)
# ---------------------------------------------------------------------------
@st.cache_resource
def fit_models(articles: pd.DataFrame, interactions: pd.DataFrame):
    # A. Content-Based: TF-IDF
    corpus = articles["subcategory"] + " " + articles["title"] + " " + articles["abstract"]
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(corpus)

    # B. Collaborative Filtering: Sparse User-Item Matrix + TruncatedSVD
    user_ids = interactions["user_id"].unique()
    article_ids = articles["article_id"].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    art_to_idx = {a: i for i, a in enumerate(article_ids)}

    valid = interactions[interactions["article_id"].isin(art_to_idx)]
    rows = valid["user_id"].map(user_to_idx).values
    cols = valid["article_id"].map(art_to_idx).values
    mat = csr_matrix((np.ones(len(valid)), (rows, cols)), shape=(len(user_ids), len(article_ids)))

    n_components = min(32, min(mat.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(mat)
    item_factors = svd.components_  # shape: (n_components, n_articles)

    cf_model = {
        "user_to_idx": user_to_idx,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "article_ids": article_ids,
    }

    # C. Subscription Propensity: Logistic Regression
    rng = np.random.default_rng(42)
    n_samples = 400
    synth = pd.DataFrame(
        {
            "articles_read_30d": rng.integers(0, 25, n_samples),
            "sessions_7d": rng.integers(1, 8, n_samples),
            "sport_diversity": rng.integers(1, 6, n_samples),
            "avg_read_time_min": rng.uniform(1.0, 10.0, n_samples),
        }
    )
    logit = (
        -4.2
        + 0.16 * synth["articles_read_30d"]
        + 0.35 * synth["sessions_7d"]
        + 0.25 * synth["sport_diversity"]
        + 0.22 * synth["avg_read_time_min"]
    )
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    propensity_model = make_pipeline(StandardScaler(), LogisticRegression())
    propensity_model.fit(synth, y)

    return tfidf_matrix, cf_model, propensity_model


def get_recommendations(
    user_id: str,
    model_choice: str,
    recency_weight: float,
    top_k: int,
    articles: pd.DataFrame,
    interactions: pd.DataFrame,
    tfidf_matrix,
    cf_scores_df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    read_ids = set(interactions.loc[interactions["user_id"] == user_id, "article_id"])
    is_cold_start = len(read_ids) == 0

    df = articles.copy()

    if is_cold_start:
        # Cold-Start Fallback: popularity (click counts) + freshness
        popularity = interactions["article_id"].value_counts()
        df["raw_score"] = df["article_id"].map(popularity).fillna(0.5)
        df["raw_score"] = df["raw_score"] / df["raw_score"].max()
    elif model_choice == "Content-Based (TF-IDF)":
        read_idx = df.index[df["article_id"].isin(read_ids)].tolist()
        user_profile = np.asarray(tfidf_matrix[read_idx].mean(axis=0))
        df["raw_score"] = cosine_similarity(user_profile, tfidf_matrix).flatten()
    else:
        u_idx = cf_scores_df["user_to_idx"][user_id]
        raw = cf_scores_df["user_factors"][u_idx] @ cf_scores_df["item_factors"]
        score_map = pd.Series(raw, index=cf_scores_df["article_ids"])
        vals = df["article_id"].map(score_map).fillna(0.0).values
        min_v, max_v = vals.min(), vals.max()
        df["raw_score"] = (vals - min_v) / (max_v - min_v + 1e-9)

    # Exponential recency decay: exp(-hours_ago / 24)
    df["freshness"] = np.exp(-df["hours_ago"] / 24.0)
    df["final_score"] = (
        (1.0 - recency_weight) * df["raw_score"] + recency_weight * df["freshness"]
    )

    # Exclude already-read articles
    recs = (
        df[~df["article_id"].isin(read_ids)]
        .sort_values("final_score", ascending=False)
        .head(top_k)
    )
    return recs, is_cold_start


# ---------------------------------------------------------------------------
# 3. UI LAYOUT
# ---------------------------------------------------------------------------
articles_df, interactions_df, users_df = load_data()
tfidf_mat, cf_df, prop_model = fit_models(articles_df, interactions_df)

with st.sidebar:
    st.header("Demo Controls")

    persona_map = {
        f"{row.user_id} — {row.persona} ({row.articles_read_30d} reads)": row.user_id
        for row in users_df.itertuples()
    }
    selected_label = st.selectbox("Select User Persona", list(persona_map.keys()))
    selected_user_id = persona_map[selected_label]

    model_choice = st.radio(
        "Recommendation Engine",
        ["Content-Based (TF-IDF)", "Collaborative Filtering (SVD)"],
        help="Compare a fast, interpretable baseline against latent-factor collaborative filtering.",
    )

    recency_weight = st.slider(
        "Recency Decay Weight",
        min_value=0.0,
        max_value=0.8,
        value=0.25,
        step=0.05,
        help="Higher values penalize older stories so breaking sports news ranks higher.",
    )

    top_k = st.slider("Articles to Recommend", min_value=3, max_value=8, value=5)

    st.divider()
    st.caption(
        "Built to demonstrate model trade-offs, cold-start handling, recency decay, "
        "and subscriber conversion scoring."
    )

# --- Main Header ---
user_row = users_df[users_df["user_id"] == selected_user_id].iloc[0]
feature_cols = [
    "articles_read_30d",
    "sessions_7d",
    "sport_diversity",
    "avg_read_time_min",
]
propensity_prob = float(prop_model.predict_proba(user_row[feature_cols].to_frame().T)[0, 1])
propensity_pct = propensity_prob * 100

if propensity_pct >= 70:
    tier, delta_color = "High Intent — Trigger Paywall Offer", "normal"
elif propensity_pct >= 35:
    tier, delta_color = "Medium Intent — Nurture via Newsletter", "off"
else:
    tier, delta_color = "Low Intent — Focus on Engagement", "inverse"

st.title("Sports News Recommendation System")
st.caption(f"Active Profile: **{user_row['persona']}** (`{selected_user_id}`)  |  Engine: **{model_choice}**")

# --- Top KPI Row ---
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Subscription Propensity",
    f"{propensity_pct:.1f}%",
    delta=tier.split(" — ")[0],
    delta_color=delta_color,
    border=True,
)
k2.metric("Articles Read (30d)", int(user_row["articles_read_30d"]), border=True)
k3.metric("Sessions (7d)", int(user_row["sessions_7d"]), border=True)
k4.metric("Leagues Followed", int(user_row["sport_diversity"]), border=True)

st.divider()

# --- Main Two-Column Content ---
feed_col, context_col = st.columns([1.6, 1.0], gap="large")

with feed_col:
    st.subheader("Recommended for You")
    recs, is_cold_start = get_recommendations(
        user_id=selected_user_id,
        model_choice=model_choice,
        recency_weight=recency_weight,
        top_k=top_k,
        articles=articles_df,
        interactions=interactions_df,
        tfidf_matrix=tfidf_mat,
        cf_scores_df=cf_df,
    )

    if is_cold_start:
        st.info(
            "**Cold-Start Fallback Active:** This visitor has no click history. "
            "Falling back to site-wide trending sports stories weighted by recency."
        )

    for rank, row in enumerate(recs.itertuples(), start=1):
        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            top_left.markdown(f"**#{rank} · {row.subcategory}** · _{row.hours_ago}h ago_")
            top_right.markdown(f"Match: **`{row.final_score:.2f}`**")
            st.markdown(f"#### {row.title}")
            st.write(row.abstract)
            st.caption(
                f"Model Score: {row.raw_score:.2f} | Freshness Factor: {row.freshness:.2f}"
            )

with context_col:
    st.subheader("Recently Read History")
    read_ids = interactions_df.loc[
        interactions_df["user_id"] == selected_user_id, "article_id"
    ]
    history = articles_df[articles_df["article_id"].isin(read_ids)]

    if history.empty:
        st.warning("No reading history yet for this user.")
    else:
        with st.container(height=320, border=True):
            for row in history.head(10).itertuples():
                st.markdown(f"**{row.subcategory}** — {row.title}")
                st.divider()

        if len(history) > 10:
            st.caption(f"Showing 10 of {len(history)} articles read.")

    st.subheader("Propensity Driver Breakdown")
    with st.container(border=True):
        st.progress(propensity_prob, text=f"Predicted Conversion Likelihood: {propensity_pct:.1f}%")
        st.markdown(f"**Recommended Action:** {tier}")

        # Show standardized feature contributions from the logistic regression model
        scaler = prop_model.named_steps["standardscaler"]
        clf = prop_model.named_steps["logisticregression"]
        scaled_vals = scaler.transform(user_row[feature_cols].to_frame().T)[0]
        contributions = pd.DataFrame(
            {
                "Feature": ["30d Reads", "7d Sessions", "League Diversity", "Avg Read Time"],
                "Impact on Log-Odds": scaled_vals * clf.coef_[0],
            }
        ).set_index("Feature")
        st.bar_chart(contributions, horizontal=True)