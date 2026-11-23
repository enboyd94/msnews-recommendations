import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
def load_articles():
    con = duckdb.connect("data/mind.duckdb", read_only=True)
    articles = con.execute("SELECT * FROM sports_news").df()
    interactions = con.execute("SELECT * FROM sports_interactions").df()
    users = con.execute("SELECT * FROM user_features").df()
    con.close()
    return articles, interactions, users
# def load_articles() -> pd.DataFrame:
#     """Mock sports catalog matching the MIND news.tsv schema + age in hours."""
#     articles = [
#         ("A01", "NFL", "Chiefs' Red Zone Playcalling Is Evolving Ahead of the Playoffs",
#          "Film breakdown of motion rates, tight end seams, and Andy Reid's late-season adjustments.", 3),
#         ("A02", "NFL", "Quarterback Pressure Rates: Which Offensive Lines Are Cracking?",
#          "Using tracking data to separate quarterback sack avoidance from true pass-blocking collapse.", 14),
#         ("A03", "NFL", "2026 NFL Draft Big Board: Top Edge Rushers Rising After Combine",
#          "Scouting reports, athletic testing percentiles, and scheme fits for first-round defenders.", 36),
#         ("A04", "NBA", "Why Small-Ball Switch Defenses Are Back in the Western Conference",
#          "Lineup data shows versatile wings are neutralizing drop coverage in clutch minutes.", 5),
#         ("A05", "NBA", "Trade Deadline Winners: How Rim Protection Shifted Title Odds",
#          "Expected points saved at the rim and updated playoff projection models post-deadline.", 22),
#         ("A06", "NBA", "Inside the 3-Point Revolution: Corner Shooting vs. Pull-Up Volume",
#          "Shot-quality metrics reveal which contenders rely on sustainable spacing.", 48),
#         ("A07", "Premier League", "Arsenal's Set-Piece Routine: Anatomy of a Near-Post Overload",
#          "Tactical analysis of blocking patterns, delivery height, and expected goals from corners.", 2),
#         ("A08", "Premier League", "Summer Transfer Window: Midfield Press-Resistance Targets",
#          "Data scouting progressive carries and ball retention under high-intensity pressure.", 18),
#         ("A09", "Premier League", "Title Race Run-In: Fixture Difficulty and Squad Fatigue Models",
#          "Minutes-load projections for the top three clubs across domestic and European competitions.", 30),
#         ("A10", "MLB", "Pitch Design Lab: How Sweeper Sliders Changed Right-on-Right Matchups",
#          "Horizontal break, seam-shifted wake, and whiff rate trends across starting rotations.", 8),
#         ("A11", "MLB", "Bullpen Leverage Management: Are Managers Pulling Starters Too Early?",
#          "Third-time-through-the-order penalties weighed against high-leverage reliever fatigue.", 40),
#         ("A12", "NHL", "Expected Goals on the Power Play: The Bumper Position Explained",
#          "How elite units manipulate penalty-kill triangles to open royal-road passes.", 10),
#         ("A13", "NFL", "Salary Cap Gymnastics: Void Years and Future Dead-Money Risks",
#          "Front office cap strategies explained for contenders pushing their window open.", 6),
#         ("A14", "NBA", "Rookie Development Curves: Who Is Outperforming Draft Position?",
#          "On/off net rating and usage efficiency for first-year rotation players.", 12),
#         ("A15", "Premier League", "Goalkeeper Sweeper Metrics: Quantifying Box Command",
#          "Measuring defensive actions outside the penalty area and prevented through-balls.", 26),
#     ]
#     return pd.DataFrame(
#         articles,
#         columns=["article_id", "subcategory", "title", "abstract", "hours_ago"],
#     )


@st.cache_data
def load_users_and_interactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mock user profiles (4 featured personas + 6 background users for SVD)."""
    users = pd.DataFrame(
        [
            # Featured personas shown in the sidebar
            ("U101", "Casual NFL Reader", 3, 2, 1, 3.2, True),
            ("U204", "Die-Hard Multi-Sport Fan", 18, 6, 4, 8.5, True),
            ("U309", "Premier League Tactician", 11, 5, 1, 7.1, True),
            ("U412", "Cold-Start New Visitor", 0, 1, 0, 1.0, True),
            # Background users to populate the collaborative filtering matrix
            ("U501", "NFL & NBA Follower", 12, 4, 2, 6.0, False),
            ("U502", "MLB Analytics Reader", 9, 4, 2, 5.8, False),
            ("U503", "Soccer & NHL Fan", 14, 5, 2, 7.4, False),
            ("U504", "Weekend Skimmer", 2, 1, 1, 2.1, False),
            ("U505", "All-Sports Subscriber", 22, 7, 5, 9.2, False),
            ("U506", "NBA Draft Junkie", 10, 4, 1, 6.5, False),
        ],
        columns=[
            "user_id",
            "persona",
            "articles_read_30d",
            "sessions_7d",
            "sport_diversity",
            "avg_read_time_min",
            "is_featured",
        ],
    )

    # Implicit click interactions (user_id, article_id)
    clicks = [
        # Featured personas
        ("U101", "A01"), ("U101", "A02"),
        ("U204", "A01"), ("U204", "A04"), ("U204", "A07"), ("U204", "A10"), ("U204", "A12"),
        ("U309", "A07"), ("U309", "A08"), ("U309", "A09"),
        # U412 intentionally has 0 clicks (Cold-Start)
        # Background users
        ("U501", "A01"), ("U501", "A02"), ("U501", "A03"), ("U501", "A04"), ("U501", "A13"),
        ("U502", "A10"), ("U502", "A11"), ("U502", "A02"), ("U502", "A13"),
        ("U503", "A07"), ("U503", "A08"), ("U503", "A12"), ("U503", "A15"),
        ("U504", "A04"), ("U504", "A06"),
        ("U505", "A01"), ("U505", "A04"), ("U505", "A05"), ("U505", "A07"), ("U505", "A10"), ("U505", "A14"),
        ("U506", "A04"), ("U506", "A05"), ("U506", "A06"), ("U506", "A14"),
    ]
    interactions = pd.DataFrame(clicks, columns=["user_id", "article_id"])
    return users, interactions


# ---------------------------------------------------------------------------
# 2. MODELS (Content-Based, Collaborative Filtering, Subscription Propensity)
# ---------------------------------------------------------------------------
@st.cache_resource
def fit_models(articles: pd.DataFrame, interactions: pd.DataFrame):
    # A. Content-Based: TF-IDF over subcategory + title + abstract
    corpus = (
        articles["subcategory"] + " " + articles["title"] + " " + articles["abstract"]
    )
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(corpus)

    # B. Collaborative Filtering: User-Item Matrix + TruncatedSVD
    pivot = (
        interactions.assign(clicked=1)
        .pivot_table(index="user_id", columns="article_id", values="clicked", fill_value=0)
        .reindex(columns=articles["article_id"], fill_value=0)
    )
    svd = TruncatedSVD(n_components=3, random_state=42)
    user_factors = svd.fit_transform(pivot.values)
    reconstructed = user_factors @ svd.components_
    cf_scores_df = pd.DataFrame(
        reconstructed, index=pivot.index, columns=pivot.columns
    )

    # C. Subscription Propensity: Logistic Regression on synthetic historical cohort
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
    probs = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probs)

    propensity_model = make_pipeline(StandardScaler(), LogisticRegression())
    propensity_model.fit(synth, y)

    return tfidf_matrix, cf_scores_df, propensity_model


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
        # Collaborative Filtering (TruncatedSVD)
        raw = cf_scores_df.loc[user_id].reindex(df["article_id"]).values
        # Min-max normalize to [0, 1] for clean display and recency blending
        min_v, max_v = raw.min(), raw.max()
        df["raw_score"] = (raw - min_v) / (max_v - min_v + 1e-9)

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
articles_df = load_articles()
users_df, interactions_df = load_users_and_interactions()
tfidf_mat, cf_df, prop_model = fit_models(articles_df, interactions_df)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Demo Controls")

    featured = users_df[users_df["is_featured"]]
    persona_map = {
        f"{row.user_id} — {row.persona}": row.user_id for row in featured.itertuples()
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

st.title("Sports News Recommendation & Subscriber Propensity")
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
        for row in history.itertuples():
            with st.container(border=True):
                st.markdown(f"**{row.subcategory}** — {row.title}")

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