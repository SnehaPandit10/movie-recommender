import streamlit as st
import pickle
import requests
from concurrent.futures import ThreadPoolExecutor
import streamlit.components.v1 as components

st.set_page_config(page_title="MovieFlix", layout="wide")

API_KEY = st.secrets["TMDB_API_KEY"]
IMG = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER = "https://placehold.co/342x513?text=No+Poster"
BATCH = 10     # movies loaded per scroll
PER_ROW = 5    # posters per row

st.markdown("""
<style>
.stApp { background-color: #141414; color: white; }
h1, h2, h3, p, label, .stCaption { color: white !important; }
.brand { color: #E50914; font-size: 2.6rem; font-weight: 900; letter-spacing: 2px; }
img { border-radius: 6px; transition: transform .2s; }
img:hover { transform: scale(1.05); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    with open('movies.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('similarity.pkl', 'rb') as f:
        sim = pickle.load(f)
    return movies, sim

movies_list, similarity = load_data()

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": API_KEY}, timeout=10)
        r.raise_for_status()
        path = r.json().get("poster_path")
        if path:
            return IMG + path
    except (requests.RequestException, ValueError):
        pass
    return PLACEHOLDER

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trending():
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/trending/movie/week",
            params={"api_key": API_KEY}, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        return [
            (m.get("title", "Untitled"), IMG + m["poster_path"])
            for m in results
            if isinstance(m, dict) and m.get("poster_path")
        ]
    except (requests.RequestException, ValueError):
        return []

def get_ranked(movie):
    """All movies sorted by similarity (no limit), excluding the movie itself."""
    i = movies_list[movies_list['title'] == movie].index[0]
    ranked = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)[1:]
    return [idx for idx, _ in ranked]

def show_grid(items):
    """items = list of (title, poster_url); draws rows of PER_ROW."""
    for start in range(0, len(items), PER_ROW):
        cols = st.columns(PER_ROW)
        for col, (title, poster) in zip(cols, items[start:start + PER_ROW]):
            with col:
                st.image(poster)
                st.caption(title)

st.markdown('<div class="brand">MOVIEFLIX</div>', unsafe_allow_html=True)
selected = st.selectbox(
    "Search", movies_list['title'].values, index=None,
    placeholder="🔍 Search for a movie...",
    label_visibility="collapsed")

if selected is None:
    st.subheader("Trending this week")
    trending = fetch_trending()
    if trending:
        show_grid(trending[:15])
    else:
        st.info("Search for a movie above to get recommendations.")
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
    st.stop()

if st.session_state.get("query") != selected:
    st.session_state.query = selected
    st.session_state.ranked = get_ranked(selected)
    st.session_state.shown = BATCH

ranked = st.session_state.ranked
shown = st.session_state.shown

st.subheader(f"Because you watched {selected}")

batch = ranked[:shown]
with ThreadPoolExecutor(max_workers=10) as ex:
    posters = list(ex.map(
        lambda idx: fetch_poster(movies_list.iloc[idx].movie_id), batch))
titles = [movies_list.iloc[idx].title for idx in batch]
show_grid(list(zip(titles, posters)))

if shown < len(ranked):
    if st.button("Load more", key="load_more", use_container_width=True):
        st.session_state.shown += BATCH
        st.rerun()

    components.html("""
    <script>
    const doc = window.parent.document;
    const btn = [...doc.querySelectorAll('button')]
                  .find(b => b.innerText.includes('Load more'));
    if (btn) {
      new IntersectionObserver((e, o) => {
        if (e[0].isIntersecting) { o.disconnect(); btn.click(); }
      }, {rootMargin: '300px'}).observe(btn);
    }
    </script>""", height=0)
else:
    st.caption("You've reached the end 🎬")

st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")