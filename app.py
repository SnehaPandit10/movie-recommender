import streamlit as st
import pickle
import requests
import html
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.set_page_config(page_title="MovieFlix", layout="wide")

# ---------- Constants (no trailing commas!) ----------
API_KEY = st.secrets["TMDB_API_KEY"]
IMG = "https://image.tmdb.org/t/p/w342"
IMG_BIG = "https://image.tmdb.org/t/p/w500"
IMG_FACE = "https://image.tmdb.org/t/p/w185"
PLACEHOLDER = "https://placehold.co/342x513?text=No+Poster"
NO_PHOTO = "https://placehold.co/185x278?text=No+Photo"
BATCH = 10        # movies loaded per scroll
CAST_LIMIT = 15   # cast members shown on details page

# ---------- Netflix-style dark theme + equal-size cards ----------
st.markdown("""
<style>
.stApp { background-color: #141414; color: white; }
h1, h2, h3, p, label, .stCaption { color: white !important; }
.brand { color: #E50914; font-size: 2.6rem; font-weight: 900; letter-spacing: 2px; }
.meta { color: #b3b3b3; font-size: 0.95rem; }
.tagline { color: #b3b3b3; font-style: italic; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 22px 16px; margin: 8px 0 20px 0; }
.card, .card:hover, .card:visited { display: block; text-decoration: none !important;
        color: white !important; }
.card .pic { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; display: block;
        border-radius: 8px; background: #222; transition: transform .2s; }
.card:hover .pic { transform: scale(1.04); }
.card .name { margin-top: 8px; font-size: .92rem; font-weight: 600; line-height: 1.25;
        height: 2.5em; overflow: hidden; display: -webkit-box;
        -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.card .sub { font-size: .8rem; color: #b3b3b3; line-height: 1.25; height: 2.5em;
        overflow: hidden; display: -webkit-box;
        -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_resource
def load_data():
    with open('movies.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('similarity.pkl', 'rb') as f:
        sim = pickle.load(f)
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(movies['tags'].fillna(''))
    titles_lower = movies['title'].str.lower().tolist()
    return movies, sim, tfidf, matrix, titles_lower

movies_list, similarity, tfidf, tfidf_matrix, titles_lower = load_data()

# ---------- TMDB calls ----------
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
            (m.get("title", "Untitled"), IMG + m["poster_path"], int(m["id"]))
            for m in results
            if isinstance(m, dict) and m.get("poster_path") and m.get("id")
        ]
    except (requests.RequestException, ValueError):
        return []

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_details(movie_id):
    """Movie info + credits (cast & crew) in a single API call."""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": API_KEY, "append_to_response": "credits"},
            timeout=10)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None

# ---------- Ranking ----------
def ranked_by_movie(i):
    ranked = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)
    return [i] + [int(idx) for idx, _ in ranked if idx != i]

ALIASES = {"scifi": "sciencefiction", "sci-fi": "sciencefiction",
           "romcom": "romance", "animated": "animation"}

def resolve_term(term):
    """Turn one search term into tokens that exist in the stemmed index."""
    vocab = tfidf.vocabulary_
    term = term.lower().strip()
    joined = re.sub(r"[^a-z0-9]", "", term)
    joined = ALIASES.get(joined, joined)

    # 1) whole phrase as one token: "tom hanks" -> tomhanks, "science fiction" -> sciencefict
    for cand in (ps.stem(joined), joined):
        if cand in vocab:
            return [cand]

    # 2) word by word
    found = []
    for w in re.findall(r"[a-z0-9]+", term):
        w = ALIASES.get(w, w)
        for cand in (ps.stem(w), w):
            if cand in vocab:
                found.append(cand)
                break
        else:
            # 3) partial name: "nolan" -> christophernolan, "hanks" -> tomhanks
            if len(w) >= 4:
                found += [t for t in vocab if w in t][:5]
    return found

def ranked_by_tags(query):
    tokens = []
    for part in query.split(","):
        if part.strip():
            tokens += resolve_term(part)
    tokens = list(dict.fromkeys(tokens))          # remove duplicates
    if not tokens:
        return []
    v = tfidf.transform([" ".join(tokens)])
    scores = cosine_similarity(v, tfidf_matrix).ravel()
    order = scores.argsort()[::-1]
    return [int(i) for i in order if scores[i] > 0]

def search(query):
    q = query.strip().lower()
    if q in titles_lower:
        return "movie", ranked_by_movie(titles_lower.index(q))
    title_hits = []
    if "," not in q:
        title_hits = [i for i, t in enumerate(titles_lower) if q in t]
    seen = set(title_hits)
    tag_hits = [i for i in ranked_by_tags(q) if i not in seen]
    return "search", title_hits + tag_hits

# ---------- Clickable card grid (pure HTML links) ----------
def href(movie_id=None, q=None, n=None):
    parts = []
    if q:
        parts.append("q=" + quote_plus(q))
    if movie_id:
        parts.append(f"m={int(movie_id)}")
    if n:
        parts.append(f"n={int(n)}")
    return "?" + "&".join(parts)

def grid_html(cards):
    """cards = list of (link, image_url, title, subtitle)."""
    out = ['<div class="grid">']
    for link, img, title, sub in cards:
        out.append(
            f'<a class="card" href="{html.escape(link)}" target="_self">'
            f'<img class="pic" loading="lazy" src="{html.escape(img)}" alt="">'
            f'<div class="name">{html.escape(title)}</div>'
            f'<div class="sub">{html.escape(sub)}</div>'
            f'</a>')
    out.append('</div>')
    return "".join(out)          # no newlines, so markdown can't break the HTML

# ---------- Details page ----------
def close_detail():
    if "m" in st.query_params:
        del st.query_params["m"]

def show_details(movie_id):
    st.button("← Back", on_click=close_detail)

    data = fetch_details(movie_id)
    if not data:
        st.error("Could not load details for this movie. Please try again.")
        return

    credits = data.get("credits", {})
    crew = credits.get("crew", [])
    cast = credits.get("cast", [])

    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    writers = []
    for c in crew:
        if c.get("job") in ("Writer", "Screenplay", "Story") and c["name"] not in writers:
            writers.append(c["name"])

    year = (data.get("release_date") or "")[:4]
    runtime = data.get("runtime")
    runtime_txt = f"{runtime // 60}h {runtime % 60}m" if runtime else "N/A"
    genres = ", ".join(g["name"] for g in data.get("genres", [])) or "N/A"
    rating = data.get("vote_average")

    left, right = st.columns([1, 2])
    with left:
        poster = data.get("poster_path")
        st.image(IMG_BIG + poster if poster else PLACEHOLDER)
    with right:
        title = data.get("title", "Untitled")
        st.title(f"{title} ({year})" if year else title)
        if data.get("tagline"):
            st.markdown(f'<div class="tagline">{html.escape(data["tagline"])}</div>',
                        unsafe_allow_html=True)
        meta = [f"⏱ {runtime_txt}", f"🎭 {genres}"]
        if rating:
            meta.append(f"⭐ {rating:.1f}/10")
        if data.get("release_date"):
            meta.append(f"📅 {data['release_date']}")
        st.markdown('<div class="meta">' + " &nbsp;•&nbsp; ".join(meta) + '</div>',
                    unsafe_allow_html=True)

        st.subheader("Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown(f"**Director:** {', '.join(directors) if directors else 'N/A'}")
        st.markdown(f"**Writer:** {', '.join(writers[:5]) if writers else 'N/A'}")

    st.subheader("Cast")
    if not cast:
        st.write("Cast information not available.")
    else:
        # clicking a cast member searches for that person
        cards = [
            (href(q=p["name"]),
             IMG_FACE + p["profile_path"] if p.get("profile_path") else NO_PHOTO,
             p["name"],
             f"as {p['character']}" if p.get("character") else "")
            for p in cast[:CAST_LIMIT]
        ]
        st.markdown(grid_html(cards), unsafe_allow_html=True)

# ---------- Header + single search box ----------
st.markdown('<div class="brand">MOVIEFLIX</div>', unsafe_allow_html=True)

q_param = st.query_params.get("q", "")
if "search_box" not in st.session_state:
    st.session_state.search_box = q_param        # restore search after a card click

query = st.text_input(
    "Search", key="search_box",
    placeholder="🔍 Search a movie, actor, director or genre...",
    label_visibility="collapsed").strip()

if query != q_param:                              # user typed a new search
    if query:
        st.query_params["q"] = query
    elif "q" in st.query_params:
        del st.query_params["q"]
    for k in ("m", "n"):
        if k in st.query_params:
            del st.query_params[k]

# ---------- Details page ----------
detail_id = st.query_params.get("m")
if detail_id:
    try:
        show_details(int(detail_id))
    except ValueError:
        st.error("Invalid movie link.")
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
    st.stop()

# ---------- Home page (nothing searched yet) ----------
if not query:
    st.subheader("Trending this week")
    trending = fetch_trending()
    if trending:
        cards = [(href(mid), poster, title, "") for title, poster, mid in trending[:15]]
        st.markdown(grid_html(cards), unsafe_allow_html=True)
    else:
        st.info("Search above to get recommendations.")
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
    st.stop()

# ---------- Results with infinite loading ----------
if st.session_state.get("query") != query:
    st.session_state.query = query
    kind, ranked = search(query)
    st.session_state.kind = kind
    st.session_state.ranked = ranked
    try:
        n = int(st.query_params.get("n", BATCH))   # restore scroll depth after Back
    except ValueError:
        n = BATCH
    st.session_state.shown = max(BATCH, min(n, 500))

kind = st.session_state.kind
ranked = st.session_state.ranked
shown = st.session_state.shown

if not ranked:
    st.warning(f"No results for “{query}”. Check the spelling or try another name.")
    st.stop()

if kind == "movie":
    st.subheader(f"Because you watched {movies_list.iloc[ranked[0]].title}")
else:
    st.subheader(f"Results for “{query}”")

batch = ranked[:shown]
ids = [int(movies_list.iloc[idx].movie_id) for idx in batch]
with ThreadPoolExecutor(max_workers=10) as ex:
    posters = list(ex.map(fetch_poster, ids))
titles = [movies_list.iloc[idx].title for idx in batch]

cards = [(href(mid, q=query, n=shown), poster, title, "")
         for mid, poster, title in zip(ids, posters, titles)]
st.markdown(grid_html(cards), unsafe_allow_html=True)

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