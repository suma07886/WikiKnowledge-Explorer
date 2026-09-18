
import streamlit as st
import pandas as pd
import re
from urllib.parse import unquote

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WikiKnowledge Explorer",
    page_icon="📚",
    layout="wide"
)

# ============================================================
# DATASET
# ============================================================

DATA_PATH = "data/enwiki_namespace_0_00000.parquet"


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


try:
    df = load_data()
except Exception as e:
    st.error("❌ Could not load the dataset.")
    st.code(str(e))
    st.stop()

# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize(text):

    if text is None:
        return ""

    try:
        text = str(text)
    except Exception:
        return ""

    text = unquote(text)
    text = text.replace("_", " ")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# SAFE VALUE CHECK
# ============================================================

def is_missing(value):

    if value is None:
        return True

    if isinstance(value, (list, tuple, dict)):
        return False

    try:

        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        return False

    except Exception:
        return False


# ============================================================
# ARTICLE LOOKUP
# ============================================================

name_lookup = {}

for name in df["name"].dropna():

    original_name = str(name)
    normalized_name = normalize(original_name)

    if normalized_name:
        name_lookup[normalized_name] = original_name


# ============================================================
# SEARCH ARTICLE
# ============================================================

def search_article(query):

    query = normalize(query)

    if not query:
        return None

    # EXACT MATCH

    if query in name_lookup:

        real_name = name_lookup[query]

        result = df[
            df["name"].astype(str) == real_name
        ]

        if not result.empty:
            return result.iloc[0]

    # PARTIAL MATCH

    for name in df["name"].dropna():

        name_text = str(name)

        if query in normalize(name_text):

            result = df[
                df["name"].astype(str) == name_text
            ]

            if not result.empty:
                return result.iloc[0]

    # WORD MATCH

    query_words = query.split()

    for name in df["name"].dropna():

        name_text = str(name)
        normalized_name = normalize(name_text)

        if all(
            word in normalized_name
            for word in query_words
        ):

            result = df[
                df["name"].astype(str) == name_text
            ]

            if not result.empty:
                return result.iloc[0]

    return None



# ============================================================
# ROBUST RELATIONSHIP EXTRACTION
# ============================================================

def extract_wikipedia_links(value):

    links = []

    if is_missing(value):
        return links

    try:
        text = str(value)
    except Exception:
        return links

    # Support Wikipedia URLs with different formats
    patterns = [
        r'(?:https?:)?//en\.wikipedia\.org/wiki/([^\\s"\'<>]+)',
        r'https?://[a-z]+\.wikipedia\.org/wiki/([^\\s"\'<>]+)'
    ]

    for pattern in patterns:

        matches = re.findall(pattern, text)

        for match in matches:

            title = match.split("#")[0]

            title = title.rstrip(
                ".,;:)]}>\"'"
            )

            title = unquote(title)
            title = title.replace("_", " ")

            if title:
                links.append(title)

    return list(dict.fromkeys(links))


# ============================================================
# FIND RELATED ARTICLES
# ============================================================

def get_relationship_details(article):

    relationships = []

    current_article = normalize(
        str(article["name"])
    )

    # Search all fields
    for field in article.index:

        if field in [
            "abstract",
            "description",
            "name"
        ]:
            continue

        value = article[field]

        if is_missing(value):
            continue

        # Convert nested values to searchable text
        try:
            text = str(value)
        except Exception:
            continue

        # Extract URLs
        links = extract_wikipedia_links(value)

        for link in links:

            normalized_title = normalize(link)

            if normalized_title == current_article:
                continue

            if normalized_title in name_lookup:

                real_name = name_lookup[
                    normalized_title
                ]

                item = {
                    "article": real_name,
                    "source": field
                }

                if item not in relationships:
                    relationships.append(item)

        # Fallback:
        # Check whether dataset article names
        # appear in the structured field text.

        text_normalized = normalize(text)

        for normalized_name, real_name in name_lookup.items():

            if normalized_name == current_article:
                continue

            # Only match reasonably sized article names
            if len(normalized_name) < 3:
                continue
            if real_name.strip().isdigit():
                 continue
            if normalized_name in text_normalized:

                item = {
                    "article": real_name,
                    "source": field
                }

                if item not in relationships:
                    relationships.append(item)

    return relationships

# ============================================================
# FORMAT LANGUAGE
# ============================================================

def format_language(language):

    if is_missing(language):
        return "English (en)"

    # Dictionary format:
    # {'identifier': 'en'}

    if isinstance(language, dict):

        identifier = language.get(
            "identifier",
            "en"
        )

        language_names = {
            "en": "English",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "hi": "Hindi",
            "az": "Azerbaijani"
        }

        language_name = language_names.get(
            identifier,
            identifier
        )

        return f"{language_name} ({identifier})"

    return str(language)


# ============================================================
# ARTICLE CLASSIFICATION
# ============================================================

def classify_article(name, abstract):

    text = (
        str(name)
        + " "
        + str(abstract)
    ).lower()

    categories = []

    person_words = [
        "born",
        "died",
        "politician",
        "actor",
        "actress",
        "scientist",
        "writer",
        "artist",
        "professor",
        "king",
        "queen",
        "president",
        "author"
    ]

    place_words = [
        "city",
        "town",
        "village",
        "country",
        "state",
        "province",
        "district",
        "located",
        "capital",
        "region"
    ]

    organization_words = [
        "organization",
        "organisation",
        "company",
        "university",
        "institute",
        "agency",
        "association",
        "government",
        "corporation"
    ]

    topic_words = [
        "theory",
        "technology",
        "science",
        "history",
        "method",
        "system",
        "process",
        "concept"
    ]

    if any(
        word in text
        for word in person_words
    ):
        categories.append("People")

    if any(
        word in text
        for word in place_words
    ):
        categories.append("Places")

    if any(
        word in text
        for word in organization_words
    ):
        categories.append("Organizations")

    if any(
        word in text
        for word in topic_words
    ):
        categories.append("Topics")

    if not categories:
        categories.append("Topics")

    return categories


# ============================================================
# HEADER
# ============================================================

st.title("📚 WikiKnowledge Explorer")

st.markdown(
    """
### Explore Wikipedia Knowledge Through Dataset Connections

Search an article and discover related information
available in the Wikimedia Structured Wikipedia Dataset.
"""
)

st.divider()

# ============================================================
# SEARCH
# ============================================================

st.subheader("🔎 Search Article")

query = st.text_input(
    "Enter an article name",
    placeholder="Try: Khan gizi spring"
)

search_button = st.button(
    "🔍 Search",
    use_container_width=True
)

# ============================================================
# SEARCH ACTION
# ============================================================

if search_button:

    if not query.strip():

        st.warning(
            "⚠️ Please enter an article name."
        )

    else:

        article = search_article(query)

        if article is None:

            st.error(
                "❌ No article found in the current dataset shard."
            )

            st.info(
                "Try another article name or a partial name."
            )

        else:

            st.session_state["article"] = article


# ============================================================
# ARTICLE PAGE
# ============================================================

if "article" in st.session_state:
    if st.button("⬅️ Back to Search"):
        st.session_state.pop("article", None)
        st.rerun()

    article = st.session_state["article"]

    article_name = str(
        article["name"]
    )

    abstract = article.get(
        "abstract",
        ""
    )

    st.divider()

    st.success(
        f"✅ Article found: {article_name}"
    )

    st.header(
        f"📖 {article_name}"
    )

    # ========================================================
    # ABSTRACT
    # ========================================================

    if not is_missing(abstract):

        if str(abstract).strip():

            st.subheader(
                "📝 Description"
            )

            st.write(
                str(abstract)
            )

    # ========================================================
    # ARTICLE INFORMATION
    # ========================================================

    st.subheader(
        "ℹ️ Article Information"
    )

    col1, col2, col3 = st.columns(3)

    # Identifier

    with col1:

        identifier = article.get(
            "identifier",
            "Not available"
        )

        st.metric(
            "Identifier",
            str(identifier)
        )

    # Language

    with col2:

        language = article.get(
            "in_language",
            {"identifier": "en"}
        )

        st.metric(
            "Language",
            format_language(language)
        )

    # Dataset size

    with col3:

        st.metric(
            "Articles in Shard",
            f"{len(df):,}"
        )

    # ========================================================
    # MAIN ENTITY
    # ========================================================

    if "main_entity" in article.index:

        main_entity = article["main_entity"]

        if not is_missing(main_entity):

            st.subheader(
                "🔗 Main Entity"
            )

            st.code(
                str(main_entity),
                language="text"
            )

    # ========================================================
    # RELATED INFORMATION
    # ========================================================

    st.divider()

    st.header(
        "🔗 Related Information"
    )

    st.write(
        """
Related articles are discovered by searching Wikipedia
URLs across the available dataset fields.

Only articles that exist in the current dataset shard
are displayed.
"""
    )

    relationships = get_relationship_details(
        article
    )
    if relationships is None:
          relationships = []

    related_articles = [
        item["article"]
        for item in relationships
    ]

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👤 People",
            "—"
        )

    with col2:

        st.metric(
            "🏢 Organizations",
            "—"
        )

    with col3:

        st.metric(
            "📍 Places",
            "—"
        )

    with col4:

        st.metric(
            "🔗 Connections",
            len(related_articles)
        )

    # ========================================================
    # CONNECTIONS
    # ========================================================

    if relationships:

        st.subheader(
            "🌐 Dataset Connections"
        )

        for index, item in enumerate(
            relationships
        ):

            related_name = item[
                "article"
            ]

            source = item[
                "source"
            ]

            col1, col2, col3 = st.columns(
                [4, 2, 1]
            )

            with col1:

                st.write(
                    f"🔗 **{related_name}**"
                )

            with col2:

                st.caption(
                    f"📄 Source: {source}"
                )

            with col3:

                if st.button(
                    "Explore",
                    key=f"explore_{index}"
                ):

                    new_article = search_article(
                        related_name
                    )

                    if new_article is not None:

                        st.session_state[
                            "article"
                        ] = new_article

                        st.rerun()

    else:

        st.info(
            """
No directly connected articles were found
inside this dataset shard.

The article may contain links to Wikipedia pages
that are not included in the downloaded shard,
or its stored fields may not contain extractable
Wikipedia article URLs.
"""
        )

    # ========================================================
    # CATEGORIES
    # ========================================================

    st.divider()

    st.subheader(
        "🏷️ Article Categories"
    )

    categories = classify_article(
        article_name,
        abstract
    )

    category_cols = st.columns(
        len(categories)
    )

    for i, category in enumerate(
        categories
    ):

        with category_cols[i]:

            st.success(
                f"📌 {category}"
            )

    # ========================================================
    # DATASET FIELDS
    # ========================================================

    st.divider()

    with st.expander(
        "🔍 View Dataset Fields"
    ):

        st.write(
            "Fields available for this article:"
        )

        for field in article.index:

            st.write(
                f"• `{field}`"
            )

else:

    st.info(
        "👆 Enter an article name above to start exploring."
    )

    st.divider()

    st.subheader(
        "🚀 How It Works"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
### 1️⃣ Search

Search for an article available
in the Wikimedia dataset.
"""
        )

    with col2:

        st.markdown(
            """
### 2️⃣ Discover

Retrieve article information and
relationships from the dataset.
"""
        )

    with col3:

        st.markdown(
            """
### 3️⃣ Explore

Explore connected articles and
navigate between them.
"""
        )

    st.divider()

    st.subheader(
        "📊 Dataset"
    )

    st.write(
        f"""
**Wikimedia Structured Wikipedia Dataset**

Currently loaded:

**{len(df):,} articles**

This application uses the downloaded dataset
shard and does not require the complete dataset.
"""
    )

    st.divider()

    st.subheader(
        "✨ Features"
    )

    st.markdown(
        """
- 🔎 Article search
- 📝 Article information
- 🔗 Dataset-based relationships
- 🌐 Connected article exploration
- 🏷️ Article categorization
- 📊 Dataset statistics
"""
    )