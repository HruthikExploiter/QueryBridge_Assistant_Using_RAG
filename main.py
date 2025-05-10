import streamlit as st
from rag import process_urls, generate_answer, reset_vector_store

st.set_page_config(page_title="QueryBridge", layout="wide")

# Custom Title
st.markdown(
    "<h3 style='text-align: center; margin-bottom: 30px;'>🔍 QueryBridge: Smarter Answers with RAG </h3>",
    unsafe_allow_html=True
)

# --- Session state to track if data is loaded ---
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# --- Layout with two columns ---
col1, col2 = st.columns(2)

# Step 1 - URL Loader (Left Side)
with col1:
    st.markdown("<h4>Step 1️⃣: Load URLs</h4>", unsafe_allow_html=True)

    urls_input = st.text_area(
        "",
        placeholder="Paste one or more URLs here (one per line)...",
        height=70
    )

    # Buttons side by side
    load_col, reset_col = st.columns([1, 1])
    with load_col:
        load_clicked = st.button("📥 Load and Index URLs")
    with reset_col:
        reset_clicked = st.button("🔄 Reset Vector DB")

    if reset_clicked:
        with st.spinner("Resetting vector database..."):
            reset_vector_store()
            st.session_state.data_loaded = False
        st.success("✅ Vector database has been reset.")

    if load_clicked:
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.strip().splitlines() if url.strip()]
            with st.spinner("🔄 Processing URLs..."):
                for update in process_urls(urls):
                    st.toast(update, icon="🔄")
            st.session_state.data_loaded = True
            st.success("✅ URLs successfully indexed!")
        else:
            st.error("❌ Please enter at least one valid URL.")

# Step 2 - Q&A (Right Side)
with col2:
    st.markdown("<h4>Step 2️⃣: Ask a Question</h4>", unsafe_allow_html=True)

    query = st.text_input("Ask something based on the loaded content:")
    ask_button = st.button("💬 Get Answer")

    if ask_button:
        if not st.session_state.data_loaded:
            st.error("❌ Please load and index URLs before asking a question.")
        elif not query.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("🤖 Thinking..."):
                answer, sources = generate_answer(query)
            st.markdown("<h5 style='background-color: #173c32; padding: 8px; border-radius: 5px;'>🧠 Answer:</h5>", unsafe_allow_html=True)
            st.write(answer if answer else "No answer found.")

            st.markdown("<h5 style='background-color: #1a2c49; padding: 8px; border-radius: 5px;'>📚 Sources:</h5>", unsafe_allow_html=True)
            st.write(sources if sources else "No sources available.")
