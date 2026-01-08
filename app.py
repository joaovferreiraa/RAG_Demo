import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from app_config import config
import re

load_dotenv()

PROMPT_TEMPLATE = """
Answer the user's question using ONLY the information provided below.

Each piece of context is labeled with a source ID in square brackets [Source X].

Document context:
{context}

User's question: {question}

Instructions:
- Answer in English
- Be concise and objective
- ALWAYS cite your sources using [Source X] format after each claim
- Cite specific data when available (percentages, numbers, etc.)
- If you cannot find the answer, say "I couldn't find this information in the document."
- You can cite multiple sources for the same statement: [Source 1, 3]

Example of good citation:
"In Brazil, most data professionals live in the Southeast region (60.5%) and South region (20.3%) [Source 2]."

Answer:
"""

@st.cache_resource
def load_vector_db():
    """Load the FAISS vector database (cached to avoid reloading)."""
    embedding_function = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    db = FAISS.load_local(
        config.DB_PATH,
        embedding_function,
        allow_dangerous_deserialization=True
    )
    return db

def query_rag(question: str, db):
    """Query the RAG system and return the answer with sources."""
    results = db.max_marginal_relevance_search(question, k=7, fetch_k=20)

    context_with_sources = []
    source_mapping = {}

    for i, doc in enumerate(results, 1):
        source_label = f"[Source {i}]"
        context_with_sources.append(f"{source_label}\n{doc.page_content}")

        source_mapping[i] = {
            'page': doc.metadata.get('page', '?'),
            'source_file': doc.metadata.get('source_file', 'Unknown')
        }

    context = "\n\n---\n\n".join(context_with_sources)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt)

    cited = set(int(n) for n in re.findall(r'\[Source\s+(\d+)', response.content))

    return response.content, source_mapping, cited

def main():
    st.set_page_config(
        page_title="RAG Q&A System",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Question Answering System")
    st.markdown("Ask questions about the State of Data Brazil 2024-2025 (by Bain & Company and DataHackers) and get AI-powered answers!")

    st.divider()

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This is a **Retrieval-Augmented Generation (RAG)** system that:

        1. 🔍 Searches for relevant information in State of Data Brazil 2024-2025
        2. 🧠 Uses AI to generate accurate answers
        3. 📚 Cites sources for transparency

        **Technology Stack:**
        - LangChain
        - OpenAI GPT-4o-mini
        - FAISS Vector Database
        - Streamlit
        """)

        st.divider()

        st.markdown("**Configuration:**")
        st.text(f"Embedding Model: {config.EMBEDDING_MODEL}")
        st.text(f"Database: {config.DB_PATH}")

    try:
        with st.spinner("Loading vector database..."):
            db = load_vector_db()
        st.success("✅ Database loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading database: {str(e)}")
        st.info("Make sure you've run `create_db.py` first to create the vector database.")
        st.stop()

    question = st.text_input(
        "❓ Enter your question:",
        placeholder="e.g., What are the best insights from the State of Data Brazil 2025?",
        help="Type your question about the documents in the database"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()

    if search_button and question:
        with st.spinner("🔍 Searching for relevant information..."):
            try:
                answer, source_mapping, cited_sources = query_rag(question, db)

                st.divider()

                st.subheader("💬 Answer:")
                st.markdown(answer)

                if cited_sources:
                    st.divider()
                    st.subheader("📚 Source References:")

                    for source_id in sorted(cited_sources):
                        meta = source_mapping[source_id]
                        with st.expander(f"📄 Source {source_id}: Page {meta['page']}"):
                            st.text(f"File: {meta['source_file']}")
                            st.text(f"Page: {meta['page']}")
                else:
                    st.info("No sources were cited in the answer.")

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                st.exception(e)

    elif search_button and not question:
        st.warning("⚠️ Please enter a question first!")

    st.divider()

    with st.expander("💡 Example Questions"):
        st.markdown("""
            You may explore questions such as:
            - What are the best insights of State of Data Brazil 2024-2025?
            - Do Brazilian data professionals prefer remote work?
            - What percentage of data professionals use Python?
            - What percentage of data professionals are female?
        """)

if __name__ == "__main__":
    main()
