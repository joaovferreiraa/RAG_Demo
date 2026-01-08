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

def main():
    """Main function to run the RAG query system with source citations."""
    
    # Get user question
    question = input("Enter your question: ")
    
    # Load the vector database
    print("\n🔍 Searching for relevant information...")
    embedding_function = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    db = FAISS.load_local(
        config.DB_PATH,
        embedding_function,
        allow_dangerous_deserialization=True
    )
    
    # Search for relevant chunks using MMR for diversity
    results = db.max_marginal_relevance_search(question, k=7, fetch_k=20)
    
    # Build context with source labels
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
    
    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    
    # Generate answer with LLM
    print("\n💬 Generating answer...\n")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt)
    
    # Display answer
    print("=" * 80)
    print(response.content)
    print("=" * 80)
    
    # Extract cited source numbers from response (simple regex)
    cited = set(int(n) for n in re.findall(r'\[Source\s+(\d+)', response.content))
    
    # Display only cited sources
    if cited:
        print(f"\n📚 Source References:")
        print("-" * 80)
        for source_id in sorted(cited):
            meta = source_mapping[source_id]
            print(f"[Source {source_id}]: {meta['source_file']} - Page {meta['page']}")
        print("-" * 80)

if __name__ == "__main__":
    main()