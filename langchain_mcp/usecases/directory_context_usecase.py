from langchain_mcp.vector_store.vector import DocumentVectorizer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model="qwen2.5-coder:7b", num_ctx=32767)


vectorizer = DocumentVectorizer(
    embedding_model_name="mxbai-embed-large:latest",
    sqlite_path="xoibit.db",
    content_column="content",
    metadata_columns=["directory"],
    db_path="./xoibit_chroma",
    chunk_size=1000,
)

query = "How is the live stock status is being displayed in the app?"
similar_docs = vectorizer.similarity_search(query=query, k=20)

combined_docs = "\n\n".join(doc.page_content for doc in similar_docs)

template = """
You are an expert who reads documents with a question. 
Read the following combined document chunks and provide the answer 
in a detailed, clear and layman explanation. In no scenario you are allowed 
to come up with your own answer.
Document:
{document}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke(
    {
        "document": combined_docs,
        "question": query,
    }
)


print(result.strip())
