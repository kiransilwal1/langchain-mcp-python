from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM


from langchain_mcp.vector_store.vector import DocumentVectorizer

model = OllamaLLM(model="qwen2.5-coder:7b")

template = """
    You are an expert in answering questions about code repositories.
    Here is the complete code repository: {document}
    Here is the question : {question}

    Please provide answer to the user relating your context to the code repository.
    If the customer wants code, please provide code with best practice.
    If the user will have to change current files or create new files, provide shell commands to do so.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


# Define the path to your existing Chroma DB
persist_directory = "./chroma_db"  # your actual path
collection_name = "code_documents"  # must match what was used originally

# Reinitialize the embedding model
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Load existing Chroma DB
existing_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name=collection_name,
)

vectorstore = DocumentVectorizer(db=existing_db)

while True:
    print("\n\n------------------------------------")
    question = input("Ask your question (Press q to quit):")
    print("\n\n")
    print(question)
    if question == "q":
        break
    documents_score = vectorstore.similarity_search_with_score(question)
    related_documents = vectorstore.similarity_search(question, k=5)
    print(related_documents)
    result = chain.invoke(
        {
            "document": f" {related_documents}",
            "question": question,
        }
    )

    print(result)
