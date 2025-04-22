import re
import os
import tiktoken

from bs4 import BeautifulSoup
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Use Ollama for embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)  # You can also try 'llama2', 'gemma', etc.


def count_tokens(text, model="cl100k_base"):
    """Count the number of tokens in the text using tiktoken."""
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main_content = soup.find("article", class_="md-content__inner")
    content = main_content.get_text() if main_content else soup.text
    return re.sub(r"\n\n+", "\n\n", content).strip()


def load_langgraph_docs():
    """Load LangGraph documentation from the official website."""
    urls = [
        "https://ai.pydantic.dev/agents/index.md",
    ]

    docs = []
    for url in urls:
        loader = RecursiveUrlLoader(url, max_depth=5, extractor=bs4_extractor)
        docs.extend(loader.lazy_load())

    print(f"Loaded {len(docs)} documents.")
    return docs


def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000, chunk_overlap=500
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(splits):
    """Create a vector store using Ollama embeddings."""
    persist_path = os.getcwd() + "/sklearn_vectorstore.parquet"

    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet",
    )
    vectorstore.persist()
    print(f"Vector store persisted at {persist_path}")
    return vectorstore


# Load, split, and store documents
# documents = load_langgraph_docs()
# split_docs = split_documents(documents)
# vectorstore = create_vectorstore(split_docs)
#
# # Create retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def doc_query_tools(query: str):
    """Query the LangGraph documentation using a retriever."""
    retriever = SKLearnVectorStore(
        embedding=OllamaEmbeddings(model="qwen2.5-coder:7b"),
        persist_path=os.getcwd() + "/sklearn_vectorstore.parquet",
        serializer="parquet",
    ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    return "\n\n".join(
        [
            f"==DOCUMENT {i + 1}==\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ]
    )


# Use Ollama for local chat model
llm = ChatOllama(model="qwen2.5-coder:7b")  # You can change this to another local model

# Augment LLM with retriever tool
augmented_llm = llm.bind_tools([doc_query_tools])

instructions = """You are a helpful assistant that can answer questions about any document.
Use the available tools for any questions about documents and provide answer accordingly.
If you don't know the answer, say 'I don't know.'"""

messages = [
    {"role": "system", "content": instructions},
    {
        "role": "user",
        "content": "What do you know about pydantic?",
    },
]

message = augmented_llm.invoke(messages)
message.pretty_print()
