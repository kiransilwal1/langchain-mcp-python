from langchain_ollama.llms import OllamaLLM

from langchain_mcp.context.context_builder import ContextBuilder
from langchain_mcp.readers.directory_reader import DirectoryFileReader
from langchain_mcp.vector_store.vector import DocumentVectorizer

# 1. Setup components
reader = DirectoryFileReader(
    directory="/Users/kiransilwal/MachineLearning/mcp/sql-connect"
)
vectorizer = DocumentVectorizer()
llm = OllamaLLM(
    model="qwen2.5-coder:0.5b"
)  # or pass a function like lambda text: your_summarizer(text)

# 2. Create ingestor
ingestor = ContextBuilder(
    file_reader=reader,
    vectorizer=vectorizer,
    llm=lambda text: llm.invoke(f"Summarize this:\n{text}"),
)

# 3. Run ingestion
ingestor.ingest()
