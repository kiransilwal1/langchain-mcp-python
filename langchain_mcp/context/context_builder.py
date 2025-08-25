from typing import Callable, Any
from langchain_core.documents import Document

from langchain_mcp.readers.directory_reader import DirectoryFileReader
from langchain_mcp.vector_store.vector import DocumentVectorizer


class ContextBuilder:
    """
    Ingests files using DirectoryFileReader, summarizes them using an LLM,
    and vectorizes them using DocumentVectorizer.
    """

    def __init__(
        self,
        file_reader: DirectoryFileReader,
        vectorizer: DocumentVectorizer,
        llm: Callable[[str], str],  # LLM function to summarize text
    ):
        self.file_reader = file_reader
        self.vectorizer = vectorizer
        self.llm = llm

    def ingest(self):
        """
        Read files, summarize each, and store them with summaries as metadata.
        """
        file_map = self.file_reader.collect_files()

        documents = []

        for path, content in file_map.items():
            if not content.strip():
                continue  # skip empty files

            try:
                # Summarize with LLM
                summary = self.llm(content)

                # Create metadata with summary and file path
                metadata = {
                    "file_path": path,
                    "summary": summary,
                }

                # Wrap as LangChain Document
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

        # Add to vector store
        return self.vectorizer.add_documents(documents)
