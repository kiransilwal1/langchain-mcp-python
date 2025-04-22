import os
from typing import List, Optional, Union, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class DocumentVectorizer:
    def __init__(
        self,
        embedding_model_name: str = "mxbai-embed-large",
        db: Optional[Chroma] = None,
        db_path: str = "./chroma_db",
        collection_name: str = "document_collection",
        chunk_size: int = 500,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the document vectorizer with specified parameters.

        Args:
            embedding_model_name: Name of the Ollama embedding model to use
            db: Optional pre-initialized Chroma vector database
            db_path: Path to store the ChromaDB database (used only if db is not provided)
            collection_name: Name of the collection in ChromaDB (used only if db is not provided)
            chunk_size: Maximum size of text chunks
            chunk_overlap: Amount of overlap between chunks
        """
        self.embedding_model_name = embedding_model_name
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(model=embedding_model_name)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Use provided DB or initialize one
        self.db = db or self._init_vector_db()

    def _init_vector_db(self) -> Chroma:
        """Initialize or load the vector database"""
        os.makedirs(self.db_path, exist_ok=True)
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

    def process_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process text content by splitting it into chunks.

        Args:
            text: The text content to process
            metadata: Optional metadata to attach to all documents

        Returns:
            List of Document objects after chunking
        """
        # Create a document from text
        doc = Document(page_content=text, metadata=metadata or {})

        # Split the document into chunks
        chunks = self.text_splitter.split_documents([doc])

        return chunks

    def add_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Add text content to the vector database.

        Args:
            text: The text content to add
            metadata: Optional metadata to attach to all documents

        Returns:
            List of document IDs for the added chunks
        """
        # Process the text into chunks
        chunks = self.process_text(text, metadata)

        # Add chunks to the vector database
        ids = self.db.add_documents(chunks)

        # Persist changes to disk
        self.db.persist()

        return ids

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add pre-created documents to the vector database.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs for the added documents
        """
        # Split the documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Add chunks to the vector database
        ids = self.db.add_documents(chunks)

        # Persist changes to disk
        self.db.persist()

        return ids

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple text contents to the vector database.

        Args:
            texts: List of text contents to add
            metadatas: Optional list of metadata dictionaries (one per text)

        Returns:
            List of document IDs for the added chunks
        """
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")

        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            print(f"{i} document added")
            documents.append(Document(page_content=text, metadata=metadata))

        return self.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for documents similar to the query.

        Args:
            query: The query string
            k: Number of results to return

        Returns:
            List of relevant Document objects
        """
        return self.db.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        Search for documents similar to the query and return relevance scores.

        Args:
            query: The query string
            k: Number of results to return

        Returns:
            List of tuples containing Document objects and their relevance scores
        """
        return self.db.similarity_search_with_score(query, k=k)
