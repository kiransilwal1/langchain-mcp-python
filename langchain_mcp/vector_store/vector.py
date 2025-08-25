import os
from typing import List, Optional, Union, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class DocumentVectorizer:
    def __init__(
        self,
        db_path: str,
        embedding_model_name: str = "mxbai-embed-large:latest",
        db: Optional[Chroma] = None,
        collection_name: str = "document_collection",
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        batch_size: int = 5000,
        sqlite_path: Optional[str] = None,
        content_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
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
            batch_size: Maximum number of documents to process in a single batch
        """
        self.embedding_model_name = embedding_model_name
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size  # Store batch size

        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(model=embedding_model_name)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Use provided DB or initialize one
        self.db = db or self._init_vector_db()
        self.sqlite_path = sqlite_path
        self.content_column = content_column
        self.metadata_columns = metadata_columns or []

    def _init_vector_db(self) -> Chroma:
        """Initialize or load the vector database"""
        os.makedirs(self.db_path, exist_ok=True)
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

    def _add_documents_in_batches(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector database in batches to avoid size limits.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs for all added documents
        """
        all_ids = []
        total_docs = len(documents)

        if total_docs <= self.batch_size:
            # If documents fit in one batch, add them directly
            ids = self.db.add_documents(documents)
            all_ids.extend(ids)
        else:
            # Process in batches
            total_batches = (total_docs + self.batch_size - 1) // self.batch_size
            print(
                f"Processing {total_docs} documents in {total_batches} batches of max {self.batch_size} each"
            )

            for i in range(0, total_docs, self.batch_size):
                batch = documents[i : i + self.batch_size]
                batch_num = i // self.batch_size + 1

                print(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)"
                )

                try:
                    batch_ids = self.db.add_documents(batch)
                    all_ids.extend(batch_ids)
                    print(f"✓ Batch {batch_num} completed successfully")
                except Exception as e:
                    print(f"✗ Error in batch {batch_num}: {e}")
                    # Optionally, you could add retry logic here
                    # For now, we'll re-raise to stop processing
                    raise

        return all_ids

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

        # Add chunks to the vector database using batch processing
        ids = self._add_documents_in_batches(chunks)

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

        # Add chunks to the vector database using batch processing
        ids = self._add_documents_in_batches(chunks)

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

    def similarity_search(self, query: str, k: int = 1) -> List[Document]:
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

    def get_batch_size(self) -> int:
        """Get the current batch size setting."""
        return self.batch_size

    def set_batch_size(self, batch_size: int) -> None:
        """Update the batch size setting."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        self.batch_size = batch_size

    def ingest_from_sqlite(self, table_name: str) -> List[str]:
        """
        Ingest and vectorize rows from an SQLite database.

        Args:
            table_name: Name of the table to read data from.

        Returns:
            List of document IDs added to the vector store.
        """
        import sqlite3

        if not self.sqlite_path:
            raise ValueError("SQLite path is not set.")
        if not self.content_column:
            raise ValueError("Content column must be specified.")

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        # Build column list for SQL query
        selected_columns = [self.content_column] + self.metadata_columns
        sql = f"SELECT {', '.join(selected_columns)} FROM {table_name}"
        cursor.execute(sql)

        texts = []
        metadatas = []

        for row in cursor.fetchall():
            content = row[0]
            metadata = {
                key: value for key, value in zip(self.metadata_columns, row[1:])
            }

            if content:
                texts.append(content)
                metadatas.append(metadata)

        conn.close()

        if not texts:
            print("No valid rows found in the database.")
            return []

        print(f"✓ Ingested {len(texts)} rows from SQLite. Starting vectorization...")
        return self.add_texts(texts, metadatas)
