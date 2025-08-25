from typing import Callable, Dict, List, Optional, Any
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from langchain_mcp.readers.directory_reader import DirectoryFileReader
from langchain_mcp.vector_store.vector import DocumentVectorizer

# Use your existing classes
# from directory_file_reader import DirectoryFileReader
# from document_vectorizer import DocumentVectorizer


class LLMSummarizer:
    def __init__(self, llm: Callable[[str], str], context_size: int = 1000):
        """
        Initialize the LLMSummarizer.
        Args:
            llm: A callable language model that takes a string input and returns a string summary.
            context_size: Maximum number of characters from the input text to use for summarization.
        """
        self.llm = llm
        self.context_size = context_size

    def summarize(
        self,
        text: str,
        instruction: str = "Summarize the following code block. The summary shall reflect what the code block is intended to do:",
    ) -> str:
        """
        Summarize the given text using the LLM with optional instruction.
        Args:
            text: The full text to summarize (only first `context_size` chars will be used).
            instruction: Prompt to prepend to the input text for the LLM.
        Returns:
            A summary string from the LLM.
        """
        try:
            print(f"Incoming text length: {len(text)} characters")
            trimmed_text = text[: self.context_size]
            prompt = f"{instruction}\n\n{trimmed_text}"
            summary = self.llm(prompt)
            print(f"Summary {len(summary)}: {summary[:100]}...")
            return summary
        except Exception as e:
            return f"Summarization failed: {str(e)}"


class DocumentProcessor:
    """
    A class that combines directory reading, summarization, and vectorization.
    """

    def __init__(
        self,
        file_reader: DirectoryFileReader,
        summarizer: LLMSummarizer,
        vectorizer: DocumentVectorizer,
    ):
        """
        Initialize the document processor.

        Args:
            file_reader: Instance of DirectoryFileReader
            summarizer: Instance of LLMSummarizer
            vectorizer: Instance of DocumentVectorizer
        """
        self.file_reader = file_reader
        self.summarizer = summarizer
        self.vectorizer = vectorizer

    def process_directory(
        self, additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Process all files in the configured directory, summarize them,
        and add them to the vector database with metadata.

        Args:
            additional_metadata: Optional additional metadata to include for all documents

        Returns:
            List of document IDs added to the vector database
        """
        # Collect files
        files_content = self.file_reader.collect_files()

        # Process each file
        all_ids = []
        for file_path, content in files_content.items():
            # Create basic metadata for the file
            print(f"Path currently summarizing : {file_path}")
            relative_path = os.path.relpath(file_path, self.file_reader.directory)
            file_metadata = {
                "file_path": file_path,
                "relative_path": relative_path,
                "file_type": os.path.splitext(file_path)[1].lower().lstrip("."),
                "file_size": len(content),
            }

            # Generate summary using LLM
            summary = self.summarizer.summarize(content)
            file_metadata["summary"] = summary

            # Add any additional metadata
            if additional_metadata:
                file_metadata.update(additional_metadata)

            print(f"Processing file: {relative_path}")

            # Add to vector database
            ids = self.vectorizer.add_text(content, metadata=file_metadata)
            all_ids.extend(ids)

        return all_ids

    def query_documents(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for documents related to the query.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of documents with their scores
        """
        return self.vectorizer.similarity_search_with_score(query, k=k)


# Example usage
if __name__ == "__main__":
    # Example LLM function - replace with your actual LLM implementation
    summarizer = OllamaLLM(model="qwen2.5-coder:7b")

    # Initialize components
    file_reader = DirectoryFileReader(
        file_extensions=["py", "md", "js", "ts", "json", "php"],
        directory="/Users/kiransilwal/MachineLearning/mcp/langchain-mcp/",
        max_depth=5,
    )

    summarizer = LLMSummarizer(llm=summarizer, context_size=500)

    vectorizer = DocumentVectorizer(
        embedding_model_name="mxbai-embed-large",
        db_path="./chroma_db",
        collection_name="code_documents",
        chunk_size=500,
        chunk_overlap=100,
    )

    # Create the processor
    processor = DocumentProcessor(file_reader, summarizer, vectorizer)

    # Process all files in the directory
    additional_metadata = {"project": "my_project", "processed_date": "2025-04-21"}

    doc_ids = processor.process_directory(additional_metadata)
    print(f"Added {len(doc_ids)} document chunks to the vector database")

    # Query example
    # results = processor.query_documents("How does the document vectorizer work?")
    # for doc, score in results:
    #     print(f"Score: {score:.4f}")
    #     print(f"Path: {doc.metadata.get('relative_path', 'N/A')}")
    #     print(
    #         f"Summary: {doc.metadata.get('summary', 'No summary available')[:100]}..."
    #     )
    #     print(f"Content: {doc.page_content[:100]}...")
    #     print("-" * 80)
    while True:
        model = OllamaLLM(model="qwen2.5-coder:7b")

        template = """
            You are an expert in answering questions about documents.
            Here is the complete document: {document}
            Here is the question : {question}

            If there is no related data in the document, then please reply 
            gracefully that you can't relate to what the user is searching for in the document provided
            If there is related data, then do not mention the data, just provide the information to the user.
            Please provide code snippets to user if the user wants you to generate code. If the user is just asking a question then answer without code
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | model

        print("\n\n------------------------------------")
        question = input("Ask your question (Press q to quit):")
        print("\n\n")
        print(question)
        if question == "q":
            break
        documents_score = vectorizer.similarity_search_with_score(question)
        related_documents = vectorizer.similarity_search(question, k=5)
        print(related_documents)
        result = chain.invoke(
            {
                "document": f" {related_documents}",
                "question": question,
            }
        )

        print(result)
