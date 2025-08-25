from typing import Optional
import hashlib
from langchain_ollama.llms import OllamaLLM
from langchain_mcp.readers.pdfextractor import PDFTextExtractor
from langchain_mcp.summary_generator import SummaryGenerator
from langchain_mcp.vector_store.vector import DocumentVectorizer
from langchain_core.prompts import ChatPromptTemplate


class PdfContext:
    def __init__(
        self,
        source: str,
        model: OllamaLLM,
        summarizer: Optional[SummaryGenerator] = None,
        base_db_dir: str = "directory_chroma_dbs",
    ):
        """
        Args:
            url: The web URL to scrape and vectorize.
            model: The LLM model to use for summarization or Q&A.
            summarizer: Optional summarizer module.
            base_db_dir: Base directory where all vector DBs are stored.
            force_rescrape: Whether to force re-scraping and re-vectorizing.
        """
        self.model = model
        self.summarizer = summarizer
        self.db_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
        self.db_path = os.path.join(base_db_dir, self.db_hash)

        db_exists = os.path.exists(self.db_path) and os.listdir(self.db_path)

        self.vectorstore = DocumentVectorizer(
            db_path=self.db_path,
        )

        if not db_exists:
            self.pdfextractor = PDFTextExtractor(source=source)
            content = self.pdfextractor.extract_text()
            if not content:
                raise ValueError("No content available to vectorize.")
            print(content)

            chunks = self.vectorstore.process_text(text=content)
            self.vectorstore.add_documents(documents=chunks)
        else:
            print(f"[Loading existing vector DB from] {self.db_path}")
            self.content = None  # Already in vectorstore

    def ask(self, question: str, k=5) -> str:
        similar_docs = self.vectorstore.similarity_search(query=question, k=k)
        if not similar_docs:
            return "No relevant documents found. Try force_rescrape=True to refresh."

        # Concatenate all relevant document chunks
        combined_docs = " ".join(doc.page_content for doc in similar_docs)
        similarity = self.vectorstore.similarity_search_with_score(query=question, k=5)
        print(f"Similarity with Score : {similarity}")

        # Prompt template
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
        chain = prompt | self.model

        result = chain.invoke(
            {
                "document": combined_docs,
                "question": question,
            }
        )

        return result.strip()
