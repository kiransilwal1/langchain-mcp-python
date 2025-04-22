from langchain_core import documents
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from directory_reader import DirectoryFileReader
from pdfextractor import PDFTextExtractor
from scraper import WebScraper
from vector import DocumentVectorizer

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

### For RAGing from a URL
# scraper = WebScraper()
# url = "https://www.immigration.govt.nz/new-zealand-visas/preparing-a-visa-application/support-family/supporting-visa-applications-for-family/student-visa-holder-supporting-visas-for-family"
# result = scraper.scrape(url)
# text = result["text"]

### for RAGing from a directory
# directory_reader = DirectoryFileReader(
#     # directory="/Users/kiransilwal/library/Mobile Documents/iCloud~md~obsidian/Documents/KiranSecondBrain/notes/"
#     directory="/Users/kiransilwal/React/Next/leagify/src/lib/features/auth/"
# )
# text = directory_reader.get_raw_combined_text()

### For RAGing PDF from URL
# pdfextractor = PDFTextExtractor(
#     source="https://www.immigration.govt.nz/documents/amendment-circulars/amendment-circular-2024-14.pdf"
# )
# text = pdfextractor.extract_text()
vectorstore = DocumentVectorizer()
# document_chunks = vectorstore.process_text(text=text)
# ids = vectorstore.add_documents(documents=document_chunks)
# print(f"Added {len(ids)} document chunks")


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
