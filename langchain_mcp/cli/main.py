from langchain_core import documents
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp.readers.directory_reader import DirectoryFileReader
from langchain_mcp.readers.pdfextractor import PDFTextExtractor
from langchain_mcp.agent.policy_agent import PolicyAgent
from langchain_mcp.readers.scraper import WebScraper
from langchain_mcp.summary_generator import SummaryGenerator
from langchain_mcp.vector_store.vector import DocumentVectorizer


# summary_generator = SummaryGenerator(
#     "hoptodesk.db", path="/Users/kiransilwal/StudioProjects/upwork/hoptodesk/"
# )
# files_with_content = summary_generator.directory_reader.collect_files()
# print(f"Reading ${len(files_with_content)} files\n")
# start_key = (
#     "/Users/kiransilwal/StudioProjects/upwork/hoptodesk/"  # e.g. "some_file.txt"
# )
# vectorizer = DocumentVectorizer(
#     embedding_model_name="mxbai-embed-large:latest",
#     sqlite_path="hoptodesk.db",
#     content_column="content",
#     metadata_columns=["directory"],
#     db_path="./hoptodesk",
#     chunk_size=1000,
# )
# start_processing = True
#
# vectorizer.ingest_from_sqlite(table_name="files")
#

# model = OllamaLLM(model="qwen2.5-coder:7b", num_ctx=1000)


# def store_document():
#     start_processing = False
#
#     for key, value in files_with_content.items():
#         print(f"Read \n {key}")
#         result = summary_generator.invoke(doc=value)
#         print(f"Summary : \n\n {result}")
#         summary_generator.create_row(file_path=key, summary=result)
#
#
# store_document()

# query = """How are requests made to the server from this app?"""
# results = vectorizer.similarity_search(query, k=5)  # get top 5 docs
# texts = "\n".join([doc.page_content for doc in results]) if results else ""
#
# policy_agent = PolicyAgent(
#     violation_store=vectorizer, model=model, policy_store=None, policy=texts
# )

# policy_status = policy_agent.policy_check(policy_data=query, k=5)


# for i, doc in enumerate(results, 1):
#     print(f"Result {i}:")
#     print(f"Content: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")
#     print("-" * 30)
#
#


# model = OllamaLLM(model="qwen2.5-coder:7b")
#
# template = """
#     You are an expert in answering questions about documents.
#     Here is the complete document: {document}
#     Here is the question : {question}
#
#     If there is no related data in the document, then please reply
#     gracefully that you can't relate to what the user is searching for in the document provided
#     If there is related data, then do not mention the data, just provide the information to the user.
#     Please provide code snippets to user if the user wants you to generate code. If the user is just asking a question then answer without code
# """
#
# prompt = ChatPromptTemplate.from_template(template)
#
# chain = prompt | model

### For RAGing from a URL
# scraper = WebScraper()
# url = "https://www.immigration.govt.nz/new-zealand-visas/preparing-a-visa-application/support-family/supporting-visa-applications-for-family/student-visa-holder-supporting-visas-for-family"
# result = scraper.scrape(url)
# text = result["text"]

### for RAGing from a directory
# directory_reader = DirectoryFileReader(
#     directory="/Users/kiransilwal/upwork/xoibitflutter/lib/"
# )
# text = directory_reader.get_raw_combined_text()

### For RAGing PDF from URL
# pdfextractor = PDFTextExtractor(
#     source="https://www.immigration.govt.nz/documents/amendment-circulars/amendment-circular-2024-14.pdf"
# )
# text = pdfextractor.extract_text()


# vectorstore = DocumentVectorizer()
# document_chunks = vectorstore.process_text(text=text)
# ids = vectorstore.add_documents(documents=document_chunks)
# print(f"Added {len(ids)} document chunks")


# while True:
#     print("\n\n------------------------------------")
#     question = input("Ask your question (Press q to quit):")
#     print("\n\n")
#     print(question)
#     if question == "q":
#         break
#     documents_score = vectorstore.similarity_search_with_score(question)
#     related_documents = vectorstore.similarity_search(question, k=5)
#     print(related_documents)
#     result = chain.invoke(
#         {
#             "document": f" {related_documents}",
#             "question": question,
#         }
#     )
#
#     print(result)
