from langchain_ollama.llms import OllamaLLM
from langchain_mcp.context.pdf_context import PdfContext

model = OllamaLLM(model="qwen2.5-coder:7b", num_ctx=32767)

pdf_context = PdfContext(
    source="nz-immigration.pdf",
    model=model,
)

answer = pdf_context.ask(
    "What is outlined in the residence instructions? Please provide reference on what part of the document mentions the answer to the question too.",
    k=20,
)
print(answer)
