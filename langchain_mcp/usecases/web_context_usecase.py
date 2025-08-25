from langchain_ollama.llms import OllamaLLM
from langchain_mcp.context.web_context import WebContext

model = OllamaLLM(model="qwen2.5-coder:7b", num_ctx=32767)

apple_policy = WebContext(
    url="https://developer.apple.com/app-store/review/guidelines/#introduction",
    model=model,
    force_rescrape=False,  # Set to True to re-scrape
)

answer = apple_policy.ask(
    "What are the rules pertaining to crypto? Please provide reference on what part of the document mentions the answer to the question too.",
    k=10,
)
print(answer)
