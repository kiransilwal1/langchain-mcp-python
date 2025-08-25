from langchain_ollama.llms import OllamaLLM


class ChainObject:
    def __init__(self, prompt: str, model: OllamaLLM):
        self.prompt = prompt
        self.template = prompt
        self.chain = self.prompt | model
