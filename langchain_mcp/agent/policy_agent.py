from langchain_mcp.chains.chain_object import ChainObject
from langchain_ollama.llms import OllamaLLM
from langchain_mcp.vector_store.vector import DocumentVectorizer
from langchain_core.prompts import ChatPromptTemplate


class PolicyAgent:
    def __init__(
        self,
        policy_store: DocumentVectorizer | None,
        violation_store: DocumentVectorizer,
        model: OllamaLLM,
        policy: str,
    ):
        self.policy_store = policy_store
        self.policy = policy
        self.template = """You are a Policy Agent. You are provided with two things. A user query and description of what the program does that are closely related to the user query. Your job is to find out if
        the code description provided by the user violates the policy or not. Below are the needed details.
        policy = {policy}
        description = {code_description}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.violation_store = violation_store
        self.chain = ChainObject(prompt=self.prompt, model=model)

    def policy_check(self, policy_data: str, k: int):
        relevant_codes = self.violation_store.similarity_search(policy_data, k=k)
        result = []
        for document in relevant_codes:
            result.append(
                self.chain.chain.invoke(
                    {"policy": f"{self.policy}", "code_description": f"{document}"}
                )
            )
        return result
