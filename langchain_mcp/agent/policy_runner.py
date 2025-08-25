from langchain_ollama.llms import OllamaLLM
from policy_agent import PolicyAgent
from vector import DocumentVectorizer

vectorizer = DocumentVectorizer(
    sqlite_path="xoibit.db",
    content_column="content",
    metadata_columns=["directory"],
    db_path="./xoibit_chroma",
    chunk_size=32767,
)

model = OllamaLLM(model="qwen2.5-coder:7b", num_ctx=32767)

query = """If you want to unlock features or functionality within your app, (by way of example: subscriptions, in-game currencies, game levels, access to premium content, or unlocking a full version), you must use in-app purchase. Apps may not use their own mechanisms to unlock content or functionality, such as license keys, augmented reality markers, QR codes, cryptocurrencies and cryptocurrency wallets, etc."""

policy_agent = PolicyAgent(
    violation_store=vectorizer, model=model, policy_store=None, policy=query
)

policy_status = policy_agent.policy_check(policy_data=query, k=5)

print(policy_status)
