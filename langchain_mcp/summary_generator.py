from langchain_mcp.vector_store.database import Database
from langchain_mcp.readers.directory_reader import DirectoryFileReader
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class SummaryGenerator:
    def __init__(
        self,
        database_name,
        model_name: str = "qwen2.5-coder:0.5b",
        num_ctx: int = 32767,
        template="""You are an expert code summarizer. 
        You will be provided with a content of a file. 
        You will summarize the code in very concise and layman terms. 
        Here is the code : {document}
        Please do not re-write any codes. Just explain what the code does
        """,
        path="/Users/kiransilwal/upwork/xoibitflutter/lib/",
    ):
        self.template = template
        self.num_ctx = num_ctx
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = OllamaLLM(model=model_name, num_ctx=num_ctx)
        self.chain = self.prompt | self.model
        self.directory_reader = DirectoryFileReader(directory=path)
        self.database_name = database_name
        self.database = Database(self.database_name)

    def invoke(self, doc: str) -> str:
        response = ""
        return (
            self.chain.invoke({"document": f"{doc}"})
            if len(doc) < self.num_ctx
            else "0"
        )

    def file_reader(self, directory: str) -> str:
        try:
            with open(directory, "r", encoding="utf-8") as file:
                value = file.read()
                return value
        except Exception as e:
            # Handle file reading errors
            return str(e.__cause__)

    def summary_generator(self, content: str) -> str:
        result = self.chain.invoke({"doucment": f"{content}"})
        return result

    def create_row(self, file_path: str, summary: str):
        self.database.create_file(directory=file_path, content=summary)
