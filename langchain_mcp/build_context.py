from langchain_mcp.context.context_builder import ContextBuilder
from langchain_mcp.context.directory_context import DirectoryContext
from langchain_mcp.context.pdf_context_builder import PDFContextBuilder
from langchain_mcp.context.web_context_builder import WebContextBuilder


class BuildContext:
    def __init__(self):
        self.context_builder = ContextBuilder()
        self.directory_context = DirectoryContext()
        self.pdf_context_builder = PDFContextBuilder()
        self.web_context_builder = WebContextBuilder()

    def build_directory_context(self, directory_path: str):
        return self.directory_context.build_context(directory_path)

    def build_pdf_context(self, pdf_path: str):
        return self.pdf_context_builder.build_context(pdf_path)

    def build_web_context(self, url: str):
        return self.web_context_builder.build_context(url)
