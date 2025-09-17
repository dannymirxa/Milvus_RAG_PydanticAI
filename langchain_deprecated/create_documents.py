from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_documents(markdown_files: list[str]) -> list[Document]:
    all_documents = []
    for markdown_file in markdown_files:
        loader = UnstructuredMarkdownLoader(markdown_file)
        data = loader.load()
        all_documents.extend(data)

    # Initialize a RecursiveCharacterTextSplitter for splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # Split the documents into chunks using the text_splitter
    docs = text_splitter.split_documents(all_documents)

    return docs