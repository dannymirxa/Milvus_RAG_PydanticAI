import glob, os
import pymupdf4llm
import pathlib
import re
from chonkie import RecursiveChunker, SemanticChunker

from dotenv import load_dotenv
load_dotenv('.env')

chunker = RecursiveChunker.from_recipe("markdown")

from chonkie.embeddings.azure_openai import AzureOpenAIEmbeddings

# Initialize Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
	azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
	azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
	model="text-embedding-3-large",
	deployment="text-embedding-3-large"
)

# Single embedding
# emb = embeddings.embed("your text here")

# Batch embedding
# embs = embeddings.embed_batch(["text1", "text2"])

# chunker = SemanticChunker(
#     # embedding_model="BAAI/bge-m3",                 # Default model
#     embedding_model=embeddings,
#     threshold=0.5,                               # Similarity threshold (0-1) or (1-100) or "auto"
#     chunk_size=3072,                              # Maximum tokens per chunk
#     min_sentences=1                              # Initial sentences per chunk
# )

def _list_files(dir: str) -> list[str]:
    file_names = [os.path.join(dir, f) for f in os.listdir(dir)]

    return file_names

def create_file_data(dir: str) -> dict[str, str]:
    file_names = _list_files(dir)
    file_contents= []

    for file_name in file_names:
        with open(file_name, "r") as f:
            file_contents.append({"source": file_name.replace(dir + os.sep, ""), "text": f.read()})

    return file_contents

# def _convert_pdf_to_md(dir: str) -> str:
#     md_text = ""
#     for file in glob.glob(f"{dir}/*.pdf", recursive=True):
#         md_text = pymupdf4llm.to_markdown(file)
#         # pathlib.Path(f"{dir}.md").write_bytes(md_text.encode())
#     return md_text

def create_docs_chunks_md(dir: str) -> dict[str, str]:
    """
    Read markdown files in dir, extract any 'Selected References' section and store it
    as metadata, then chunk the remaining content (with the references section removed).
    Each returned chunk dict will include a "references" field (empty string if none found).
    """
    docs_chunks = []

    # regex to find the "Selected References" section (header starts with "## **Selected References")
    # capture everything until the next "## " header or end of file
    refs_pattern = re.compile(
        r"^##\s*\*\*Selected References[^\n]*\n(.*?)(?=\n##\s|\Z)",
        flags=re.S | re.M,
    )

    for file in glob.glob(f"{dir}/*.md", recursive=True):
        with open(file, "r", encoding="utf8") as f:
            content = f.read()

        # Extract references if present
        references = ""
        m = refs_pattern.search(content)
        if m:
            # Normalize whitespace and strip surrounding newlines
            references = m.group(1).strip()
            # Remove the entire references section from the content before chunking
            content_without_refs = refs_pattern.sub("\n", content)
        else:
            content_without_refs = content

        # Chunk the content that excludes the references section
        chunks = chunker(content_without_refs)

        for index, chunk in enumerate(chunks):
            entry = {
                "source_id": f"{file.replace(dir + os.sep, '')}_{index}",
                # "source": file.replace(dir + os.sep, ''),
                "text": chunk.text,
            }
            # Include references metadata (same for all chunks from this file)
            entry["references"] = references
            docs_chunks.append(entry)

    return docs_chunks

# md_text = pymupdf4llm.to_markdown("Files/Business_Leadership_Playbook_Change_Transformation.pdf")
# import pathlib
# pathlib.Path("Files/Business_Leadership_Playbook_Change_Transformation.md").write_bytes(md_text.encode())

# def create_docs_chunks_pdf(dir: str) -> dict[str, str]:
#     docs_chunks = []
#     content = _convert_pdf_to_md(dir)
#     chunks = chunker(content)
    
#     for index, chunk in enumerate(chunks):
#         docs_chunks.append(
#             {   
#                 "source_id": str(index),
#                 # "source": file.replace(dir + os.sep, ''),
#                 "text": chunk.text,
#             }
#         )
#     return docs_chunks

# import json
# with open('file_data.json', 'w') as fp:
#     json.dump(create_docs_chunks("Transformation Model"), fp)

# print(glob.glob('Transformation Model/*.md', recursive=True))


# result = create_docs_chunks("Transformation Model")
# print(result)


# print(create_docs_chunks_pdf("Files"))
