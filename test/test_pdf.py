import pymupdf4llm
import pathlib

md_text = pymupdf4llm.to_markdown("Files/Organization Culture and Leadership 5th Edition.pdf")

pathlib.Path("output.md").write_bytes(md_text.encode())