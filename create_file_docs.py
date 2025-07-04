import glob, os
from chonkie import RecursiveChunker

chunker = RecursiveChunker.from_recipe("markdown")

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

def create_docs_chunks(dir: str) -> dict[str, str]:

    docs_chunks = []

    for file in glob.glob(f"{dir}/*.md", recursive=True):
        with open(file, "r", encoding="utf8") as f:
            content = f.read()

        chunks = chunker(content)
        
        for index, chunk in enumerate(chunks):
            docs_chunks.append(
                {   
                    "source_id": f"{file.replace(dir + os.sep, '')}_{index}",
                    # "source": file.replace(dir + os.sep, ''),
                    "text": chunk.text,
                }
            )
    return docs_chunks

# import json
# with open('file_data.json', 'w') as fp:
#     json.dump(create_docs_chunks("Transformation Model"), fp)

# print(glob.glob('Transformation Model/*.md', recursive=True))


# result = create_docs_chunks("Transformation Model")
# print(result)

