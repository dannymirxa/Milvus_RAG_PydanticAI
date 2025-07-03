import os


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


import json
with open('file_data.json', 'w') as fp:
    json.dump(create_file_data(dir="Transformation Model"), fp)