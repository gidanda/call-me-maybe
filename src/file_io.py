import json

def read_json(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
