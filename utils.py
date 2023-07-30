import json


def read_logs(file_name):
    res = []
    with open(file_name, "r") as f:
        for line in f:
            res.append(json.loads(line))
    return res


def read_json(file_name):
    with open(file_name, "r") as f:
        return json.loads(f.read())
