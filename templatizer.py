import json
import re
from datetime import datetime

from utils import read_logs

STRING_REGEX = r'([^\\])\'((\')|(.*?([^\\])\'))'
DOUBLE_QUOTE_STRING_REGEX = r'([^\\])"((")|(.*?([^\\])"))'

INT_REGEX = r'([^a-zA-Z])-?\d+(\.\d+)?'  # To prevent us from capturing table name like "a1"

HASH_REGEX = r'(\'\d+\\.*?\')'


def get_template(query, timestamp, templated_workload):
    template = re.sub(HASH_REGEX, r"@@@", query)
    template = re.sub(STRING_REGEX, r"\1&&&", template)
    template = re.sub(DOUBLE_QUOTE_STRING_REGEX, r"\1&&&", template)
    template = re.sub(INT_REGEX, r"\1#", template)

    if template in templated_workload:
        # add timestamp
        if timestamp in templated_workload[template]:
            templated_workload[template][timestamp] += 1
        else:
            templated_workload[template][timestamp] = 1
    else:
        templated_workload[template] = dict()
        templated_workload[template][timestamp] = 1

    return templated_workload


def generate_templates(logs):
    min_timestamp = datetime.max
    max_timestamp = datetime.min

    templated_workload = {}
    for query_log in logs:
        # accurate to the minute
        timestamp = datetime.fromisoformat(query_log['timestamp']).replace(second=0)
        get_template(query_log['query'], timestamp.isoformat(), templated_workload)

    return templated_workload


if __name__ == "__main__":
    logs = read_logs("tpc_ds.log")
    with open("templated_queries.json", "w") as f:
        f.write(json.dumps(generate_templates(logs)))
