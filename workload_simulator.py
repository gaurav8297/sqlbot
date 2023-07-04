import json
import random
from datetime import datetime, timedelta

start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 3, 30)

hours_to_query_range = {(12, 14): [1, 20], (15, 17): [21, 49], (18, 20): [50, 99]}


def generate_fake_logs(current_date, end_date):
    with open("tpc_ds.log", "w") as f:
        while current_date <= end_date:
            current_date += timedelta(seconds=random.randrange(10, 60))
            for key in hours_to_query_range:
                val = hours_to_query_range[key]
                if current_date.hour in key:
                    q_id = random.randrange(val[0], val[1])
                    with open(f"queries/query_{q_id}.sql", "r") as q:
                        query = q.read()
                        data = {"query": query.strip(), "timestamp": current_date.isoformat()}
                        f.write(json.dumps(data))
                        f.write("\n")


if __name__ == "__main__":
    # For now it only generates cycles
    # Todo: Add peaks on peak day and workload frequency increase
    generate_fake_logs(start_date, end_date)
