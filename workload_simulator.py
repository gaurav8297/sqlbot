import json
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import read_logs

start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 4, 1)

hours_to_query_range = {(12, 15): [1, 20], (15, 18): [21, 49], (18, 20): [50, 99]}
peak_days = [[datetime(2022, 1, 10), 150], [datetime(2022, 1, 28), 150],
             [datetime(2022, 2, 10), 150], [datetime(2022, 2, 28), 150],
             [datetime(2022, 3, 10), 150], [datetime(2022, 3, 28), 150]]
    # ,
    #          [datetime(2022, 4, 10), 150], [datetime(2022, 4, 30), 150],
    #          [datetime(2022, 5, 10), 1000], [datetime(2022, 5, 30), 1000],
    #          [datetime(2022, 6, 10), 1000], [datetime(2022, 6, 29), 1000],
    #          [datetime(2022, 7, 11), 1000], [datetime(2022, 7, 31), 1000],
    #          [datetime(2022, 8, 12), 1000], [datetime(2022, 8, 30), 1000],
    #          [datetime(2022, 9, 10), 1000], [datetime(2022, 9, 30), 1000],
    #          [datetime(2022, 10, 10), 1000], [datetime(2022, 10, 30), 1000],
    #          [datetime(2022, 11, 10), 1000], [datetime(2022, 11, 28), 1000],
    #          [datetime(2022, 12, 10), 1000], [datetime(2022, 12, 30), 1000]]
small_batch_per_4_day = 2
new_release_iter_mul = 2
day_to_iter = {30: (30, 40), 190: (30, 50), 300: (50, 100)}


def generate_fake_logs():
    min_iter = 10
    max_iter = 20
    min_iter_except_peak = min_iter
    max_iter_except_peak = max_iter
    total_hours = (end_date - start_date).days * 24
    current_date = start_date
    days_increment = []
    old_date = current_date
    day = 1
    for hour in range(0, total_hours):
        peak_max_iter = None
        peak_min_iter = None
        for timestamp, iterations in peak_days:
            if current_date.day == timestamp.day and current_date.month == timestamp.month:
                peak_max_iter = iterations
                peak_min_iter = iterations - 5
                break

        if peak_max_iter:
            min_iter = min(peak_min_iter, (min_iter + 20))
            max_iter = min(peak_max_iter, (max_iter + 20))
            days_increment.append((current_date, (min_iter, max_iter)))
        else:
            min_iter = max(min_iter_except_peak, (min_iter - 40))
            max_iter = max(max_iter_except_peak, (max_iter - 40))
            days_increment.append((current_date, (min_iter, max_iter)))

        if (current_date - old_date).days >= 1:
            if day_to_iter.get(day):
                min_iter_except_peak = day_to_iter.get(day)[0]
                max_iter_except_peak = day_to_iter.get(day)[1]
            old_date = current_date
            day += 1

        current_date = current_date + timedelta(hours=1)

    dates = []
    min_iters = []
    max_iters = []
    for date, (min_iter, max_iter) in days_increment:
        dates.append(date)
        min_iters.append(min_iter)
        max_iters.append(max_iter)

    # print(min_iters)
    # print(max_iters)


    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
    plt.plot(dates, min_iters)
    plt.gcf().autofmt_xdate()
    plt.show()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
    plt.plot(dates, max_iters)
    plt.gcf().autofmt_xdate()
    plt.show()

    current_date = start_date
    total = (end_date - start_date).days
    old_date = current_date
    with open("tpc_ds.log", "w") as f:
        while current_date <= end_date:
            current_date += timedelta(minutes=1)
            # Run every hour

            loop_iters = 0
            for date, (min_iter, max_iter) in days_increment:
                if date > current_date:
                    break
                loop_iters = random.randrange(min_iter, max_iter)

            if (current_date - old_date).days >= 1:
                progress = (end_date - current_date).days / total
                print(f"progress: {progress}")
                old_date = current_date

            for key in hours_to_query_range:
                val = hours_to_query_range[key]
                if key[0] <= current_date.hour < key[1]:
                    for i in range(0, loop_iters):
                        q_id = random.randrange(val[0], val[1])
                        with open(f"queries/query_{q_id}.sql", "r") as q:
                            query = q.read()
                            data = {"query": query.strip(), "timestamp": current_date.isoformat()}
                            f.write(json.dumps(data))
                            f.write("\n")
                else:
                    choice = random.choice([True, False])
                    if choice:
                        iters = random.randrange(0, 2)
                        for i in range(0, iters):
                            q_id = random.randrange(1, 99)
                            with open(f"queries/query_{q_id}.sql", "r") as q:
                                query = q.read()
                                data = {"query": query.strip(), "timestamp": current_date.isoformat()}
                                f.write(json.dumps(data))
                                f.write("\n")


def plot():
    logs = read_logs("tpc_ds.log")
    min_timestamp = datetime.max
    max_timestamp = datetime.min

    timestamp_min_count = {}
    timestamp_max_count = {}
    for query_log in logs:
        # accurate to the minute
        timestamp = datetime.fromisoformat(query_log['timestamp']).replace(second=0)
        timestamp_hour = timestamp.replace(minute=0)
        min_timestamp = min(min_timestamp, timestamp)
        max_timestamp = max(max_timestamp, timestamp)

        if timestamp_min_count.get(timestamp, None) is not None:
            timestamp_min_count[timestamp] += 1
        else:
            timestamp_min_count[timestamp] = 1

        if timestamp_max_count.get(timestamp_hour, None) is not None:
            timestamp_max_count[timestamp_hour] += 1
        else:
            timestamp_max_count[timestamp_hour] = 1

    min_date = []
    min_cnt = []
    current_date = min_timestamp
    for i in range(0, 60 * 24 * 3):
        min_date.append(current_date)
        min_cnt.append(timestamp_max_count.get(current_date, 0))
        current_date += timedelta(minutes=1)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.plot(min_date, min_cnt)
    plt.gcf().autofmt_xdate()
    plt.show()

    current_date = min_timestamp.replace(minute=0)
    min_date = []
    min_cnt = []
    for i in range(24 * 1, 24 * 5):
        d = current_date + timedelta(hours=i)
        min_date.append(d)
        min_cnt.append(timestamp_max_count.get(d, 0))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.plot(min_date, min_cnt)
    plt.gcf().autofmt_xdate()
    plt.ylabel('# of queries / hour')
    plt.show()

    current_date = min_timestamp.replace(minute=0)
    min_date = []
    min_cnt = []
    for i in range(24 * 8, 24 * 20):
        d = current_date + timedelta(hours=i)
        min_date.append(d)
        min_cnt.append(timestamp_max_count.get(d, 0))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.plot(min_date, min_cnt)
    plt.gcf().autofmt_xdate()
    plt.ylabel('# of queries / hour')
    plt.show()

    current_date = min_timestamp.replace(minute=0)
    min_date = []
    min_cnt = []
    for i in range(24 * 28, 24 * 35):
        d = current_date + timedelta(hours=i)
        min_date.append(d)
        min_cnt.append(timestamp_max_count.get(d, 0))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.plot(min_date, min_cnt)
    plt.gcf().autofmt_xdate()
    plt.ylabel('# of queries / hour')
    plt.show()


def print_total_logs():
    logs = read_logs("tpc_ds.log")
    print(len(logs))


if __name__ == "__main__":
    # generate_fake_logs()
    plot()
    # print_total_logs()
