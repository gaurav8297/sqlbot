from datetime import datetime, timedelta


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import read_json


def plot(top_cluster):
    min_timestamp = datetime.max
    max_timestamp = datetime.min

    timestamps = []
    query_counts = []
    timestamp_hour_count = {}
    for time, count in top_cluster:
        # accurate to the minute
        timestamp = datetime.fromisoformat(time)
        min_timestamp = min(min_timestamp, timestamp)
        max_timestamp = max(max_timestamp, timestamp)

        timestamps.append(timestamp)
        query_counts.append(count)

        timestamp_hour = timestamp.replace(minute=0)
        if timestamp_hour_count.get(timestamp_hour, None) is not None:
            timestamp_hour_count[timestamp_hour] += count
        else:
            timestamp_hour_count[timestamp_hour] = count

    mins = 6 * 24  # 3 days

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00:00'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.plot(timestamps[0: mins], query_counts[0: mins])
    plt.gcf().autofmt_xdate()
    plt.ylabel('# of queries / 10 mins')
    plt.show()

    current_date = min_timestamp.replace(minute=0)
    timestamp_hours = []
    query_count_hours = []
    for i in range(24 * 9, 24 * 30):
        d = current_date + timedelta(hours=i)
        timestamp_hours.append(d)
        query_count_hours.append(timestamp_hour_count.get(d, 0))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.plot(timestamp_hours, query_count_hours)
    plt.gcf().autofmt_xdate()
    plt.ylabel('# of queries / hour')
    plt.show()


if __name__ == '__main__':
    clusters = read_json("result_clusters.json")
    top_clusters = read_json("top_clusters.json")
    print(len(clusters))

    top_cluster_1 = top_clusters[-1][1][0][0]
    plot(clusters[str(top_cluster_1)])
    top_cluster_2 = top_clusters[-1][1][1][0]
    plot(clusters[str(top_cluster_2)])
    top_cluster_3 = top_clusters[-1][1][2][0]
    plot(clusters[str(top_cluster_3)])
    # top_cluster_4 = top_clusters[-1][1][3][0]
    # plot(clusters[str(top_cluster_3)])
    # top_cluster_5 = top_clusters[-1][1][4][0]
    # plot(clusters[str(top_cluster_3)])
