import numpy as np

from utils import read_json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot(result, top_cluster):
    top_cluster_results = result[top_cluster]

    predicted_dates = top_cluster_results.get("predictedDates")
    actual = top_cluster_results.get("actual")
    predicted = top_cluster_results.get("predicted")

    timestamp_hour_count_actual = {}
    timestamp_hour_count_predicted = {}
    timestamp_hours = []
    for i in range(0, len(predicted_dates)):
        # accurate to the minute
        timestamp = datetime.fromisoformat(predicted_dates[i]).replace(second=0)
        timestamp_hour = timestamp.replace(minute=0)

        pred = predicted[i]
        if pred < 0:
            pred = 0

        act = actual[i]
        if act < 0:
            act = 0

        if timestamp_hour_count_actual.get(timestamp_hour, None) is not None:
            timestamp_hour_count_actual[timestamp_hour] += act
        else:
            timestamp_hour_count_actual[timestamp_hour] = act
            timestamp_hours.append(timestamp_hour)

        if timestamp_hour_count_predicted.get(timestamp_hour, None) is not None:
            timestamp_hour_count_predicted[timestamp_hour] += pred
        else:
            timestamp_hour_count_predicted[timestamp_hour] = pred

    dates = []
    actual_count = []
    predicted_count = []
    for i in range(24 * 35, 24 * 50):
        timestamp_hour = timestamp_hours[i]
        dates.append(timestamp_hour)
        actual_count.append(timestamp_hour_count_actual.get(timestamp_hour, 0))
        predicted_count.append(timestamp_hour_count_predicted.get(timestamp_hour, 0))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.plot(dates, actual_count)
    plt.plot(dates, predicted_count)
    plt.gcf().autofmt_xdate()
    plt.legend(['actual', 'predicted (on model trained till last day)'], loc='upper left')
    plt.ylabel('# of queries / hour')
    plt.show()


def calculate_mean_square_error(result, top_clusters):
    ret = 0
    for top_cluster in top_clusters:
        top_cluster_results = result[top_cluster]

        actual = top_cluster_results.get("actual")
        predicted = top_cluster_results.get("predicted")

        y = np.array(actual)
        y_hat = np.array(predicted)

        data_min = 2 - np.min([np.min(y), np.min(y_hat)])
        se = (np.log(y + data_min) - np.log(y_hat + data_min)) ** 2
        ret = ret + np.mean(se)

    return ret / len(top_clusters)


def plot_accuracies(accuracy, horizon):
    # Sample data
    categories = ['LAR', 'LSTM', 'Ensemble', 'Kernel estimator']
    values = []

    for a in accuracy:
        values.append(round(a, 3))

    # Define soft colors for the bars
    colors = ['#6a9fc5', '#8abf7f', '#e78ac3', '#f2a35e']

    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Plot the bar graph with custom colors and no grid lines
    bars = ax.bar(categories, values, color=colors, edgecolor='black')

    # Customizing the plot
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('MSE (log space)', fontsize=12)
    plt.title('Prediction horizon ' + horizon, fontsize=14)
    plt.grid(False)  # Turn off the grid lines

    # Adding labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Removing top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.tight_layout()  # To prevent labels from getting cut off
    plt.show()


if __name__ == '__main__':
    file_paths = ["result_lr_60.json", "result_lstm_60.json", "result_ensemble_60.json", "result_kr_60.json"]
    result_error = []
    for path in file_paths:
        # plot(read_json(path), "76")
        result_error.append(calculate_mean_square_error(read_json(path), ["70", "76", "53", "19", "15"]))

    print(result_error)
    plot_accuracies(result_error, "60 mins")

    file_paths = ["result_lr_2880.json", "result_lstm_2880.json", "result_ensemble_2880.json", "result_kr_2880.json"]
    result_error = []
    for path in file_paths:
        # plot(read_json(path), "76")
        result_error.append(calculate_mean_square_error(read_json(path), ["70", "76", "53", "19", "15"]))

    print(result_error)
    plot_accuracies(result_error, "2 days")


    # file_path = "result_ensemble_60.json"
    # plot(read_json(file_path), "70")
    # print(calculate_mean_square_error(read_json(file_path), "70"))
    # plot(read_json(file_path), "76")
    # print(calculate_mean_square_error(read_json(file_path), "76"))
    #
    # file_path = "result_lr_60.json"
    # plot(read_json(file_path), "70")
    # print(calculate_mean_square_error(read_json(file_path), "70"))
    # plot(read_json(file_path), "76")
    # print(calculate_mean_square_error(read_json(file_path), "76"))
    #
    # file_path = "result_kr_60.json"
    # plot(read_json(file_path), "70")
    # print(calculate_mean_square_error(read_json(file_path), "70"))
    # plot(read_json(file_path), "76")
    # print(calculate_mean_square_error(read_json(file_path), "76"))

    # file_path = "result__60.json"
    # plot(read_json(file_path), "76")
    # print(calculate_mean_square_error(read_json(file_path), "76"))

