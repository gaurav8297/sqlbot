import json
from datetime import datetime, timedelta, time

import scipy as sp
import torch
from sklearn.ensemble import RandomForestRegressor
from sortedcontainers import SortedDict
import numpy as np
from torch import optim, nn
from torch.autograd import Variable

from models import LSTMModel, LinearModel, KernelRegressionModel
from utils import read_json

MAX_CLUSTER_NUM = 5


# LR + KR + LSTM


# split the data into batches
def batchify(data, batch_size):
    # Dimension of the observation
    n_observe = data.shape[1]
    # Work out how cleanly we can divide the dataset into bsz parts.
    n_batch = data.shape[0] // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:n_batch * batch_size]
    # Evenly divide the data across the bsz batches.
    data = data.reshape(batch_size, -1, n_observe)
    # Transpose the data to fit the model input
    data = data.transpose(1, 0, 2)
    # data.shape = (sequence length, batch size, dim of the observation)
    return data


def load_data(aggregate):
    clusters = read_json("result_clusters.json")
    result = {}
    for cluster in clusters:
        result[int(cluster)] = SortedDict()
        for timestamp_str, query_count in clusters[cluster]:
            timestamp = datetime.fromisoformat(timestamp_str)
            hour = timestamp.hour
            if aggregate > 60:
                hour //= aggregate // 60

            aggr_timestamp = datetime(timestamp.year, timestamp.month, timestamp.day, hour,
                                      timestamp.minute - (timestamp.minute %
                                                          aggregate), 0)
            if aggr_timestamp not in result[int(cluster)]:
                result[int(cluster)][aggr_timestamp] = 0

            result[int(cluster)][aggr_timestamp] += query_count

    return result


def generate_pair(data, horizon, input_dim):
    n = data.shape[0]
    m = data.shape[1]

    x = []
    y = []

    for i in range(n - horizon - input_dim + 1):
        x.append(data[i:i + input_dim].flatten())
        y.append(data[i + input_dim + horizon - 1])

    return np.array(x), np.array(y)


def get_matrix(x):
    xx = x.T.dot(x)
    xx += np.identity(xx.shape[0])
    return np.linalg.inv(xx).dot(x.T)


def training(x, y):
    params = []
    for j in range(y.shape[1]):
        params.append(x.dot(y[:, j]))

    return params


def testing(params, x):
    y_hat = None

    for j in range(len(params)):
        y = x.dot(params[j])

        y = y.reshape((-1, 1))

        if y_hat is None:
            y_hat = y
        else:
            y_hat = np.concatenate((y_hat, y), axis=1)

    return y_hat


def normalize(data):
    # normalizing data
    data_min = 1 - np.min(data)
    data = np.log(data + data_min)
    data_mean = np.mean(data)
    data -= data_mean
    data_std = np.std(data)
    data /= data_std

    return data, data_min, data_mean, data_std


def get_batch(source, i, bptt, evaluation=False, horizon=1):
    seq_len = min(bptt, source.shape[0] - horizon - i)
    data = source[i:i + seq_len]
    target = source[i + horizon:i + horizon + seq_len]
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()


def pretty_print(description, loss):
    print('=' * 89)
    print('|| ', description, ' || loss {:5.3f}'.format(loss))
    print('=' * 89)


def train_pass(data, model, method, criterion, learning_rate, horizon, regress_dim, batch_size, bptt):
    if method == "lr":
        x, y = generate_pair(data, horizon, regress_dim)
        xx = get_matrix(x)
        model.params = training(xx, y)
        return

    if method == "kr":
        x, y = generate_pair(data, horizon, regress_dim)
        model.data = (x, y)
        return

    if method == "lstm":
        model.train()
        total_loss = 0
        losses = []
        batch_size = max(1, min(batch_size, len(data) // (horizon + bptt)))
        data = batchify(data, batch_size)
        n_data = data.shape[0]
        n_batch = data.shape[1]
        hidden = model.init_hidden(n_batch)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for batch, i in enumerate(range(0, n_data - horizon, bptt)):
            data_batch, targets = get_batch(data, i, bptt, False, horizon)
            input = Variable(torch.Tensor(data_batch.astype(float)))
            targets = Variable(torch.Tensor(targets.astype(float)))
            if hidden is not None:
                repackage_hidden(hidden)
            # optimizer.zero_grad()
            model.zero_grad()
            output, hidden = model(input, hidden)
            loss = criterion(output, targets)

            total_loss += loss.data.numpy()
            losses.append(loss.data.numpy())

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(-learning_rate, p.grad.data)

            if batch % 100 == 0:
                total_loss /= 100
                pretty_print(' lr: ' + str(learning_rate) + ' batches: '
                             + str(batch) + '/' + str(n_data // bptt),
                             total_loss)
                total_loss = 0
        pretty_print('Average Train Loss: ', np.mean(losses))
        return

    if method == "rf":
        x, y = generate_pair(data, horizon, regress_dim)
        model.fit(x, y)
        return


def eval_pass(data, model, method, criterion, horizon, regress_dim, bptt):
    if method == "lr":
        x, y = generate_pair(data, horizon, regress_dim)
        y_hat = testing(model.params, x)
        return np.mean((y - y_hat) ** 2), y, y_hat

    if method == "kr":
        x, y = generate_pair(data, horizon, regress_dim)
        k_x, k_y = model.data

        pairwise_sq_dists = sp.spatial.distance.cdist(x, k_x, 'seuclidean')
        kernel = sp.exp(-pairwise_sq_dists)
        y_hat = kernel.dot(k_y) / np.sum(kernel, axis=1, keepdims=True)

        return np.mean((y - y_hat) ** 2), y, y_hat

    if method == "lstm":
        model.train()
        total_loss = 0
        # Because the prediction must be continuous, we cannot use batch here
        data = batchify(data, 1)
        n_data = data.shape[0]
        n_batch = data.shape[1]
        hidden = model.init_hidden(n_batch)
        y = None
        y_hat = None
        batch = 0
        for batch, i in enumerate(range(0, n_data - horizon, bptt)):
            data_batch, targets = get_batch(data, i, bptt, False, horizon)
            input = Variable(torch.Tensor(data_batch.astype(float)))
            targets = Variable(torch.Tensor(targets.astype(float)))
            if hidden is not None:
                hidden = repackage_hidden(hidden)
            output, hidden = model(input, hidden)

            # Calculations for loss
            loss = criterion(output, targets)

            total_loss += loss.data.numpy()

            if y is None:
                y = targets.data.numpy()
            else:
                y = np.concatenate((y, targets.data.numpy()))

            if y_hat is None:
                y_hat = output.data.numpy()
            else:
                y_hat = np.concatenate((y_hat, output.data.numpy()))

        # transpose the output back to normal order
        y = y.transpose(1, 0, 2).reshape((-1, y.shape[2]))
        y_hat = y_hat.transpose(1, 0, 2).reshape((-1, y_hat.shape[2]))

        return (total_loss / (batch + 1)), y, y_hat

    if method == "rf":
        x, y = generate_pair(data, horizon, regress_dim)
        y_hat = model.predict(x)
        return np.mean((y - y_hat) ** 2), y, y_hat

    return None


def get_model(data, batch_size, method):
    if method == "lstm":
        train = batchify(data, batch_size)
        n_tokens = train.shape[2]
        return LSTMModel(n_tokens, ninp=25, nhid=20, nlayers=2, dropout=0.2, tie_weights=False)

    if method == "lr":
        return LinearModel()

    if method == "kr":
        return KernelRegressionModel()

    if method == "rf":
        return RandomForestRegressor(random_state=0)

    return None


def get_multi_data(input, clusters, date, num_days, interval, num_mins, aggregate):
    date_list = [date - timedelta(minutes=x) for x in range(num_days * interval * aggregate,
                                                            -num_mins, -aggregate)]
    result = []
    for date in date_list:
        obs = []
        for c in clusters:
            if c in input:
                data_date = next(input[c].irange(maximum=date, inclusive=(True, False), reverse=True), None)
            else:
                data_date = None
                print("cluster %d is not in input!!!", c)

            if data_date is None:
                data_point = 0
            else:
                data_point = input[c][data_date]
            obs.append(data_point)

        result.append(obs)
    traj = np.array(result)
    return traj


def convert_to_timestamp(timestamp_str):
    return datetime.fromisoformat(timestamp_str)


def predict(input, top_clusters, batch_size, horizon, start_pos, interval, aggregate, max_training_intervals,
            paddling_intervals, epochs, learning_rate, regress_dim, bptt, method):
    result = {}
    model = None
    criterion = nn.MSELoss()
    for date, cluster_list in top_clusters[start_pos // interval:- max(horizon // interval, 1)]:
        first_date = convert_to_timestamp(top_clusters[0][0])
        date = convert_to_timestamp(date)

        train_delta_intervals = min(
            ((date - first_date).days * 1440 + (date - first_date).seconds // 60) // (aggregate * interval),
            max_training_intervals)

        predict_delta_minutes = horizon * aggregate
        print(date, first_date, date + timedelta(minutes=predict_delta_minutes))

        clusters = next(zip(*cluster_list))[:MAX_CLUSTER_NUM]

        data = get_multi_data(input, clusters, date, train_delta_intervals, interval, predict_delta_minutes, aggregate)

        data, data_min, data_mean, data_std = normalize(data)

        print(data.shape)
        train_data = data[:-interval - horizon]
        print(train_data.shape)
        test_data = data[-(paddling_intervals * interval + horizon + interval):]
        print(test_data.shape)

        if model is None or method != "lstm":
            model = get_model(train_data, method=method, batch_size=batch_size)

        # Loop over epochs.
        for epoch in range(1, epochs + 1):
            print('epoch: ', epoch)
            if epoch > 100:
                learning_rate = 0.2
            train_pass(train_data, model, method=method, criterion=criterion, learning_rate=learning_rate,
                       horizon=horizon, regress_dim=regress_dim, batch_size=batch_size, bptt=bptt)
            print('about to evaluate: ')
            val_loss, y, y_hat, = eval_pass(test_data, model, method=method, criterion=criterion,
                                            horizon=horizon, regress_dim=regress_dim, bptt=bptt)
            pretty_print('Validation Loss: Epoch' + str(epoch), np.mean((y[-interval:] -
                                                                         y_hat[-interval:]) ** 2))

        print('about to test')
        val_loss, y, y_hat, = eval_pass(test_data, model, method=method, criterion=criterion,
                                        horizon=horizon, regress_dim=regress_dim, bptt=bptt)

        y = y[-interval:]
        y_hat = y_hat[-interval:]
        pretty_print('Test Loss', np.mean((y - y_hat) ** 2))
        pretty_print('Test Data Variance', np.mean(y ** 2))

        y = np.exp(y * data_std + data_mean) - data_min
        y_hat = np.exp(y_hat * data_std + data_mean) - data_min
        predict_dates = [(date + timedelta(minutes=horizon * aggregate - x)).isoformat() for x in
                         range(interval * aggregate, 0, -aggregate)]

        for i, c in enumerate(clusters):
            if c not in result:
                result[c] = {"predictedDates": [], "actual": [], "predicted": []}
            result[c]["predictedDates"].extend(predict_dates)
            result[c]["actual"].extend(y[:, i])
            result[c]["predicted"].extend(y_hat[:, i])
    return result


if __name__ == '__main__':
    aggregate = 10  # Figure out the meaning
    horizon = 1440 * 2 # 2 days
    start_pos = 14400
    interval = 1440
    BATCH_SIZE = {1: 30, 5: 20, 10: 15, 20: 12, 30: 8, 60: 4, 120: 2}
    REGRESS_DIM = {1: 1440, 5: 288, 10: 144, 20: 72, 30: 48, 60: 24, 120: 12}
    BPTT = {1: 240, 5: 200, 10: 120, 20: 90, 30: 60, 60: 48, 120: 30}

    horizon //= aggregate
    start_pos //= aggregate
    interval //= aggregate

    bptt = BPTT[aggregate]
    batch_size = BATCH_SIZE[aggregate]
    regress_dim = REGRESS_DIM[aggregate]

    input = load_data(aggregate)
    top_clusters = read_json("top_clusters.json")
    result = predict(input, top_clusters, horizon=horizon, start_pos=start_pos,
                     interval=interval, aggregate=aggregate,
                     max_training_intervals=25, paddling_intervals=7, batch_size=batch_size,
                     method="kr", epochs=1, bptt=bptt, learning_rate=0.02,
                     regress_dim=regress_dim)

    result_2 = {}
    for c in result:
        result_2[c] = {}
        result_2[c]["predictedDates"] = result[c]["predictedDates"]
        result_2[c]["actual"] = []
        result_2[c]["predicted"] = []

        for i in range(0, len(result[c]["actual"])):
            result_2[c]["actual"].append(float(result[c]["actual"][i]))
            result_2[c]["predicted"].append(float(result[c]["predicted"][i]))

    with open("result_kr_2880.json", "w") as f:
        f.write(json.dumps(result_2))
