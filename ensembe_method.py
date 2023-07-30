import json

import numpy as np

from utils import read_json


def generate_data(actual_1, predicted_1, actual_2, predicted_2):
    r = [actual_1, actual_2]
    r_hat = [predicted_1, predicted_2]
    data_actual = np.mean(np.array(r), axis=0)
    r_hat = np.array(r_hat)
    data_min = 2 - np.min(r_hat)
    avg = np.log(r_hat + data_min)
    data_ensemble = np.exp(np.mean(avg, axis=0)) - data_min
    return data_actual, data_ensemble


def generate_ensemble(model_path_1, model_path_2):
    result = {}

    model_result_1 = read_json(model_path_1)
    model_result_2 = read_json(model_path_2)

    for cluster in model_result_1:
        predicted_dates = model_result_1[cluster].get("predictedDates")

        actual_1 = np.array(model_result_1[cluster].get("actual"))
        predicted_1 = np.array(model_result_1[cluster].get("predicted"))

        actual_2 = np.array(model_result_2[cluster].get("actual"))
        predicted_2 = np.array(model_result_2[cluster].get("predicted"))

        ensemble_actual, ensemble_predicted = generate_data(actual_1, predicted_1, actual_2, predicted_2)
        result[cluster] = {
            "predictedDates": predicted_dates,
            "actual": ensemble_actual,
            "predicted": ensemble_predicted
        }

    return result


if __name__ == '__main__':
    result = generate_ensemble("result_lr_2880.json", "result_lstm_2880.json")
    result_2 = {}
    for c in result:
        result_2[c] = {}
        result_2[c]["predictedDates"] = result[c]["predictedDates"]
        result_2[c]["actual"] = []
        result_2[c]["predicted"] = []

        for i in range(0, len(result[c]["actual"])):
            result_2[c]["actual"].append(float(result[c]["actual"][i]))
            result_2[c]["predicted"].append(float(result[c]["predicted"][i]))

    with open("result_ensemble_2880.json", "w") as f:
        f.write(json.dumps(result_2))
