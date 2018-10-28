import numpy as np
import pandas as pd
import tensorflow as tf


def get_goodletrace_data(path, aspects):
    names = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId", "meanCPUUsage", "canonical memory usage",
             "AssignMem", "unmapped_cache_usage", "page_cache_usage", "max_mem_usage", "mean_diskIO_time",
             "mean_local_disk_space", "max_cpu_usage", "max_disk_io_time", "cpi", "mai", "sampling_portion", "agg_type",
             "sampled_cpu_usage"]
    df = pd.read_csv(path, names=names)
    data = df.loc[:, aspects].values

    a_max = np.amax(data, axis=0)
    a_min = np.amin(data, axis=0)

    # print(a_max, a_min)

    normalized_data = (data - a_min)/(a_max - a_min)
    # print(normalized_data[0:100])

    return normalized_data, a_max, a_min


def get_data_samples(data, n_slidings, predicted_aspect, rate):

    sliding = n_slidings
    n_samples = data.shape[0]
    n_aspects = data.shape[1]

    data_samples = n_samples - sliding
    data_feed = np.zeros((data_samples, sliding, n_aspects))

    for i in range(data_samples):
        a = i
        b = i + sliding
        data_point = data[a:b, :]
        data_feed[i] += data_point

    n_test = int(data_samples/(rate+1))
    n_train = int(data_samples - n_test)

    x_train = data_feed[0:n_train, :, :].reshape((n_train, sliding, n_aspects))
    x_test = data_feed[n_train:data_samples, :, :].reshape((n_test, sliding, n_aspects))

    y_feed = data[sliding:n_samples, :]
    if predicted_aspect == "meanCPUUsage":
        y_train = y_feed[0:n_train, 0].reshape((n_train, 1))
        y_test = y_feed[n_train:data_samples, 0].reshape((n_test, 1))

    if predicted_aspect == "canonical memory usage":
        y_train = y_feed[0:n_train, 1].reshape((n_train, 1))
        y_test = y_feed[n_train:data_samples, 1].reshape((n_test, 1))

    # print(x_train[0:10])
    # print(y_train[0:10])
    return x_train, y_train, x_test, y_test


path = "../Data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv"
aspects = ["meanCPUUsage", "canonical memory usage"]
predicted_aspect = "meanCPUUsage"
n_slidings_encoder = 10
n_slidings_decoder = 5
rate = 4

nor_data, amax, amin = get_goodletrace_data(path, aspects)
x_train_encoder, y_train, x_test, y_test = get_data_samples(nor_data, n_slidings_encoder, predicted_aspect, rate)


def x_decoder(data_encoder, n_slidings_decoder):
    n_slidings_encoder = data_encoder.shape[1]
    a = data_encoder.shape[0]
    b = data_encoder.shape[2]

    data_decoder = np.zeros((a, n_slidings_decoder, b))

    for i in range(a):
        data_point = x_train_encoder[i]
        gap = n_slidings_encoder - n_slidings_decoder
        point_decoder = data_point[n_slidings_decoder:n_slidings_encoder, :].reshape((gap, b))
        print(point_decoder.shape)
        data_decoder[i] += point_decoder

    return data_decoder


x_train_decoder = x_decoder(x_train_encoder, n_slidings_decoder)