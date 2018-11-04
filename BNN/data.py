import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path


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


def get_data_decoder(x_train_encoder, x_test_encoder, n_slidings_decoder):
    n_slidings_encoder = x_train_encoder.shape[1]
    a = n_slidings_encoder - n_slidings_decoder
    x_train_decoder = x_train_encoder[:, a:n_slidings_encoder, :]
    x_test_decoder = x_test_encoder[:, a:n_slidings_encoder, :]
    return x_train_decoder, x_test_decoder


def getValidationSet(x_train_encoder, x_train_decoder, y_train, n):
    n_train = x_train_encoder.shape[0]
    n_valid = int(n_train/n)
    n_train_new = n_train - n_valid

    x_val_encoder = x_train_encoder[n_train_new:n_train, :, :]
    x_val_decoder = x_train_decoder[n_train_new:n_train, :, :]
    y_val = y_train[n_train_new:n_train, :].reshape((n_valid, 1))

    x_train_encoder_new = x_train_encoder[0:n_train_new, :, :]
    x_train_decoder_new = x_train_decoder[0:n_train_new, :, :]
    y_train_new = y_train[0:n_train_new, :].reshape((n_train_new, 1))

    return x_train_encoder_new, x_train_decoder_new, y_train_new, x_val_encoder, x_val_decoder, y_val


def saveData(combination, loss_test_act, epoch_i, result_file_path, training_encoder_time):
    combination_x = [combination]
    result = {'combination': combination_x,
              'loss': loss_test_act,
              'epoch': epoch_i,
              'training_encoder_time': training_encoder_time}

    df = pd.DataFrame(result)
    if not os.path.exists(result_file_path):
        columns = ['combination', 'loss', 'epoch', 'training_encoder_time']
        df[columns]
        df.to_csv('result_encoder_decoder.csv', index=False, columns=columns)
    else:
        with open('result_encoder_decoder.csv', 'a') as csv_file:
            df.to_csv(csv_file,  mode='a', header=False, index=False)

    name = ''
    name += str(combination)
    name += ' epoch='
    name += str(epoch_i)
    name += ' loss='
    name += str(loss_test_act)
    name += ' training_encoder_time='
    name += str(training_encoder_time)
    print(name)

    return name