import numpy as np
import data as data
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)


def encoder_decoder(X_encoder, X_decoder, size_model, activation):

    with tf.variable_scope('encoder'):
        n_layers = len(size_model)
        cells = []

        if activation == "tanh":
            for i in range(n_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=size_model[i])
                cells.append(cell)

        if activation == "sigmoid":
            for i in range(n_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=size_model[i], activation=tf.nn.sigmoid)
                cells.append(cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        outputs_encoder, state_encoder = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=X_encoder,
            dtype=tf.float32,
        )
    with tf.variable_scope('decoder'):
        n_layers = len(size_model)
        cells = []
        if activation == "tanh":
            for i in range(n_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=size_model[i])
                cells.append(cell)

        if activation == "sigmoid":
            for i in range(n_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=size_model[i], activation=tf.nn.sigmoid)
                cells.append(cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        outputs_decoder, state_decoder = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=X_decoder,
            dtype=tf.float32,
            initial_state=state_encoder
        )

    output = tf.layers.dense(outputs_decoder[:, -1, :], 1)

    return output, outputs_encoder, outputs_decoder
    # return outputs_encoder, outputs_decoder


def main():
    path = "../Data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv"
    aspects = ["meanCPUUsage", "canonical memory usage"]
    predicted_aspect = "meanCPUUsage"
    n_slidings_encoder = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_slidings_decoder = [5, 6, 7, 8, 9]
    batch_sizes = [16, 32]
    learning_rate = 0.005
    num_epochs = 10
    size_models = [[16], [32], [8, 4], [16, 8]]
    activations = ["tanh", "sigmoid"]
    rate = 5

    result_file_path = 'result_encoder_decoder.csv'

    combinations = []
    for n_sliding_encoder in n_slidings_encoder:
        for n_sliding_decoder in n_slidings_decoder:
            for batch_size in batch_sizes:
                for size_model in size_models:
                    for activation in activations:
                        combination_i = [n_sliding_encoder, n_sliding_decoder, batch_size, size_model, activation]
                        combinations.append(combination_i)

    for combination in combinations:

        tf.reset_default_graph()

        n_sliding_encoder = combination[0]
        n_sliding_decoder = combination[1]
        batch_size = combination[2]
        size_model = combination[3]
        activation = combination[4]

        nor_data, amax, amin = data.get_goodletrace_data(path, aspects)
        x_train_encoder, y_train, x_test_encoder, y_test = data.get_data_samples(nor_data, n_sliding_encoder,
                                                                                 predicted_aspect, rate)
        x_train_decoder, x_test_decoder = data.get_data_decoder(x_train_encoder, x_test_encoder, n_sliding_decoder)
        x_train_encoder, x_train_decoder, y_train, x_val_encoder, x_val_decoder, y_val = \
            data.getValidationSet(x_train_encoder, x_train_decoder, y_train, 5)
        # print(x_train_encoder.shape, x_train_decoder.shape, y_train.shape, x_val_encoder.shape, x_val_decoder.shape,
        #       y_val.shape)
        # return 0

        loss_train_value = []
        loss_valid_value = []

        n_train = y_train.shape[0]
        num_batches = int(x_train_encoder.shape[0] / batch_size)

        timestep_encoder = n_sliding_encoder
        timestep_decoder = n_sliding_decoder
        input_dim = len(aspects)
        X_encoder = tf.placeholder(tf.float32, [None, timestep_encoder, input_dim], name='encoder')
        X_decoder = tf.placeholder(tf.float32, [None, timestep_decoder, input_dim], name='decoder')
        y = tf.placeholder(tf.float32, [None, 1])

        output, outputs_encoder, outputs_decoder = encoder_decoder(X_encoder, X_decoder, size_model, activation)

        loss = tf.reduce_mean(tf.squared_difference(output, y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            pre_loss_valid = 100
            x = 0
            early_stopping_val = 5
            for i in range(num_epochs):
                num_epochs_i = i + 1
                for j in range(num_batches + 1):
                    a = batch_size * j
                    b = a + batch_size
                    if b > n_train:
                        b = n_train
                    x_batch_encoder = x_train_encoder[a:b, :, :]
                    x_batch_decoder = x_train_decoder[a:b, :, :]
                    y_batch = y_train[a:b, :]
                    # print(x_batch.shape, y_batch.shape)

                    loss_j, _ = sess.run([loss, optimizer], feed_dict={X_encoder: x_batch_encoder,
                                                                       X_decoder: x_batch_decoder,
                                                                       y: y_batch})
                loss_train_i = sess.run(loss, feed_dict={X_encoder: x_train_encoder,
                                                         X_decoder: x_train_decoder,
                                                         y: y_train})
                loss_valid_i = sess.run(loss, feed_dict={X_encoder: x_val_encoder,
                                                         X_decoder: x_val_decoder,
                                                         y: y_val})
                # print(num_epochs_i, loss_train_i, loss_valid_i)

                loss_train_value.append(loss_train_i)
                loss_valid_value.append(loss_valid_i)

                if loss_valid_i > pre_loss_valid:
                    x = x+1
                    if x == early_stopping_val:
                        break
                else:
                    x = 0
                pre_loss_valid = loss_valid_i

            output_test = sess.run(output, feed_dict={X_encoder: x_test_encoder,
                                                      X_decoder: x_test_decoder,
                                                      y: y_test})
            output_test = output_test * (amax[0] - amin[0]) + amin[0]
            y_test_act = y_test * (amax[0] - amin[0]) + amin[0]

            loss_test_act = np.mean(np.abs(output_test - y_test_act))
            # print(loss_test_act)

            name = data.saveData(combination, loss_test_act, num_epochs_i, result_file_path)


if __name__ == '__main__':
    main()