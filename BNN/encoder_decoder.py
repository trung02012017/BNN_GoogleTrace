import numpy as np
import data as data
import tensorflow as tf
import os
import shutil

tf.set_random_seed(1)
np.random.seed(1)


def encoder_decoder(X_encoder, X_decoder, size_model, activation, input_keep_prob, output_keep_prob, state_keep_prob,
                    variational_recurrent=True, input_size=2):

    with tf.variable_scope('encoder'):
        n_layers = len(size_model)
        cells = []

        if activation == "tanh":
            for i in range(n_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=size_model[i])
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     input_keep_prob=input_keep_prob,
                                                     output_keep_prob=output_keep_prob,
                                                     state_keep_prob=state_keep_prob,
                                                     variational_recurrent=variational_recurrent,
                                                     input_size=input_size,
                                                     dtype=tf.float32)
                cells.append(cell)

        if activation == "sigmoid":
            for i in range(n_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=size_model[i], activation=tf.nn.sigmoid)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     input_keep_prob=input_keep_prob,
                                                     output_keep_prob=output_keep_prob,
                                                     state_keep_prob=state_keep_prob,
                                                     variational_recurrent=variational_recurrent,
                                                     input_size=input_size,
                                                     dtype=tf.float32)
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
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     input_keep_prob=input_keep_prob,
                                                     output_keep_prob=output_keep_prob,
                                                     state_keep_prob=state_keep_prob,
                                                     variational_recurrent=variational_recurrent,
                                                     input_size=input_size,
                                                     dtype=tf.float32)
                cells.append(cell)

        if activation == "sigmoid":
            for i in range(n_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=size_model[i], activation=tf.nn.sigmoid)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     input_keep_prob=input_keep_prob,
                                                     output_keep_prob=output_keep_prob,
                                                     state_keep_prob=state_keep_prob,
                                                     variational_recurrent=variational_recurrent,
                                                     input_size=input_size,
                                                     dtype=tf.float32)
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
    num_epochs = 1000
    learning_rate = 0.005
    n_slidings_encoder = [16, 22, 26, 28]
    n_slidings_decoder = [2, 4, 6]
    batch_sizes = [16, 32]
    size_models = [[16], [32], [8, 4], [16, 8]]
    activations = ["tanh", "sigmoid"]
    input_keep_probs = [0.95, 0.9]
    output_keep_probs = [0.9]
    state_keep_probs  = [0.95, 0.9]
    # n_slidings_encoder = [32]
    # n_slidings_decoder = [2]
    # batch_sizes = [16]
    # size_models = [[16]]
    # activations = ["tanh"]
    rate = 5
    result_file_path = 'result_encoder_decoder.csv'

    combinations = []
    for n_sliding_encoder in n_slidings_encoder:
        for n_sliding_decoder in n_slidings_decoder:
            for batch_size in batch_sizes:
                for size_model in size_models:
                    for activation in activations:
                        for input_keep_prob in input_keep_probs:
                            for output_keep_prob in output_keep_probs:
                                for state_keep_prob in state_keep_probs:
                                    combination_i = [n_sliding_encoder, n_sliding_decoder,
                                                     batch_size, size_model, activation,
                                                     input_keep_prob, output_keep_prob,
                                                     state_keep_prob]
                                    combinations.append(combination_i)

    for combination in combinations:

        tf.reset_default_graph()

        n_sliding_encoder = combination[0]
        n_sliding_decoder = combination[1]
        batch_size = combination[2]
        size_model = combination[3]
        activation = combination[4]
        input_keep_prob = combination[5]
        output_keep_prob = combination[6]
        state_keep_prob = combination[7]

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
        X_encoder = tf.placeholder(tf.float32, [None, timestep_encoder, input_dim], name='X_encoder')
        X_decoder = tf.placeholder(tf.float32, [None, timestep_decoder, input_dim], name='X_decoder')
        y = tf.placeholder(tf.float32, [None, 1], name='output')

        output, outputs_encoder, outputs_decoder = encoder_decoder(X_encoder, X_decoder, size_model, activation,
                                                                   input_keep_prob, output_keep_prob, state_keep_prob)
        outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')

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

            outputs_encoder = sess.run(outputs_encoder, feed_dict={X_encoder: x_train_encoder,
                                                 X_decoder: x_train_decoder,
                                                 y: y_train})
            # print(outputs_encoder[:, -1, :].shape)



            # print('\nSaving...')
            cwd = os.getcwd()
            saved_path = 'model/model'
            saved_path += str(combination)
            saved_path += '.ckpt'
            saved_path = os.path.join(cwd, saved_path)
            print(saved_path)
            shutil.rmtree(saved_path, ignore_errors=True)
            saver = tf.train.Saver()
            saver.save(sess=sess, save_path=saved_path)
            # print("ok")

            sess.close()


if __name__ == '__main__':
    main()