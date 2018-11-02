import tensorflow as tf
import numpy as np
import data as data

def load_model(path):

    sess = tf.Session()

    path = 'model_encoder_decoder'
    encoder_saver = tf.train.import_meta_graph(path + '.meta')
    encoder_graph = tf.get_default_graph()
    X_encoder = encoder_graph.get_tensor_by_name('X_encoder:0')
    outputs_encoder = encoder_graph.get_tensor_by_name('outputs_encoder:0')
    last_output_encoder = outputs_encoder[:, -1, :]
    print(last_output_encoder.shape, outputs_encoder.shape)
    last_output_encoder = tf.reshape(last_output_encoder, (-1, last_output_encoder.shape[1]))
    print(last_output_encoder.shape)
    sliding_encoder = outputs_encoder.shape[1]
    encoder_saver.restore(sess=sess, save_path=path)

path = "../Data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv"
aspects = ["meanCPUUsage", "canonical memory usage"]
predicted_aspect = "meanCPUUsage"
# n_slidings_encoder = [16, 22, 26, 28]
# n_slidings_decoder = [2, 4, 6]
# batch_sizes = [16, 32]
# size_models = [[16], [32], [8, 4], [16, 8]]
# activations = ["tanh", "sigmoid"]
n_slidings_encoder = 32
n_slidings_decoder = 2
batch_sizes = 16
learning_rate = 0.005
num_epochs = 1
size_models = 16
activations = "tanh"
rate = 5
result_file_path = 'result_encoder_decoder.csv'

nor_data, amax, amin = data.get_goodletrace_data(path, aspects)
x_train_encoder, y_train, x_test_encoder, y_test = data.get_data_samples(nor_data, n_slidings_encoder,
                                                                         predicted_aspect, rate)
x_train_decoder, x_test_decoder = data.get_data_decoder(x_train_encoder, x_test_encoder, n_slidings_decoder)
x_train_encoder, x_train_decoder, y_train, x_val_encoder, x_val_decoder, y_val = \
    data.getValidationSet(x_train_encoder, x_train_decoder, y_train, 5)

sess = tf.Session()
path = 'model_encoder_decoder'
encoder_saver = tf.train.import_meta_graph(path + '.meta')
encoder_graph = tf.get_default_graph()
X_encoder = encoder_graph.get_tensor_by_name('X_encoder:0')
outputs_encoder = encoder_graph.get_tensor_by_name('outputs_encoder:0')
last_output_encoder = outputs_encoder[:, -1, :]
print(last_output_encoder.shape, outputs_encoder.shape)
last_output_encoder = tf.reshape(last_output_encoder, (-1, last_output_encoder.shape[1]))
print(last_output_encoder.shape)
sliding_encoder = outputs_encoder.shape[1]
encoder_saver.restore(sess=sess, save_path=path)

outputs_encoder = sess.run(outputs_encoder, feed_dict={X_encoder: x_train_encoder})
print(outputs_encoder.shape)