import tensorflow as tf
from MNISTData import MNISTData
from AutoEncoder import AutoEncoder
import numpy as np

if __name__ == "__main__":
    print("Hi. I am an AutoEncoder Tester.")
    batch_size = 32
    num_epochs = 5

    data_loader = MNISTData()
    data_loader.load_data()
    x_train = data_loader.x_train
    x_test_noisy = MNISTData.add_noise(data_loader.x_test)
    input_output_dim = data_loader.in_out_dim

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    load_path = "./model/ae_model.weights.h5"
    print("load model weights from %s" % load_path)
    auto_encoder.load_weights(load_path)

    # print for test
    num_test_items = 56
    test_data = data_loader.x_test[0:num_test_items, :]
    test_label = data_loader.y_test[0:num_test_items]
    test_data_x_print = test_data.reshape(num_test_items, data_loader.width, data_loader.height)

    print("const by codes")
    print("x_test_noisy shape:", x_test_noisy.shape)

    reconst_data = auto_encoder.en_decoder.predict(x_test_noisy)
    print("reconst_data shape:", reconst_data.shape)

    reconst_data_subset = reconst_data[:num_test_items]  # 56개만 추출
    reconst_data_x_print = reconst_data_subset.reshape(num_test_items, data_loader.width, data_loader.height)
    reconst_data_x_print = tf.math.sigmoid(reconst_data_x_print)
    MNISTData.print_56_pair_images(test_data_x_print, reconst_data_x_print, test_label)




    print("const by code means for each digit")
    avg_codes = np.zeros([10, 32])
    avg_add_cnt = np.zeros([10])

    latent_vecs = auto_encoder.encoder.predict(x_test_noisy)
    for i, label in enumerate(test_label):
        avg_codes[label] += latent_vecs[i]
        avg_add_cnt[label] += 1.0

    for i in range(10):
        if avg_add_cnt[i] > 0.1:
            avg_codes[i] /= avg_add_cnt[i]

    avg_code_tensor = tf.convert_to_tensor(avg_codes)
    reconst_data_by_vecs = auto_encoder.decoder.predict(avg_code_tensor)
    reconst_data_x_by_mean_print = reconst_data_by_vecs.reshape(10, data_loader.width, data_loader.height)
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    MNISTData.print_10_images(reconst_data_x_by_mean_print, label_list)
    print("reconst_data shape:", reconst_data.shape)