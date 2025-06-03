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
    x_test = data_loader.x_test
    y_test = data_loader.y_test
    input_output_dim = data_loader.in_out_dim

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    load_path = "./model/ae_model.weights.h5"
    print("load model weights from %s" % load_path)
    auto_encoder.load_weights(load_path)

    # (1) 1000개 원본 테스트 이미지 중 각 class별 평균 latent vector 구하기
    num_sample = 1000
    x_test_sample = x_test[:num_sample]  # 노이즈 없음 (원본)
    y_test_sample = y_test[:num_sample]

    # Encoder를 통과시켜 latent vector 얻기
    latent_vecs = auto_encoder.encoder.predict(x_test_sample)
    print("latent_vecs shape:", latent_vecs.shape)  # (1000, latent_dim)

    # 평균을 위한 준비
    num_classes = 10
    latent_dim = latent_vecs.shape[1]
    avg_codes = np.zeros((num_classes, latent_dim))
    count_per_class = np.zeros(num_classes)

    for i in range(num_sample):
        label = y_test_sample[i]
        avg_codes[label] += latent_vecs[i]
        count_per_class[label] += 1

    # 평균 계산
    for i in range(num_classes):
        if count_per_class[i] > 0:
            avg_codes[i] /= count_per_class[i]

    # (2) 평균 latent vector를 Decoder에 넣어 이미지 생성
    avg_code_tensor = tf.convert_to_tensor(avg_codes, dtype=tf.float32)
    reconst_from_avg = auto_encoder.decoder.predict(avg_code_tensor)

    # 이미지 출력용 reshape
    reconst_images = reconst_from_avg.reshape(num_classes, data_loader.width, data_loader.height)
    label_list = list(range(10))
    MNISTData.print_10_images(reconst_images, label_list)
