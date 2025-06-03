import numpy as np
import matplotlib.pyplot as plt
from AutoEncoder import AutoEncoder
from MNISTData import MNISTData

# 1. AutoEncoder 모델 불러오기
autoencoder = AutoEncoder()
autoencoder.build_model()  # ✅ 이 줄을 꼭 추가해야 함!
autoencoder.load_weights('./model/ae_model.weights.h5')

# 2. MNIST 데이터 로딩
data_loader = MNISTData()
_, (x_test, y_test) = data_loader.get_data()  # ✅ 이제 정상 작동

# 3. 테스트 데이터 중 1000개 선택 (no noise)
x_selected = x_test[:1000]
y_selected = y_test[:1000]

# 4. code 벡터 얻기
codes = autoencoder.encoder.predict(x_selected)

# 5. 숫자별 평균 및 표준편차 계산
class_codes = {i: [] for i in range(10)}
for code, label in zip(codes, y_selected):
    class_codes[label].append(code)

class_avg = {}
class_std = {}
for i in range(10):
    code_vectors = np.array(class_codes[i])
    class_avg[i] = np.mean(code_vectors, axis=0)
    class_std[i] = np.std(code_vectors, axis=0)

# 6. 각 숫자별로 5개의 새로운 코드 생성
generated_images = []
for i in range(10):
    avg = class_avg[i]
    std = class_std[i]
    for _ in range(5):
        rand = np.random.uniform(-1, 1, size=avg.shape)
        new_code = avg + std * rand
        generated_image = autoencoder.decoder.predict(np.array([new_code]))[0]
        generated_images.append(generated_image)

# 7. 결과 시각화
plt.figure(figsize=(10, 5))
for i, img in enumerate(generated_images):
    plt.subplot(10, 5, i + 1)
    plt.imshow(img.reshape(data_loader.width, data_loader.height), cmap='gray')
    plt.axis('off')
plt.suptitle('Generated Images using Avg ± Std * rand')
plt.tight_layout()
plt.show()
