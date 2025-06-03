import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from AutoEncoder import AutoEncoder
from MNISTData import MNISTData

# 1. MNIST 데이터 로드
data_loader = MNISTData()
(_, _), (x_test, y_test) = data_loader.get_data()

# 2. 학습된 AutoEncoder 모델 로드
ae = AutoEncoder()
ae.load_weights("./model/ae_model.weights.h5")  # 저장된 모델 폴더 이름

# 3. 테스트 데이터 인코딩 (latent vector 얻기)
encoded = ae.encoder.predict(x_test)  # shape: (10000, latent_dim)

# 4. 숫자별 평균 latent vector 계산
mean_vectors = []
labels = []

for digit in range(10):
    digit_vectors = encoded[y_test == digit]
    mean_vector = np.mean(digit_vectors, axis=0)
    mean_vectors.append(mean_vector)
    labels.append(str(digit))

mean_vectors = np.array(mean_vectors)  # shape: (10, latent_dim)

# 5. T-SNE로 2차원 차원 축소
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced = tsne.fit_transform(mean_vectors)  # shape: (10, 2)

# 6. 시각화
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    x, y = reduced[i]
    plt.scatter(x, y, label=label)
    plt.text(x + 0.2, y + 0.2, label, fontsize=12)

plt.title("T-SNE Visualization of Class-wise Mean Latent Vectors")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.legend()
plt.grid(True)
plt.show()
