import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 랜덤으로 5개의 이미지를 불러와서 확인
# s = np.random.choice(np.arange(x_train.shape[0]), 5)
# plt.figure(figsize=(20, 100))
# for cnt, i in enumerate(s):
#     plt.subplot(1, 5, cnt+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i])
#     plt.xlabel(class_names[y_train[i][0]])
# plt.show()

# 데이터 shape와 클래스 개수를 설정
input_shape = x_train.shape[1:]  # (32, 32, 3) == (가로, 세로, RGB(채널))
num_classes = len(np.unique(y_train))  # 정답(label)의 클래스 개수 (10)

# 데이터를 255로 나눠서 전처리
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 모델 설계
model = Sequential()
# Conv2D(필터 개수, 커널 크기, 활성화 함수)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# 풀링 사이즈가 (2,2)이므로, 각 필터의 가로세로는 절반씩 줄어들어, 1/4의 크기가 됩니다.
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# n차원의 출력을 1차원으로 쭉 늘려줍니다. np.reshape(x, (-1))과 동일
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# Dropout층을 통해 오버피팅을 완화
model.add(layers.Dropout(0.2))
# Dense 레이어로 어느정도 분석
model.add(layers.Dense(64, activation='relu'))
# num_classes, 즉 10개의 출력값을 가진 Dense 레이어에 Softmax 함수를 활성화함수로 사용
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# 모델을 요약해서 확인
history = model.fit(x_train, y_train, batch_size=32, epochs=15, validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test)  # 학습 완료 후 검증
print("손실률:", loss)  # 손실률: 0.9346853494644165
print("정확도:", accuracy)  # 정확도: 0.70169997215271

# 모델이 어떻게 예측을 하는지 간단히 확인
# s = np.random.choice(np.arange(x_test.shape[0]), 5)
# preds = model.predict_classes(x_test[s])  # 다중분류이므로, predict_classes
# plt.figure(figsize=(20, 100))
# for cnt, i in enumerate(s):
#     plt.subplot(1, 5, cnt + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_test[i])
#     plt.ylabel("predict: " + class_names[preds[cnt]])
#     plt.xlabel("label: " + class_names[y_test[i].argmax()])
# plt.show()
