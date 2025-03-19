import sys, os  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
import numpy as np 
import matplotlib.pyplot as plt  
from dataset.mnist import load_mnist  
from two_layer_net import TwoLayerNet  

# 데이터 읽기: MNIST 데이터셋을 정규화하고 원-핫 인코딩하여 로드
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 2계층 신경망 생성: 입력 크기 784, 은닉층 크기 50, 출력 크기 10 설정
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터 설정
iters_num = 10000  # 총 반복 횟수 설정 (에포크 수가 아님)
train_size = x_train.shape[0]  # 훈련 데이터의 총 샘플 수 계산
batch_size = 100   # 미니배치 크기 설정 (한 번에 처리할 데이터 샘플 수)
learning_rate = 0.1  # 학습률 설정

# 학습 진행 상황 기록용 리스트 초기화
train_loss_list = []  # 배치별 손실 값을 저장할 리스트
train_acc_list = []  # 에포크별 훈련 정확도를 저장할 리스트
test_acc_list = []  # 에포크별 테스트 정확도를 저장할 리스트

# 1 에포크당 반복 수 계산: 전체 데이터를 배치 크기로 나눈 값 (최소 1 이상)
iter_per_epoch = max(train_size / batch_size, 1)

# 학습 루프 시작: 총 iters_num 만큼 반복
for i in range(iters_num):
    # 미니배치 획득: 훈련 데이터에서 무작위로 배치 크기만큼의 인덱스 선택
    batch_mask = np.random.choice(train_size, batch_size)  # 무작위 인덱스 생성
    x_batch = x_train[batch_mask]  # 선택된 인덱스에 해당하는 입력 데이터 추출
    t_batch = t_train[batch_mask]  # 선택된 인덱스에 해당하는 정답 레이블 추출
    
    # 기울기 계산: 역전파를 통해 미니배치에 대한 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)  # 수치 미분을 통한 기울기 계산 (비효율적)
    grad = network.gradient(x_batch, t_batch)  # 효율적인 역전파 알고리즘을 사용하여 기울기 계산
    
    # 매개변수 갱신: 경사 하강법을 사용하여 각 파라미터를 업데이트
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]  # 학습률과 기울기를 곱해 각 파라미터를 감소시킴
    
    # 학습 경과 기록: 현재 미니배치의 손실값 계산 후 리스트에 저장
    loss = network.loss(x_batch, t_batch)  # 현재 미니배치의 손실값 계산
    train_loss_list.append(loss)  # 손실값을 기록 리스트에 추가
    
    # 1 에포크마다 정확도 계산: 전체 훈련 및 테스트 데이터에 대해 정확도 평가
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)  # 전체 훈련 데이터에 대한 정확도 계산
        test_acc = network.accuracy(x_test, t_test)  # 전체 테스트 데이터에 대한 정확도 계산
        train_acc_list.append(train_acc)  # 계산된 훈련 정확도를 리스트에 추가
        test_acc_list.append(test_acc)  # 계산된 테스트 정확도를 리스트에 추가
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))  # 현재 에포크의 정확도 출력

# 그래프 그리기: 에포크에 따른 훈련 및 테스트 정확도 시각화
x = np.arange(len(train_acc_list))  # x축: 에포크 번호 (정확도 기록 리스트 길이에 따라 생성)
plt.plot(x, train_acc_list, label='train acc')  # 훈련 정확도 곡선 그리기
plt.plot(x, test_acc_list, label='test acc', linestyle='--')  # 테스트 정확도 곡선 그리기 (점선 스타일)
plt.xlabel("epochs")  # x축 라벨 설정
plt.ylabel("accuracy")  # y축 라벨 설정
plt.ylim(0, 1.0)  # y축 범위를 0부터 1까지로 제한
plt.xlim(0, 16)  # x축 범위를 0부터 16까지로 제한
plt.legend(loc='lower right')  # 범례를 오른쪽 아래에 위치시킴
plt.show()  # 그래프 출력