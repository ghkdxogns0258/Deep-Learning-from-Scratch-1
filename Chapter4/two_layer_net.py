import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    # 초기화 함수 - 네트워크의 구조와 파라미터 초기화
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        # 첫 번째 가중치(W1) : input_size(입력층 크기) x hidden_size(히든층 크기)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 첫 번째 바이어스(b1) : hidden_size(히든층 크기)
        self.params['b1'] = np.zeros(hidden_size)
        # 두 번째 가중치(W2) : hidden_size(히든층 크기) x output_size(출력층 크기)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 두 번째 바이어스(b2) : output_size(출력층 크기)
        self.params['b2'] = np.zeros(output_size)

    # 예측 함수 : 입력값 x에 대해 예측값 y를 반환
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        # 첫 번째 층의 계산 : 입력 x와 가중치 W1의 행렬 곱과 바이어스 b1 더함
        a1 = np.dot(x, W1) + b1
        # 첫 번째 층에서의 활성화 함수(시그모이드) 적용
        z1 = sigmoid(a1)
        # 두 번째 층의 계산 : 첫 번째 층의 출력 z1과 가중치 W2의 행렬 곱과 바이어스 b2 더함
        a2 = np.dot(z1, W2) + b2
        # 소프트맥스 함수로 출력값 y 계산
        y = softmax(a2)
        
        return y
        
    # 손실 함수 : 예측값 y와 정답 레이블 t를 비교하여 손실값 반환
    def loss(self, x, t):
        y = self.predict(x)  # 예측값 계산
        
        return cross_entropy_error(y, t)  # 크로스 엔트로피 오차 계산
    
    # 정확도 계산 함수 : 예측값 y와 정답 레이블 t의 정확도 계산
    def accuracy(self, x, t):
        y = self.predict(x)  # 예측값 계산
        y = np.argmax(y, axis=1)  # 예측값에서 가장 큰 값의 인덱스를 선택
        t = np.argmax(t, axis=1)  # 정답 레이블에서 가장 큰 값의 인덱스를 선택
        
        # 예측값과 실제 정답이 일치하는 비율 계산
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # 수치 미분을 통한 기울기 계산
    def numerical_gradient(self, x, t):
        # 손실 함수 정의
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        # 각 파라미터에 대해 수치 미분을 통해 기울기 계산
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    # 오차역전파법을 통한 기울기 계산
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]  # 배치 크기 계산
        
        # 순전파 과정 (forward)
        # 첫 번째 층의 계산 : 입력 x와 가중치 W1의 행렬 곱과 바이어스 b1 더함
        a1 = np.dot(x, W1) + b1
        # 첫 번째 층에서 활성화 함수(시그모이드) 적용
        z1 = sigmoid(a1)
        # 두 번째 층의 계산 : 첫 번째 층의 출력 z1과 가중치 W2의 행렬 곱과 바이어스 b2 더함
        a2 = np.dot(z1, W2) + b2
        # 출력층에서 소프트맥스 함수 적용
        y = softmax(a2)
        
        # 역전파 과정 (backward)
        # 출력층에서의 기울기 계산 (y - t) / 배치 크기
        dy = (y - t) / batch_num
        # 두 번째 가중치 W2의 기울기 계산
        grads['W2'] = np.dot(z1.T, dy)
        # 두 번째 바이어스 b2의 기울기 계산
        grads['b2'] = np.sum(dy, axis=0)
        
        # 첫 번째 층에 대한 기울기 계산을 위한 준비
        da1 = np.dot(dy, W2.T)  # W2에 대한 역전파
        dz1 = sigmoid_grad(a1) * da1  # 시그모이드 함수의 미분값을 곱해줌
        # 첫 번째 가중치 W1의 기울기 계산
        grads['W1'] = np.dot(x.T, dz1)
        # 첫 번째 바이어스 b1의 기울기 계산
        grads['b1'] = np.sum(dz1, axis=0)

        return grads