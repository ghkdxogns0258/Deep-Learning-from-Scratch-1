import numpy as np                    
import matplotlib.pylab as plt          
from gradient_2d import numerical_gradient 

# --------------------------------------
# 경사하강법(Gradient Descent) 함수 정의
# --------------------------------------
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    경사하강법을 통해 함수 f의 최솟값을 찾는 함수
    :param f: 최적화할 함수 (목표 함수)
    :param init_x: 초기값 (x0, x1 등 초기 좌표)
    :param lr: 학습률 (learning rate)
    :param step_num: 반복할 스텝 수
    :return: 최종 x 값과, 이동 경로 기록 (x_history)
    """

    x = init_x                  # 현재 위치 (초기값으로 시작)
    x_history = []              # 이동 경로 기록을 위한 리스트

    # 지정한 스텝 수만큼 반복
    for i in range(step_num):
        # 현재 위치를 기록 (경로 시각화를 위해 저장)
        x_history.append(x.copy())

        # 수치 미분으로 기울기(gradient) 계산
        grad = numerical_gradient(f, x)

        # 경사하강법 핵심 → 기울기의 반대 방향으로 학습률 만큼 이동
        x -= lr * grad           # x = x - lr * grad (값 업데이트)

    # 최종 위치 x와 이동 경로 기록 반환
    return x, np.array(x_history)

# --------------------------------------
# 최적화 대상 함수 정의
# f(x, y) = x^2 + y^2 → 최솟값은 (0, 0)
# --------------------------------------
def function_2(x):
    return x[0]**2 + x[1]**2    # 2차원 입력 벡터를 받아서 두 값의 제곱합 반환

# --------------------------------------
# 초기값 및 하이퍼파라미터 설정
# --------------------------------------
init_x = np.array([-3.0, 4.0])  # 초기값 (x0=-3.0, x1=4.0)

lr = 0.1                        # 학습률 (learning rate)
step_num = 20                   # 반복할 스텝 수

# --------------------------------------
# 경사하강법 실행
# --------------------------------------
x, x_history = gradient_descent(
    function_2,                 # 최적화할 함수
    init_x,                     # 초기값
    lr=lr,                      # 학습률
    step_num=step_num           # 반복할 스텝 수
)

# --------------------------------------
# 결과 시각화
# --------------------------------------

# 기준선(보조선)을 그려줌 → x축과 y축 (원점을 중심으로 표시)
plt.plot([-5, 5], [0, 0], '--b')   # x축 (y=0)
plt.plot([0, 0], [-5, 5], '--b')   # y축 (x=0)

# x_history에 저장된 경로를 시각화
# 각 지점은 경사하강법이 이동한 좌표를 의미함
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

# x축과 y축 범위 지정
plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)

# 축 이름 지정
plt.xlabel("X0")  # x축은 첫 번째 변수
plt.ylabel("X1")  # y축은 두 번째 변수

# 그래프 출력
plt.show()
