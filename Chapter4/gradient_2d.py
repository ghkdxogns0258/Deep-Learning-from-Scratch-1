import numpy as np                        
import matplotlib.pylab as plt              
from mpl_toolkits.mplot3d import Axes3D    

# ------------------------------------------------------------
# 하나의 벡터(1D 배열)에 대해 수치 미분으로 기울기(gradient)를 계산하는 함수
# ------------------------------------------------------------
def _numerical_gradient_no_batch(f, x):
    h = 1e-4                        # 미분 계산을 위한 아주 작은 값 (오차 최소화)
    grad = np.zeros_like(x)         # x와 같은 크기의 0으로 초기화된 배열 (기울기 저장용)
    
    # 각 요소별 편미분 계산
    for idx in range(x.size):
        tmp_val = x[idx]            # 원래 값을 임시로 저장
        
        # x[idx] + h 인 경우의 함수 값 계산 (전진 차분)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # x[idx] - h 인 경우의 함수 값 계산 (후진 차분)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        # 중심 차분으로 편미분 근사값 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        
        # 원래 값으로 복원 (다음 idx 계산을 위해)
        x[idx] = tmp_val
        
    return grad                     # 계산된 편미분(기울기) 반환


# ------------------------------------------------------------
# 입력이 벡터 하나면 단일 기울기 반환 / 여러 벡터면 각 기울기들을 반환
# ------------------------------------------------------------
def numerical_gradient(f, X):
    # X가 1차원 벡터라면 → 바로 계산 후 반환
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    
    # X가 2차원 이상이면 → 각 행마다 기울기를 계산하고 저장
    else:
        grad = np.zeros_like(X)     # X와 같은 크기의 0배열을 만들어 저장소 준비
        
        for idx, x in enumerate(X): # X의 각 벡터를 하나씩 꺼내서 기울기를 계산
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad                 # 모든 벡터에 대한 기울기 반환


# ------------------------------------------------------------
# 예시로 사용되는 2차원 함수 (기울기 시각화 대상)
# 입력: x0, x1 / 출력: x0^2 + x1^2
# ------------------------------------------------------------
def function_2(x):
    # x가 1차원 벡터일 경우 → 각 요소의 제곱 합을 반환
    if x.ndim == 1:
        return np.sum(x ** 2)
    
    # x가 2차원 배열일 경우 → 각 행(벡터)마다 제곱합 반환
    else:
        return np.sum(x ** 2, axis=1)


# ------------------------------------------------------------
# 특정 점에서 접선 방정식(직선)을 반환하는 함수
# ------------------------------------------------------------
def tangent_line(f, x):
    d = numerical_gradient(f, x)    # 기울기(gradient) 계산
    
    print(d)                        # 기울기 값 출력 (디버깅용)
    
    # 접선의 y절편 계산: y = f(x) - 기울기 * x
    y = f(x) - d * x
    
    # 람다 함수를 반환: 입력값 t를 받아서 접선 위 점을 계산
    return lambda t: d * t + y


# ============================================================
# 메인 실행부 (직접 실행할 때만 동작)
# ============================================================
if __name__ == '__main__':
    
    # x0, x1 좌표값 생성: -2 ~ 2 범위에서 0.25 간격
    x0 = np.arange(-2, 2.5, 0.25)   # x축 값들
    x1 = np.arange(-2, 2.5, 0.25)   # y축 값들
    
    # 좌표값 그리드 생성 → (X는 x0 좌표 매트릭스, Y는 x1 좌표 매트릭스)
    X, Y = np.meshgrid(x0, x1)
    
    # meshgrid 결과를 1차원 배열로 평탄화 → 이후 벡터 필드로 사용할 준비
    X = X.flatten()                 # x 좌표
    Y = Y.flatten()                 # y 좌표
    
    # 각 좌표에 대한 기울기(gradient)를 계산
    grad = numerical_gradient(function_2, np.array([X, Y]))
    # grad[0] → x축 방향의 기울기
    # grad[1] → y축 방향의 기울기
    
    # ---------------------------
    # 기울기 벡터를 시각화 (quiver plot)
    # ---------------------------
    plt.figure()                    # 새로운 그래프 창 생성
    
    # quiver를 사용하여 벡터(화살표)를 그림
    # (X, Y) 좌표에서 시작해 (-grad[0], -grad[1]) 방향으로 화살표를 그림
    plt.quiver(X, Y,               # 화살표의 시작점 좌표
               -grad[0], -grad[1], # 화살표의 방향 (음수로 뒤집어서 하강 방향)
               angles="xy",         # 축 기준으로 방향 설정
               color="#666666")     # 화살표 색상 (회색)
    
    # ---------------------------
    # 그래프 설정 및 출력
    # ---------------------------
    plt.xlim([-2, 2])               # x축 범위 지정
    plt.ylim([-2, 2])               # y축 범위 지정
    
    plt.xlabel('x0')                # x축 이름
    plt.ylabel('x1')                # y축 이름
    
    plt.grid()                      # 배경에 격자 표시
    
    plt.legend()                    # 범례 
    
    plt.draw()                      # 그래프 렌더링
    plt.show()                      # 그래프 출력
