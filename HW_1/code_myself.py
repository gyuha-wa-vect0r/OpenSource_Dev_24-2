import random

# 카드 요소 배열 선언
x = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4 #숫자
y = ["H", "D", "S", "C"] * 13 #무늬
z = [] #랜덤 5개 고른 쌍 저장

# 52개 경우의 모든 카드쌍 (13*4) 생성
for i in range(4*13):
    z.append(x[i]+y[i])
    
# 모든 카드쌍 배열에서 비복원추출로 5개 출력
print(random.sample(z, k=5)) 