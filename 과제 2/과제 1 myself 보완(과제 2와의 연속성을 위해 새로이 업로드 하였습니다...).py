import random

# 카드 요소 배열 선언
x = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4 # 숫자
y = ["H", "D", "S", "C"] * 13 # 무늬

z = [] # 전체 경우 저장
output_card = [] # 출력할 카드 저장

# 52개 경우의 모든 카드쌍 (13*4) 생성
for i in range(4*13):
    z.append(x[i]+y[i])
    
# 모든 카드쌍 배열에서 비복원추출로 5개 출력
output_card = random.sample(z, k=5)
for i in output_card:
    if i is output_card[-1]: # 마지막 배열일 경우에만 쉼표 없이 출력
        print(i)
    else: # 나머지는 쉼표 달고 출력
        print(i, end=", ")
