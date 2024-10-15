import random

# 카드 요소 배열 선언
x = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4 # 숫자
y = ["H", "D", "S", "C"] * 13 # 무늬
z = [] # 전체 카드 경우 저장


# 52개 경우의 모든 카드쌍 (13*4) 생성
for i in range(4*13):
    z.append(x[i]+y[i])

#인원수 선택
playerNum = int(input("Input number of players [1 ~ 5] : "))


while 1: #마지막에 엔터 or z 입력에 따른 게임 반복


    for j in range(playerNum):
        outputCard = [] # 선택된 카드 저장
        selectedNum = [] # 선택된 카드의 숫자와 무늬를 분리해서 숫자만 저장
        # 모든 카드쌍 배열에서 비복원추출로 5장 선택
        outputCard = random.sample(z, k=5)

        # 선택 된 카드를 다시 숫자와 무늬로 분리
        # 카드 자체도 문자 배열이고, 마지막 글자만 모양을 의미하기 때문에 
        # 그것을 빼면 숫자만 남게 됨.
        for i in outputCard:
            selectedNum.append(i[:-1])

        """
        # 중간 출력 확인용 print 문
        print(outputCard)
        print(selectedNum)
        print(set(selectedNum))
        """

        # 카드 출력 (제출한 과제 1에서 보완하였습니다.)
        for s in outputCard:
            if s is outputCard[-1]: 
                # 마지막 순서일 경우에만 쉼표 없이 출력
                print(s, end = " ")
                # 집합 함수 set() 활용해서 중복 여부 확인
                if len(selectedNum) != len(set(selectedNum)):
                   print("pair\a")
                else:
                   print("\a")
            # 나머지는 쉼표 달고 출력
            else: 
               print(s, end=", ")

    # while 반복문 내에서만 카드 뽑기 때문에 인원수를 뽑는 과정을 다시 하지 않는다
    nextValue = input() # 엔터 입력 or z 입력 한 것을 저장 할 변수

    if nextValue == "": # 엔터를 누르면 다음판으로 넘어가고
        continue
    elif nextValue == "z": #  z를 누르면 게임을 종료하게 됨.
        break