import random

# 카드 요소 배열 선언
x = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4  # 숫자
y = ["H", "D", "S", "C"] * 13  # 무늬

z = []  # 전체 카드 저장 (덱)
output_cards = []  # 각 플레이어의 카드 저장

# 52개 경우의 모든 카드쌍 (13*4) 생성
for i in range(4 * 13):
    z.append(x[i] + y[i])

# 플레이어 수 입력 받기
while True:
    try:
        num_players = int(input("Input number of players [1-5]: "))
        if num_players < 1 or num_players > 5:
            raise ValueError("플레이어 수는 1에서 5 사이여야 합니다.")
        break
    except ValueError as e:
        print(e)

# 메인 게임 루프
while True:
    # 모든 카드쌍 배열에서 비복원추출로 플레이어 수 * 5개 출력 (중복되지 않도록)
    output_cards = random.sample(z, k=num_players * 5)

    # 각 플레이어의 카드 출력
    for player in range(num_players):
        # 각 플레이어의 5장씩 카드 배분
        hand = output_cards[player * 5: (player + 1) * 5]

        # 카드 숫자 중복 확인
        card_numbers = [card[:-1] for card in hand]  # 카드 숫자만 추출 (예: "AC" -> "A")
        has_pair = any(card_numbers.count(number) > 1 for number in card_numbers)  # 숫자 중복 여부 확인

        # 출력 형식 맞추기
        hand_str = ", ".join(hand)  # 쉼표로 구분하여 카드 문자열 생성
        if has_pair:
            print(f"{hand_str} pair")  # 중복이 있는 경우 pair 추가
        else:
            print(hand_str)  # 중복이 없는 경우 그대로 출력

    # 게임 종료 또는 다음 판 시작 선택
    next_action = input("\nPress Enter to continue, or 'z' to quit: ").lower()
    if next_action == 'z':
        print("게임을 종료합니다.")
        break
