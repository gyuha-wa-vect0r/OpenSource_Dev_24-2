import osmnx as ox  # OpenStreetMap 데이터를 활용하여 네트워크 그래프를 생성하는 라이브러리
import heapq  # 다익스트라 알고리즘을 위해 우선순위 큐를 사용하는 heapq 라이브러리
import tracemalloc  # 메모리 사용량을 측정하기 위한 라이브러리
import time  # 실행 시간을 측정하기 위한 라이브러리


############################################
# Step 1: 데이터 준비 및 네트워크 그래프 생성 #
############################################

# 청주 중심 좌표와 반경으로 도로망 가져오기
# 시계탑 오거리을 기준으로 반경 6KM 지역의 자동차 도로망 데이터를 가져옴
print("도로망 그래프 생성 중...")
def get_osm_network_from_point(center_point=(36.63547, 127.46891), dist=6000):
    # OpenStreetMap 데이터를 이용해 특정 지점 주변의 도로망을 그래프 형태로 생성
    G = ox.graph_from_point(center_point, dist=dist, network_type="drive")  # 도보에 적당하지 않은 거리로 굴릴거라 자동차 도로망만 선택
    print(f"{G.number_of_nodes()}개의 노드들과 and {G.number_of_edges()}개의 엣지들이 모였습니다!")  # 노드와 엣지 수를 출력
    return G  # 생성된 도로망 그래프를 반환

# 네트워크를 가져와서 시각화하는 과정
network = get_osm_network_from_point()  # 위 함수로 도로망 데이터를 가져옴
ox.plot_graph(network,  # 그래프 시각화
              node_size=3,  # 노드의 크기 설정
              edge_linewidth=0.5,  # 엣지의 두께 설정
              edge_color="gray"  # 엣지의 색상을 회색으로 설정
              )
print(f"노드의 수: {network.number_of_nodes()}")  # 전체 노드 수 출력
print(f"엣지의 수: {network.number_of_edges()}")  # 전체 엣지 수 출력

# 노드와 엣지 데이터를 GeoDataFrame 형태로 추출
nodes, edges = ox.graph_to_gdfs(network)  # 노드와 엣지 데이터를 GeoDataFrame으로 변환
nodes.head(8000)  # 노드 데이터 중 일부 데이터를 확인. (총 8000개의 노드 데이터 확인)
edges.head(22000)  # 엣지 데이터 중 일부 데이터를 확인. (총 22000개의 엣지 데이터 확인)

# 중복 노드 검증 함수
print("노드들의 중복 검증 중...")
def check_duplicate_nodes(nodes):
    # 'x', 'y' 좌표 기준으로 중복된 노드가 있는지 확인합니다.
    duplicate_nodes = nodes.duplicated(subset=['x', 'y']).sum()  # 'x', 'y' 좌표가 중복된 노드를 찾습니다.
    if duplicate_nodes > 0:  # 만약 중복된 노드가 있다면
        print(f"중복 노드를 발견했어요! : {duplicate_nodes}.")  # 중복된 노드의 개수를 출력합니다.
    else:  # 중복된 노드가 없을 경우
        print("데이터 내 좌표에 대한 중복 데이터는 존재하지 않아요.")  # 중복된 노드가 없다고 출력합니다.

# 중복 엣지 검증 함수
def check_duplicate_edges(edges):
    # 'geometry'와 'length'를 기준으로 중복된 엣지가 있는지 확인
    duplicate_edges = edges.duplicated(subset=['geometry', 'length']).sum()  # 'geometry'와 'length'가 중복된 엣지를 찾기
    if duplicate_edges > 0:  # 만약 중복된 엣지가 있다면
        print(f"'geometry' and 'length' 파라미터에 따라 중복 엣지를 발견했어요! : {duplicate_edges}")  # 중복된 엣지의 개수를 출력
    else:  # 중복된 엣지가 없을 경우
        print("'geometry' and 'length' 파라미터에 따라 중복 엣지는 존재하지 않아요.")  # 중복된 엣지가 없다고 출력

# 중복 노드 및 엣지 검증 실행
check_duplicate_nodes(nodes)  # 중복 노드 검증 함수 실행
check_duplicate_edges(edges)  # 중복 엣지 검증 함수 실행


#############################################################
# Step 2: Dijkstra algorithm 및 Bellman-Ford Algorithm 구현 #
#############################################################

# 출발지와 도착지 좌표 설정
start_coords = (36.617646, 127.517828)  # 출발지 좌표: 동남지구 올리브영 사거리
end_coords = (36.666138, 127.453691)  # 도착지 좌표: 강서2동 행정복지센터

# 그래프에서 출발지와 도착지에 가장 가까운 노드를 검색
start_node = ox.distance.nearest_nodes(network, start_coords[1], start_coords[0])  # 출발지 근처의 가장 가까운 노드 검색
end_node = ox.distance.nearest_nodes(network, end_coords[1], end_coords[0])  # 도착지 근처의 가장 가까운 노드 검색

print(f"출발지의 노드 ID: {start_node}, 도착지의 노드 ID: {end_node}")  # 찾은 출발지와 도착지의 노드 ID를 출력

# 다익스트라 알고리즘 정의
def dijkstra_algorithm(graph, start_node, end_node):
    # 최단 거리 테이블 초기화
    distances = {node: float('inf') for node in graph.nodes}  # 모든 노드에 대해 최단 거리를 무한대로 초기화
    predecessors = {node: None for node in graph.nodes}  # 경로 추적을 위한 이전 노드를 저장
    distances[start_node] = 0  # 시작 노드의 거리를 0으로 설정

    # 우선순위 큐 초기화 (힙 구조 사용)
    priority_queue = [(0, start_node)]  # (현재 거리, 노드) 형식으로 시작 노드를 우선순위 큐에 삽입

    while priority_queue:
        # 우선순위 큐에서 가장 짧은 거리의 노드를 추출
        current_distance, current_node = heapq.heappop(priority_queue)

        # 이미 처리된 노드라면 pass
        if current_distance > distances[current_node]:
            continue

        # 인접 노드 탐색
        for neighbor, edge_data in graph[current_node].items():
            weight = edge_data[0]['length']  # 엣지의 길이를 가중치로 사용
            new_distance = current_distance + weight  # 새로운 거리를 계산함

            # 더 짧은 경로를 발견한 경우 거리 갱신
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance  # 최단 거리를 갱신
                predecessors[neighbor] = current_node  # 이전 노드 갱신
                heapq.heappush(priority_queue, (new_distance, neighbor))  # 갱신된 노드를 우선순위 큐에 추가

    # 최단 경로를 역추적
    path = []
    current_node = end_node
    while current_node is not None:  # 도착지부터 출발지까지 이전 노드를 추적하며 경로를 생성
        path.append(current_node)
        current_node = predecessors[current_node]

    return path[::-1], distances[end_node]  # 최단 경로와 해당 경로의 거리를 반환

print("다익스트라 알고리즘을 통해 최단 경로 생성 중...")
# 다익스트라 알고리즘 실행 및 성능 측정
start_time = time.time()  # 현재 시간을 기록하여 실행 시간을 측정
dijkstra_path, dijkstra_distance = dijkstra_algorithm(network, start_node, end_node)  # 다익스트라 알고리즘 실행
dijkstra_time = time.time() - start_time  # 실행이 끝난 후 걸린 시간을 계산

# 다익스트라 알고리즘 결과 출력
print("[다익스트라 알고리즘]")
print("최단 경로에 따른 노드들:", dijkstra_path)  # 경로에 포함된 노드들을 출력
print("거리 (m):", dijkstra_distance)  # 최단 경로의 거리를 출력
print(f"최단 경로 도출에 소요된 시간: {dijkstra_time:.4f} seconds")  # 실행 시간을 초 단위로 출력
ox.plot_graph_route(network, route=dijkstra_path, route_linewidth=3, node_size=3, edge_linewidth=0.5, edge_color="gray")  # 최단 경로 시각화

# 벨만-포드 알고리즘 정의
def bellman_ford_algorithm(graph, start_node, end_node):
    # 최단 거리 테이블 초기화
    distances = {node: float('inf') for node in graph.nodes}  # 모든 노드의 최단 거리를 무한대로 초기화
    predecessors = {node: None for node in graph.nodes}  # 경로 추적을 위한 이전 노드를 저장
    distances[start_node] = 0  # 출발 노드의 최단 거리를 0으로 설정

    # 그래프 내 모든 엣지를 (노드 개수 - 1)번 반복하여 최단 거리를 갱신
    num_nodes = len(graph.nodes)  # 노드의 개수를 가져옴

    for _ in range(num_nodes - 1):
        for u, v, edge_data in graph.edges(data=True):  # 각 엣지에 대해 거리 갱신
            weight = edge_data['length']  # 엣지의 길이를 가중치로 사용
            if distances[u] + weight < distances[v]:  # 더 짧은 경로가 발견되면
                distances[v] = distances[u] + weight  # 최단 거리를 갱신하고
                predecessors[v] = u  # 이전 노드도 갱신

    # 최단 경로를 역추적
    path = []
    current_node = end_node
    while current_node is not None:  # 도착지부터 출발지까지 이전 노드를 추적하며 경로를 생성
        path.append(current_node)
        current_node = predecessors[current_node]

    return path[::-1], distances[end_node]  # 최단 경로와 해당 경로의 거리 반환

print("벨만-포드 알고리즘을 통해 최단 경로 생성 중...")
# 벨만-포드 알고리즘 실행 및 성능 측정
start_time = time.time()  # 현재 시간을 기록하여 실행 시간을 측정
bellman_ford_path, bellman_ford_distance = bellman_ford_algorithm(network, start_node, end_node)  # 벨만-포드 알고리즘 실행
bellman_ford_time = time.time() - start_time  # 실행이 끝난 후 걸린 시간을 계산

# 벨만-포드 알고리즘 결과 출력
print("[벨만-포드 알고리즘]")
print("최단 경로에 따른 노드들:", bellman_ford_path)  # 경로에 포함된 노드들을 출력
print("거리 (m):", bellman_ford_distance)  # 최단 경로의 거리를 출력
print(f"최단 경로 도출에 소요된 시간: {bellman_ford_time:.4f} seconds")  # 실행 시간을 초 단위로 출력
ox.plot_graph_route(network, route=bellman_ford_path, route_linewidth=3, node_size=3, edge_linewidth=0.5, edge_color="gray")  # 최단 경로 시각화


############################################################################################
# Step 3: Python에서 구현한 Dijkstra algorithm 및 Bellman-Ford Algorithm의 경로 및 성능 비교 #
############################################################################################

print("다익스트라 알고리즘과 벨만-포드 알고리즘의 최단 경로 비교 중...")
# 두 알고리즘의 경로 비교
if dijkstra_path == bellman_ford_path:  # 두 경로가 동일한지 비교
    print("두 알고리즘에 의해 도출된 경로가 같아요!")  # 동일하다면 경로가 같다고 출력
else:
    print("두 알고리즘에 의해 도출된 경로가 달라요!")  # 다르다면 경로가 다르다고 출력
    print("다익스트라 알고리즘 경로:", dijkstra_path)  # 각 알고리즘의 경로를 출력
    print("벨만-포드 알고리즘 경로:", bellman_ford_path)

print("다익스트라 알고리즘과 벨만-포드 알고리즘의 최단 경로 거리 비교 중...")
# 두 알고리즘의 거리 비교
if dijkstra_distance == bellman_ford_distance:  # 두 경로의 거리가 동일한지 비교.
    print("두 알고리즘에 의해 도출된 경로의 거리가 같아요!")  # 동일하다면 거리가 같다고 출력
else:
    print("두 알고리즘에 의해 도출된 경로의 거리가 달라요!")  # 다르다면 거리가 다르다고 출력
    print(f"다익스트라 알고리즘 거리: {dijkstra_distance} m")  # 다익스트라 경로의 거리 출력
    print(f"벨만-포드 알고리즘 거리: {bellman_ford_distance} m")  # 벨만-포드 경로의 거리 출력

# 두 경로 시각적으로 비교
ox.plot_graph_routes(
    network,
    routes=[dijkstra_path, bellman_ford_path],  # 두 경로를 함께 시각화
    route_colors=['red', 'blue'],  # 다익스트라는 빨간색, 벨만-포드는 파란색으로 표시
    route_linewidth=3,  # 각 경로의 선 굵기 설정
    node_size=3,  # 노드의 크기 설정
    edge_linewidth=0.5,  # 엣지의 두께 설정
    edge_color="gray"  # 엣지의 기본 색상 설정
)

print("다익스트라 알고리즘과 벨만-포드 알고리즘의 메모리 사용량 측정 중...")
# 다익스트라 알고리즘의 메모리 사용량 측정
tracemalloc.start()  # 메모리 추적 시작
dijkstra_result = dijkstra_algorithm(network, start_node, end_node)  # 다익스트라 알고리즘 실행
current, peak_dijkstra = tracemalloc.get_traced_memory()  # 현재 메모리 사용량과 최대 메모리 사용량을 기록
tracemalloc.stop()  # 메모리 추적 종료

# 벨만-포드 알고리즘의 메모리 사용량 측정
tracemalloc.start()  # 메모리 추적 시작
bellman_ford_result = bellman_ford_algorithm(network, start_node, end_node)  # 벨만-포드 알고리즘 실행
current, peak_bellman_ford = tracemalloc.get_traced_memory()  # 현재 메모리 사용량과 최대 메모리 사용량을 기록
tracemalloc.stop()  # 메모리 추적 종료

# 메모리 사용량 결과 출력
print(f"다익스트라 알고리즘의 메모리 사용량: {peak_dijkstra / 1024 / 1024:.4f} MB")  # 다익스트라 알고리즘의 메모리 사용량 출력
print(f"벨만-포드 알고리즘의 메모리 사용량: {peak_bellman_ford / 1024 / 1024:.4f} MB")  # 벨만-포드 알고리즘의 메모리 사용량 출력


print("\n끝!")
