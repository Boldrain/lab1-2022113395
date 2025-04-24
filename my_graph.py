import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def build_word_graph(word_list):
    word_list = [word.lower() for word in word_list]
    edge_weights = defaultdict(int)

    for i in range(len(word_list) - 1):
        a, b = word_list[i], word_list[i + 1]
        edge_weights[(a, b)] += 1

    return edge_weights

def visible_graph(graph_data):
    G = nx.DiGraph()

    # 添加边和权重（确保每对节点最多一条边）
    for (src, dst), weight in graph_data.items():
        G.add_edge(src, dst, weight=weight)

    # 选择布局方式
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue",
            arrows=True, font_size=10, width=2, arrowstyle='->')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Word Adjacency Directed Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def word_graph(graph_data):
    print('----------------------------------------------------')
    for (src, dst), weight in graph_data.items():
        print(f"{src} -> {dst} [weight={weight}]")

def find_bridge_words(graph_data, word1='', word2='', call_flag=False):
    if not call_flag:
        print('----------------------------------------------------')
        word1 = input('键入第一个桥接字:')
        word2 = input('键入第二个桥接字:')
    
    if word1 == '' or word2 == '':
        print("你输入的内容有误！")
        return []
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()
    bridge_words = []

    # 获取图中所有节点
    nodes_from = {src for (src, _) in graph_data}
    nodes_to = {dst for (_, dst) in graph_data}
    all_nodes = nodes_from.union(nodes_to)

    # 检查两个单词是否在图中
    if word1 not in all_nodes :
        if word2 not in all_nodes :
            if not call_flag:
                print(f"No {word1} or {word2} in the graph!")
            return bridge_words
        else:
            if not call_flag:
                print(f"No {word1} in the graph!")
            return bridge_words
    elif word2 not in all_nodes:
        if not call_flag:
            print(f"No {word2} in the graph!")
        return bridge_words

    
    for (src1, mid) in graph_data:
        if src1 == word1:
            if (mid, word2) in graph_data:
                bridge_words.append(mid)

    if not bridge_words:
        if not call_flag:
            print(f"No bridge words from {word1} to {word2}!")
    else:
        # 拼接输出格式
        if len(bridge_words) == 1:
            if not call_flag:
                print(f"The bridge word from {word1} to {word2} is: {bridge_words[0]}.")
        else:
            if not call_flag:
                bridge_str = ', '.join(bridge_words[:-1]) + f", and {bridge_words[-1]}"
                print(f"The bridge words from {word1} to {word2} are: {bridge_str}.")
    return bridge_words

def insert_bridge_words(graph_data):
    print('----------------------------------------------------')
    sentence = input('请输入你想要插入桥接字的句子:')
    if (sentence == ''):
        print("你没有输入任何内容！")
        return 
    
    words = sentence.strip().lower().split()
    new_sentence = []

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        new_sentence.append(w1)

        bridges = find_bridge_words(graph_data, w1, w2, True)
        if bridges:
            bridge = random.choice(bridges)
            new_sentence.append(bridge)

    new_sentence.append(words[-1])  # 添加最后一个词
    print('结果:', end='')
    print(' '.join(new_sentence))
    return

def find_shortest_path(graph_data):
    print('----------------------------------------------------')
    start = input('请输入起点:')
    end = input('请输入终点:')
    if (start == '' or end == ''):
        print("你没有输入任何内容！")
        return
    path, distance = dijkstra(graph_data, start, end)
    if path is None:
        print(f"{start} 到 {end} 不可达.")
    else:
        print(f"最短路径: {' -> '.join(path)}.")
        print(f"最短距离: {distance}.")
    return

def dijkstra(graph_data, start, end):
    # 获取图中所有节点
    nodes_from = {src for (src, _) in graph_data}
    nodes_to = {dst for (_, dst) in graph_data}
    all_nodes = nodes_from.union(nodes_to)

    unvisited = set(all_nodes) - {start}
    distances = {node: graph_data[(start, node)] if (start, node) in graph_data else float('inf') for node in all_nodes}
    # for node_left in all_nodes:
    #     distances.update({node_right: graph_data[node_left][node_right] if (node_left, node_right) in graph_data else float('inf') for node_right in (all_nodes - {node_left})})
    previous = {node: None for node in all_nodes}
    for node in all_nodes:
        if (start, node) in graph_data:
            previous[node] = start
    distances[start] = 0

    while unvisited:
        # 找当前距离最小的节点
        current = min(unvisited, key=lambda node: distances[node])
        if current is None or distances[current] == float('inf'):
            break  # 无法再到达其他节点

        unvisited.remove(current)

        # 如果到达终点，可以提前结束
        if current == end:
            break

        for node in all_nodes:
            if (current, node) in graph_data and node in unvisited:
                new_distance = distances[current] + graph_data[(current, node)]
                if new_distance < distances[node]:
                    distances[node] = new_distance
                    previous[node] = current

    # 重建路径
    path = []
    curr = end
    while curr:
        path.append(curr)
        curr = previous.get(curr)
    path.reverse()

    if path and path[0] == start:
        return path, distances[end]
    else:
        return None, None
    
def get_pagerank(graph_data, damping=0.85, max_iter=100, tol=1e-6):
    print('----------------------------------------------------')
    specific_node = input('请输入你想要查询pr值的节点:')
    # edges: List of (from, to)
    graph = defaultdict(set)       # 出边
    reverse_graph = defaultdict(set)  # 入边
    nodes = set()

    for src, dst in graph_data:
        graph[src].add(dst)
        reverse_graph[dst].add(src)
        nodes.add(src)
        nodes.add(dst)

    N = len(nodes)
    pr = {node: 1 / N for node in nodes}

    for iteration in range(max_iter):
        new_pr = {}
        delta = 0  # 用于判断是否收敛

        for node in nodes:
            inbound = reverse_graph[node]
            rank_sum = 0
            for q in inbound:
                out_degree = len(graph[q])
                if out_degree > 0:
                    rank_sum += pr[q] / out_degree
            new_pr[node] = (1 - damping) / N + damping * rank_sum
            delta += abs(new_pr[node] - pr[node])

        pr = new_pr

        if delta < tol:
            break
    if specific_node:
        if specific_node in pr:
            print(f"{specific_node}的PageRank值: {pr[specific_node]}")
        else:
            print(f"{specific_node} 不在图中.")
    else:
        print(f'全部的PageRank值:{pr}')
    return pr

def random_walk(graph_data):
    print('----------------------------------------------------')
    visited_edges = set()
    visited_nodes = []
    # path_edges = []

    # 随机选一个起点
    current = random.choice([left for left, _ in graph_data])
    visited_nodes.append(current)

    print(f"起点: {current}")

    while True:
        neighbors = [dst for (src, dst) in graph_data if src == current]
        if not neighbors:
            print(f"节点 {current} 没有出边，结束遍历。")
            break

        next_node = random.choice(neighbors)
        edge = (current, next_node)

        if edge in visited_edges:
            visited_nodes.append(next_node)
            # path_edges.append(edge)
            print(f"遇到重复边 {edge}，结束遍历。")
            break

        visited_nodes.append(next_node)
        # path_edges.append(edge)
        visited_edges.add(edge)
        current = next_node

    walk_path = ' '.join(visited_nodes)
    print(f"遍历路径: {walk_path}")
    with open('random_walk_path.txt', 'w', encoding='utf-8') as f:
        f.write(walk_path)
    return

# 示例测试
if __name__ == "__main__":
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "the", "quick"]
    edge_weights = build_word_graph(words)
    visible_graph(edge_weights)
