from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os

class graph:
    def __init__(self, file_path=''):
        self.file_path = file_path
        words = process_text_file(file_path)
        self.graph_data = build_word_graph(words)
        self.nodes = self.get_all_nodes()

    def reload_graph(self, file_path):
        if (file_path == ''):
            file_path = 'sample.txt'
        elif file_path.endswith('.txt') == False:
            file_path = file_path + '.txt'

        if (os.path.exists(file_path) == False):
            print("文件不存在，请检查路径")
            exit(1)

        self.file_path = file_path
        words = process_text_file(file_path)
        self.graph_data = build_word_graph(words)
        self.nodes = self.get_all_nodes()
    
    def get_all_nodes(self):
        # 获取图中所有节点
        nodes_from = {src for (src, _) in self.graph_data}
        nodes_to = {dst for (_, dst) in self.graph_data}
        all_nodes = nodes_from.union(nodes_to)
        return all_nodes

    def visible_graph(self):
        G = nx.DiGraph()

        # 添加边和权重
        for (src, dst), weight in self.graph_data.items():
            G.add_edge(src, dst, weight=weight)

        # 更美观的布局（你也可以换回 spring_layout）
        pos = nx.kamada_kawai_layout(G)

        plt.figure(figsize=(12, 8))

        # 绘制边（放前面，确保在节点“底下”）
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=2,
            arrowstyle='-|>', arrows=True,
            connectionstyle='arc3,rad=0.2',  # 增大曲率，避开节点中心
            alpha=0.8,
            min_target_margin=15  # 离目标节点远一点，防止箭头重叠
        )

        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_size=1000,  # 调小一点
            node_color="skyblue",
            edgecolors='black',  # 增加对比边缘
            linewidths=1.5,
            alpha=0.95
        )

        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        # 边权重
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)

        plt.title("Word Adjacency Directed Graph", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("/home/zry/software_lab/lab1/graph.png", dpi=300, bbox_inches='tight')
        plt.show()

    def word_graph(self):
        print('----------------------------------------------------')
        for (src, dst), weight in self.graph_data.items():
            print(f"{src} -> {dst} [weight={weight}]")

    def find_bridge_words(self, word1='', word2='', call_flag=False):
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

        graph_data = self.graph_data

        # 获取图中所有节点
        all_nodes = self.nodes

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
                print(f"没有从 {word1} 到 {word2} 的桥接字!")
        else:
            # 拼接输出格式
            if len(bridge_words) == 1:
                if not call_flag:
                    print(f"{word1} 到 {word2} 的桥接字是: {bridge_words[0]}.")
            else:
                if not call_flag:
                    bridge_str = ', '.join(bridge_words[:-1]) + f", and {bridge_words[-1]}"
                    print(f"{word1} 到 {word2} 的桥接字是: {bridge_str}.")
        return bridge_words
    
    def find_shortest_path(self):
        print('----------------------------------------------------')
        start = input('请输入起点:')
        end = input('请输入终点:')
        if (start == '' and end == ''):
            print("你没有输入任何内容！")
            return
        elif (start == '' and end != '') or (start != '' and end == ''):
            graph_data = self.graph_data
            all_nodes = self.nodes
            root = start if start != '' else end
            if root not in all_nodes:
                print(f"{root} 不在图中.")
                return
            
            paths, distance = dijkstra_all_from_start(graph_data, all_nodes, root)
            for node, path in paths.items():
                path_str = ' -> '.join(path)
                print(f"从 {root} 到 {node} 的路径: {path_str}, 距离: {distance[node]}.")
                
        else :
            graph_data = self.graph_data
            all_nodes = self.nodes
            paths, distance = dijkstra_all_path(graph_data, all_nodes, start, end)
            if paths is None:
                print(f"{start} 到 {end} 不可达.")
            else:
                for path in paths:
                    path_str = ' -> '.join(path)
                    print(f"从 {start} 到 {end} 的路径: {path_str}, 距离: {distance}.")
            return
        
    def get_pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        print('----------------------------------------------------')
        specific_node = input('请输入你想要查询pr值的节点:')

        graph = defaultdict(set)
        reverse_graph = defaultdict(set)
        nodes = set()
        graph_data = self.graph_data

        for src, dst in graph_data:
            graph[src].add(dst)
            reverse_graph[dst].add(src)
            nodes.add(src)
            nodes.add(dst)

        N = len(nodes)

        tfidf_scores = get_tfidf(graph_data)
        # TF-IDF 初始化
        if tfidf_scores:
            total_score = sum(tfidf_scores.get(node, 0.0) for node in nodes)
            if total_score == 0:
                pr = {node: 1 / N for node in nodes}
            else:
                pr = {node: tfidf_scores.get(node, 0.0) / total_score for node in nodes}
        else:
            pr = {node: 1 / N for node in nodes}

        iteration = 0
        for iteration in range(max_iter):
            new_pr = {}
            delta = 0

            # 统计所有出度为0节点的pr总和
            dangling_sum = sum(pr[node] for node in nodes if len(graph[node]) == 0)

            for node in nodes:
                inbound = reverse_graph[node]
                rank_sum = 0
                for q in inbound:
                    out_degree = len(graph[q])
                    if out_degree > 0:
                        rank_sum += pr[q] / out_degree
                # 加上悬挂节点贡献：平均分配
                new_pr[node] = (1 - damping) / N + damping * (rank_sum + dangling_sum / N)
                delta += abs(new_pr[node] - pr[node])

            pr = new_pr

            if delta < tol:
                break

        for (key, value) in pr.items():
            pr[key] = float(value)

        if specific_node:
            if specific_node in pr:
                print(f"{specific_node}的PageRank值: {pr[specific_node]}")
            else:
                print(f"{specific_node} 不在图中.")
        else:
            print(f'全部的PageRank值:')
            for node, value in pr.items():
                print(f"{node}: {value:.6f}")
        return pr

    def random_walk(self):
        print('----------------------------------------------------')
        graph_data = self.graph_data
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
    
def get_tfidf(graph_dict):
    # 1. 构造“文档集合”：每个出发点视为一个“文档”
    docs = defaultdict(list)  # {src: [dst1, dst2, dst2, ...]} （按权重复）
    for (src, dst), count in graph_dict.items():
        docs[src].extend([dst] * count)  # 根据出现次数扩展

    # 2. 准备文本：把 dst 词连成一个字符串
    node_texts = []
    node_list = []
    for src, dst_list in docs.items():
        node_list.append(src)
        node_texts.append(" ".join(dst_list))

    # 3. 用 TF-IDF 处理
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(node_texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix = X.toarray()

    # 4. 累加每个节点（词）的 tf-idf 得分
    tfidf_scores = defaultdict(float)
    for row in tfidf_matrix:
        for i, score in enumerate(row):
            tfidf_scores[feature_names[i]] += score

    return dict(tfidf_scores)

def insert_bridge_words(graph):
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

        bridges = graph.find_bridge_words(w1, w2, True)
        if bridges:
            bridge = random.choice(bridges)
            new_sentence.append(bridge)

    new_sentence.append(words[-1])  # 添加最后一个词
    print('结果:', end='')
    print(' '.join(new_sentence))
    return

def dijkstra(graph_data, nodes, start, end):
    # 获取图中所有节点
    all_nodes = nodes

    unvisited = set(all_nodes) - {start}
    distances = {node: graph_data[(start, node)] if (start, node) in graph_data else float('inf') for node in all_nodes}
    distances[start] = 0
    previous = {}
    for node in all_nodes:
        previous[node] = start if (start, node) in graph_data else None

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

def dijkstra_all_from_start(graph_data, nodes, root):
    all_nodes = nodes
    distances = {node: float('inf') for node in all_nodes}
    previous = {node: None for node in all_nodes}
    distances[root] = 0

    unvisited = set(all_nodes)

    while unvisited:
        current = min(unvisited, key=lambda node: distances[node])
        if distances[current] == float('inf'):
            break
        unvisited.remove(current)

        for neighbor in all_nodes:
            if (current, neighbor) in graph_data and neighbor in unvisited:
                new_distance = distances[current] + graph_data[(current, neighbor)]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current

    # 重建所有路径
    paths = {}
    for node in all_nodes:
        if distances[node] < float('inf') and node != root:
            path = []
            curr = node
            while curr:
                path.append(curr)
                curr = previous[curr]
            path.reverse()
            paths[node] = path

    return paths, distances


def dijkstra_all_path(graph_data, nodes, start, end):
    all_nodes = nodes

    unvisited = set(all_nodes)
    distances = {node: float('inf') for node in all_nodes}
    distances[start] = 0

    # 每个节点的前驱节点列表
    previous = defaultdict(list)

    while unvisited:
        current = min(unvisited, key=lambda node: distances[node])
        if distances[current] == float('inf'):
            break

        unvisited.remove(current)

        for node in all_nodes:
            if (current, node) in graph_data and node in unvisited:
                new_distance = distances[current] + graph_data[(current, node)]
                if new_distance < distances[node]:
                    distances[node] = new_distance
                    previous[node] = [current]  # 重新设置前驱
                elif new_distance == distances[node]:
                    previous[node].append(current)  # 添加额外的前驱

    # 回溯所有路径
    all_paths = []

    def backtrack(curr, path):
        if curr == start:
            all_paths.append([start] + path[::-1])
            return
        for prev in previous[curr]:
            backtrack(prev, path + [curr])

    if distances[end] != float('inf'):
        backtrack(end, [])
        return all_paths, distances[end]
    else:
        return [], None

    
def floyd(graph_data, nodes):
    # 初始化距离矩阵
    dist = defaultdict(int)
    # 获取图中所有节点
    all_nodes = nodes

    for left in all_nodes:
        for right in all_nodes :
            if left == right :
                dist[(left, right)] = 0
            else :
                dist[(left, right)] = graph_data.get((left, right), float('inf'))
    
    # Floyd-Warshall 核心算法
    for k in all_nodes:
        for i in all_nodes:
            for j in all_nodes:
                # 通过中间节点 k 更新从 i 到 j 的最短路径
                if dist[(i, j)] > dist[(i, k)] + dist[(k, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
    
    return dist
    


def process_text_file(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 将换行符替换为空格
    content = content.replace('\n', ' ').replace('\r', ' ')

    # 定义保留字符为英文字母，其余都当作空格
    processed = []
    for char in content:
        if char.isalpha():
            processed.append(char)
        else:
            processed.append(' ')

    # 拼接并按空格分割为单词列表
    cleaned_text = ''.join(processed)
    cleaned_text = cleaned_text.split()
    

    return cleaned_text

def build_word_graph(word_list):
    word_list = [word.lower() for word in word_list]
    edge_weights = defaultdict(int)

    for i in range(len(word_list) - 1):
        a, b = word_list[i], word_list[i + 1]
        edge_weights[(a, b)] += 1

    return edge_weights

# 示例测试
if __name__ == "__main__":
    graph_test = graph(file_path='sample.txt')
    dict_test = get_tfidf(graph_test.graph_data)
    print(dict)
