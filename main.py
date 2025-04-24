import string
import os
import my_graph

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


# 示例：使用文件路径调用函数
if __name__ == "__main__":
    file_path = input('键入你的文件路径:')  # 请将此处替换为你的文件路径
    if (file_path == ''):
        file_path = 'sample.txt'
    elif file_path.endswith('.txt') == False:
        file_path = file_path + '.txt'

    if (os.path.exists(file_path) == False):
        print("文件不存在，请检查路径")
        exit(1)

    words = process_text_file(file_path)
    words_graph = my_graph.build_word_graph(words)

    # 文字化图
    # my_graph.word_graph(words_graph)
    # 可视化图
    my_graph.visible_graph(words_graph)
    # 桥接字
    # word1 = input('键入第一个桥接字:')
    # word2 = input('键入第二个桥接字:')
    # if word1 != '' and word2 != '':
    # my_graph.find_bridge_words(words_graph)
    # 插入桥接字
    # my_graph.insert_bridge_words(words_graph)
    # 最短路径
    # my_graph.find_shortest_path(words_graph)
    # 求pr值
    # my_graph.get_pagerank(words_graph)
    # 随机游走
    # my_graph.random_walk(words_graph)