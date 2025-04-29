import os
import my_graph
import sys


# 示例：使用文件路径调用函数
if __name__ == "__main__":
    if len(sys.argv) < 2:
        file_path = input('键入你的文件路径:')  # 请将此处替换为你的文件路径  
    else:
        file_path = sys.argv[1]
        
    if (file_path == ''):
        file_path = 'sample.txt'
    elif file_path.endswith('.txt') == False:
        file_path = file_path + '.txt'   
    if (os.path.exists(file_path) == False):
        print("文件不存在，请检查路径")
        exit(1)

    graph = my_graph.graph(file_path=file_path)
    graph_data = graph.graph_data

    while True:
        choice = input('功能列表：\n1.文字化图\n2.可视化图\n3.桥接字\n4.插入桥接字\n5.最短路径\n6.求pr值\n7.随机游走\n8.重新输入文件路径\n9.退出\n请输入你的选择:')
        if choice == '1':
            graph.word_graph()
        elif choice == '2':
            graph.visible_graph()
        elif choice == '3':
            graph.find_bridge_words()
        elif choice == '4':
            my_graph.insert_bridge_words(graph)
        elif choice == '5':
            graph.find_shortest_path()
        elif choice == '6':
            graph.get_pagerank()
        elif choice == '7':
            graph.random_walk()
        elif choice == '8':
            file_path = input('键入你的文件路径:')
            graph.reload_graph(file_path)
        elif choice == '9':
            break
        else:
            print("无效的选择，请重新输入。")
        
        input('按回车键继续...')
        os.system('cls' if os.name == 'nt' else 'clear')