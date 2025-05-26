import pytest

import sys
import os

# 添加项目根目录到模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import my_graph


@pytest.fixture
def graph_obj():
    g = my_graph.graph("test_data.txt")
    return g

def test_find_bridge_words_exist(graph_obj):
    bridges = graph_obj.find_bridge_words('to', 'out', call_flag=True)
    assert 'seek' in bridges

def test_find_bridge_words_none(graph_obj):
    bridges = graph_obj.find_bridge_words('strange', 'new', call_flag=True)
    assert bridges == []

def test_find_bridge_words_node_not_found(graph_obj):
    bridges = graph_obj.find_bridge_words('to', 'dog', call_flag=True)
    assert bridges == []

def test_find_bridge_words_empty_input(graph_obj):
    bridges = graph_obj.find_bridge_words('', 'dog', call_flag=True)
    assert bridges == []

def test_find_bridge_words_both_not_in_graph(graph_obj):
    bridges = graph_obj.find_bridge_words('dog', 'cat', call_flag=True)
    assert bridges == []
