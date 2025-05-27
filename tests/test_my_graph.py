import pytest
from unittest.mock import patch
from io import StringIO
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

def run_with_input_and_capture(solver, inputs):
    with patch('builtins.input', side_effect=inputs):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            solver.find_shortest_path()
            return mock_stdout.getvalue()

def test_find_shortest_path_both_none(graph_obj):
    output = run_with_input_and_capture(graph_obj, ['', ''])
    assert "你没有输入任何内容" in output

def test_find_shortest_path_not_in_graph(graph_obj):
    output = run_with_input_and_capture(graph_obj, ['dog', ''])
    assert "不在图中" in output

def test_find_shortest_path_from_one(graph_obj):
    output = run_with_input_and_capture(graph_obj, ['new', ''])
    assert "从 new 到 worlds 的路径: new -> worlds, 距离: 1" in output
    assert "从 new 到 strange 的路径: new -> worlds -> to -> explore -> strange, 距离: 4" in output
    assert "从 new 到 explore 的路径: new -> worlds -> to -> explore, 距离: 3" in output
    assert "从 new 到 out 的路径: new -> worlds -> to -> seek -> out, 距离: 4" in output
    assert "从 new 到 life 的路径: new -> life, 距离: 1" in output
    assert "从 new 到 civilizations 的路径: new -> civilizations, 距离: 1" in output
    assert "从 new 到 seek 的路径: new -> worlds -> to -> seek, 距离: 3" in output
    assert "从 new 到 and 的路径: new -> life -> and, 距离: 2" in output
    assert "从 new 到 to 的路径: new -> worlds -> to, 距离: 2" in output

def test_find_shortest_path_from_same(graph_obj):
    output = run_with_input_and_capture(graph_obj, ['new', 'new'])
    assert "从 new 到 new 的路径: new, 距离: 0" in output

def test_find_shortest_path_no_path(graph_obj):
    output = run_with_input_and_capture(graph_obj, ['civilizations', 'new'])
    assert "不可达" in output

def test_find_shortest_path_from_two(graph_obj):
    output = run_with_input_and_capture(graph_obj, ['to', 'worlds'])
    assert "从 to 到 worlds 的路径: to -> explore -> strange -> new -> worlds, 距离: 4" in output
    assert "从 to 到 worlds 的路径: to -> seek -> out -> new -> worlds, 距离: 4" in output