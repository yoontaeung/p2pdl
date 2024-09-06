from flask import Flask, jsonify, request
import threading
import random
from p2pdl.node.node import Node
from p2pdl.models.model import SimpleCNN, MLP
from p2pdl.datasets.dataset import load_data 
from p2pdl.utils.crypto import KeyServer

# Flask app 초기화
app = Flask(__name__)

NUM_CLIENTS = 7
TRAINING_ROUNDS = 10
TRAINING_EPOCHS = 5

nodes = []
key_server = KeyServer()
data_loaders = load_data(NUM_CLIENTS, dataset_name='MNIST')

# 노드 초기화
def initialize_nodes():
    global nodes
    nodes = [
        Node(model=MLP(), data=data_loaders[i], key_server=key_server, addr="127.0.0.1", port=6001 + i)
        for i in range(NUM_CLIENTS)
    ]
    
    for node in nodes:
        node.start()
    
    # 노드들끼리 연결
    for node in nodes:
        for other_node in nodes:
            if node != other_node:
                node.connect(other_node)

initialize_nodes()

# 학습 시작을 위한 함수
def start_learning_concurrently(node, rounds, epochs):
    node.set_start_learning(rounds=rounds, epochs=epochs)

# 학습을 시작하는 엔드포인트
@app.route("/start_training", methods=["POST"])
def start_training():
    selected_trainers = random.sample(nodes, 3)
    remaining_nodes = [node for node in nodes if node not in selected_trainers]
    selected_testers = remaining_nodes
    
    for trainer in selected_trainers:
        trainer.trainers_list = selected_trainers
        trainer.testers_list = selected_testers
    
    for tester in selected_testers:
        tester.trainers_list = selected_trainers
        tester.testers_list = selected_testers

    # 모든 노드의 delivered flag 초기화
    for node in nodes:
        node.reset_delivered_flag()

    # 학습 쓰레드 시작
    threads = []
    for trainer in selected_trainers:
        t = threading.Thread(target=start_learning_concurrently, args=(trainer, 1, TRAINING_EPOCHS))
        t.start()
        threads.append(t)

    # 모든 쓰레드 완료 대기
    for t in threads:
        t.join()

    # delivered flag 대기
    for tester in selected_testers:
        tester.wait_for_delivered()

    # 테스트 노드들이 학습 결과 평가
    for tester in selected_testers:
        tester.testing()

    return jsonify({"message": "Training round completed"})

# 노드 상태를 확인하는 엔드포인트
@app.route("/status", methods=["GET"])
def status():
    node_status = [{"port": node.port, "model": node.model.__class__.__name__} for node in nodes]
    return jsonify({"nodes": node_status})

# Flask 서버 시작
if __name__ == "__main__":
    app.run(port=5000)