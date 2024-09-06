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
TRAINING_ROUNDS = 2
TRAINING_EPOCHS = 5

nodes = []
learning_progress = []
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
    global learning_progress
    learning_progress = []  # Clear the list at the start of training

    for training_round in range(TRAINING_ROUNDS):
        
        selected_trainers = random.sample(nodes, 3)
        remaining_nodes = [node for node in nodes if node not in selected_trainers]
        selected_testers = remaining_nodes

        # Store trainer identities for this round
        trainers_info = [f"{trainer.addr}:{trainer.port}" for trainer in selected_trainers]

        for trainer in selected_trainers:
            trainer.trainers_list = selected_trainers
            trainer.testers_list = selected_testers

        for tester in selected_testers:
            tester.trainers_list = selected_trainers
            tester.testers_list = selected_testers

        # Reset delivered flag for all nodes
        for node in nodes:
            node.reset_delivered_flag()

        # Start training threads
        threads = []
        for trainer in selected_trainers:
            t = threading.Thread(target=start_learning_concurrently, args=(trainer, 1, TRAINING_EPOCHS))
            t.start()
            threads.append(t)

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Wait for delivered flag for testers
        for tester in selected_testers:
            tester.wait_for_delivered()

        # Gather accuracy from testers and store in learning progress
        accuracies = [tester.testing().get('accuracy', 0) for tester in selected_testers]  # Adjusted

        # Append the round data to learning progress
        learning_progress.append({
            'round': training_round + 1,  # 1-based round index
            'trainers': trainers_info,
            'accuracies': accuracies
        })

    # Format the learning progress as required
    final_progress = {
        f'round_{progress["round"]}': {
            'trainers': progress['trainers'],
            'accuracies': [f'{accuracy:.2f}%' for accuracy in progress['accuracies']]
        }
        for progress in learning_progress
    }

    # Return the final formatted progress in JSON format
    return jsonify({
        "message": f"Training completed after {TRAINING_ROUNDS} rounds",
        "learning_progress": final_progress
    })

# 노드 상태를 확인하는 엔드포인트
@app.route("/status", methods=["GET"])
def status():
    node_status = [{"port": node.port, "model": node.model.__class__.__name__} for node in nodes]
    return jsonify({"nodes": node_status})

# Flask 서버 시작
if __name__ == "__main__":
    app.run(port=5000)