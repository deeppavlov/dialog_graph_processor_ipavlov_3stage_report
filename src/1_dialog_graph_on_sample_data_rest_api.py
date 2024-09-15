import time
import pathlib
from flask import Flask, request, jsonify

from DGAC_MP.intent_prediction import IntentPredictor
from Ranking.ranking import Ranking

app = Flask(__name__)

pathlib.Path("test_dir").mkdir(exist_ok=True)

DATA_PATH = "data/dataset/small_data.json"
EMBEDDINGS_FILE = "test_dir/embeddings.npy"
language = "multilingual"
num_speakers = 2
num_clusters_per_stage = [200, 30]

build_start = time.time()

intent_predictor = IntentPredictor(DATA_PATH, EMBEDDINGS_FILE, language, num_speakers, num_clusters_per_stage)
intent_predictor.dialog_graph_auto_construction()
intent_predictor.dump_dialog_graph("./test_dir/dialog_graph")
intent_predictor.dgl_graphs_preprocessing()
intent_predictor.init_message_passing_model("./test_dir/")

build_end = time.time()

GRAPH_PATH = "./test_dir/dialog_graph"
USER_MP_PATH = "./test_dir/GAT_system"
SYSTEM_MP_PATH = "./test_dir/GAT_user"
MODEL_PATHS = [USER_MP_PATH, SYSTEM_MP_PATH]

ranking = Ranking(GRAPH_PATH, MODEL_PATHS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    dialog = data.get('dialog')
    next_utterance = data.get('next_utterance')

    if not dialog or not next_utterance:
        return jsonify({"error": "Invalid input"}), 400
    
    start_time = time.time()
    ranking_result = ranking.ranking(dialog, next_utterance)
    end_time = time.time()

    response = {
        "ranking_result": ranking_result,
        "execution_time": end_time - start_time
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
