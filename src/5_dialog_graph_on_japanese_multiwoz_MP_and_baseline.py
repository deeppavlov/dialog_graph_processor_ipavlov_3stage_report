# %% [markdown]
# # Пример использования: Построение диалогового графа и модели предсказание следующей вершины.

# %% [markdown]
# На вход подаются следующие параметры:
# - Число ролей (на данный момент поддерживаются одна и две роли соответственно)
# - Путь к датасету в формате json, представляющему собой список диалогов:  *List[Dict[String, List]]*, со
# - *Dict[String, List]* имеет два ключа и задает ровно один диалог:
    # - Ключ utterance, значение - список реплик диалога
    # - Ключ speaker, значение - список ролей. На i-ой позиции номер роли участника диалога, которому принадлежит i-ая реплика
    # - 0 - user, 1 - system
# - Язык датасета
# - Число кластеров для автопостроения диалогового графа

# two speakers
import time
import pathlib

from DGAC_MP.intent_prediction import IntentPredictor
from Ranking.ranking import Ranking
pathlib.Path("japanese_multiwoz_dir").mkdir()

DATA_PATH = "data/japanese_multiwoz.json"
EMBEDDINGS_FILE = "japanese_multiwoz_dir/embeddings.npy"
language = "en"
num_speakers = 2
num_clusters_per_stage = [200, 30]

build_start = time.time()

intent_predictor = IntentPredictor(DATA_PATH, EMBEDDINGS_FILE, language, num_speakers, num_clusters_per_stage, is_split = True)
intent_predictor.dialog_graph_auto_construction()
intent_predictor.dump_dialog_graph("./japanese_multiwoz_dir/dialog_graph")

build_end = time.time()

base_model_start = time.time()

intent_predictor.run_markov_chain_baseline()

base_model_end = time.time()

base_model_metrics_start = time.time()

intent_predictor.get_markov_chain_metrics()

base_model_metrics_end = time.time()

graph_model_start = time.time()

intent_predictor.dgl_graphs_preprocessing()
intent_predictor.init_message_passing_model("./japanese_multiwoz_dir")

graph_model_end = time.time()

graph_model_metrics_start = time.time()

intent_predictor.get_message_passing_metrics("./japanese_multiwoz_dir")

graph_model_metrics_end = time.time()

print(f"Диалоговый граф построен. Время выполнения построение графа {build_end - build_start}s.")
print(f"Время обучения базовой модели {base_model_end - base_model_start}s")
print(f"Время обучения графовой модели {graph_model_end - graph_model_start}s")
print(f"Получены метрики. Время расчета метрик базовой модели {base_model_metrics_end - base_model_metrics_start}s")
print(f"Получены метрики. Время расчета метрик графовой модели {graph_model_metrics_end - graph_model_metrics_start}s")
