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
import os, torch
from tqdm import tqdm

import time
import pathlib

from DGAC_MP.intent_prediction import IntentPredictor
from Ranking.ranking import Ranking
pathlib.Path("test_dir").mkdir()

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

dialog = {
    'utterance': ['Ух ты, это потрясающе!',
          'Я подумал, что тебе понравится, так как ты интересуешься военной историей.',
          'Есть здесь живописный вид?',
          'Да, здесь есть прекрасный вид на залив Гумбольдта и полуостров Самоа.',
          'Какие предметы представлены на выставке?',
          'На выставке много предметов, включая гаубицкую пушку и историческое здание больницы.',
          'Что еще мы можем увидеть в парке?',
          'Здесь есть исторический сад, в котором есть лекарственные, съедобные и декоративные растения.',
          'Когда парк открылся?'
    ],
    'speaker': [0, 1, 0, 1, 0, 1, 0, 1, 0]
}

next_utterance = [
     'Технический университет Дармштадт в топ-25 университетов Европы с наибольшим научным влиянием по рейтингу QS World University Rankings 2020.',
]


next_intent_start = time.time()
ranking_result = ranking.ranking(dialog, next_utterance)

next_intent_end = time.time()

print(f"Диалоговый граф построен. Время выполнения построение графа {build_end - build_start}s.")
print(f"Время предсказания следующей вершины {next_intent_end - next_intent_start}s")
