# %% [markdown]
# # Пример использования: Фильтрация кандидатов для следующего ответа для продолжения диалога

# %% [markdown]

# Для инициализации класса в качестве параметров подается следующее:
# - Путь к бинарному файлу объекта класса Clusters 
# - Пути к модели [MP_PATH]/моделям [USER_MP_PATH, SYSTEM_MP_PATH] message passing

# Для ранжирования: 
# - Диалог: *Dict[String, List]* имеет два ключа и задает ровно один диалог:
#     - Ключ utterance, значение - список реплик диалога
#     - Ключ speaker, значение - список ролей. На i-ой позиции номер роли участника диалога, которому принадлежит i-ая реплика
#     - 0 - user, 1 - system
# - Список реплик, который будет ранжироватьс
# %%
import time
from Ranking.ranking import Ranking

# %%
# two speakers
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

next_utterances = [
     'Технический университет Дармштадт в топ-25 университетов Европы с наибольшим научным влиянием по рейтингу QS World University Rankings 2020.',
     'В рейтинге университетов Европы Технический университет Дармштадт занимает 15-е место',
     'Погода прекрасная, не находите?',
     'Максимальная длина гавани Отаго составляет 21 километр (13 миль).',
     'Первое мероприятие состоялось на стадионе 24 октября 1971 года.',
     'Это произошло 6 миллионов лет назад.',
     'Арена закрылась в июне 2010 года.',
     'Парк был основан в 1955 году.'
]

start = time.time()
ranking_result = ranking.ranking(dialog, next_utterances)

end = time.time()
for utterance, score in ranking_result:
    print(score, utterance)

print(f"Выполнено ранжирование кандидатов. Время выполнения {end - start}s")