# Программный компонент автоматической генерации сценарных диалоговых навыков на основе моделей машинного и глубокого  обучения

Этот репозиторий предоставляет подробный обзор алгоритма, реализованного в программном компоненте для автоматического создания навыков диалога на основе сценариев с использованием моделей машинного и глубокого обучения. Компонент состоит из модулей для построения многодольного сценарного диалогового графа и ранжирования кандидатов для ответов в диалоге.

## Обзор алгоритма

Рабочий процесс алгоритма начинается с построения многотурнирного диалогового графа. Процесс включает в себя кодирование высказываний в диалоге с использованием кодера предложений, двухэтапную кластеризацию для создания узлов графа и определение намерений участников диалога. Этот алгоритм позволяет создавать структуру, учитывающую как содержание высказываний, так и их контекст.

После построения диалогового графа описывается процесс ранжирования кандидатов для ответов в диалоге. Этот процесс включает в себя анализ контекста диалога и многотурнирового графа для определения наиболее подходящих ответов на следующее высказывание. Кандидаты ранжируются на основе их актуальности и пригодности в данном контексте. Фильтр кандидатов для следующего ответа и построение векторного представления для персоны выполняется на основе диалога.

## Запуск сервиса

### Сервис построения графа и предсказания ответов

Для создания сервиса выполните следующую команду:

```bash
docker build -t dialog_graph_processor:1.0 .
```

Для тестирования сервиса выполните следующие команды:

```bash
docker run -ti --rm dialog_graph_processor:1.0 bash
# Построение диалогового графа и графовой модели предсказание следующей вершины, ввод пользовательских высказываний через REST API по HTTP-протоколу
python3 1_dialog_graph_on_sample_data_rest_api.py
# Пример curl запроса для 6 теста
curl -X POST http://172.17.0.2:5000/predict -H "Content-Type: application/json" -d '{
    "dialog": {
        "utterance": ["Ух ты, это потрясающе!",
              "Я подумал, что тебе понравится, так как ты интересуешься военной историей.",
              "Есть здесь живописный вид?",
              "Да, здесь есть прекрасный вид на залив Гумбольдта и полуостров Самоа.",
              "Какие предметы представлены на выставке?",
              "На выставке много предметов, включая гаубицкую пушку и историческое здание больницы.",
              "Что еще мы можем увидеть в парке?",
              "Здесь есть исторический сад, в котором есть лекарственные, съедобные и декоративные растения.",
              "Когда парк открылся?"
        ],
        "speaker": [0, 1, 0, 1, 0, 1, 0, 1, 0]
    },
    "next_utterance": ["Технический университет Дармштадт в топ-25 университетов Европы с наибольшим научным влиянием по рейтингу QS World University Rankings 2020."]
}'
# Фильтрация кандидатов для следующего ответа для продолжения диалога.
python3 2_ranking.py
# Построение векторного представления для персоны в диалоге.
python3 3_persona_embedding.py
# Построение диалогового графа, базовой и графовой моделей и подсчет метрик на датасете Multiwoz.
python3 4_dialog_graph_on_multiwoz_MP_and_baseline.py
# Построение диалогового графа, базовой и графовой моделей и подсчет метрик на датасете Japanese Multiwoz.
python3 5_dialog_graph_on_japanese_multiwoz_MP_and_baseline.py

```

Этот сервис обрабатывает диалоговые графы, предсказывает следующий ответ в диалоге на основе построенного графа, фильтрует кандидатов для следующего ответа, построение векторного представления для персоны в диалоге.

**Примечание:** Убедитесь, что у вас установлен Docker на вашей системе перед выполнением вышеуказанных команд. В противном случае, пожалуйста, обратитесь к официальному [руководству по установке Docker](https://docs.docker.com/get-docker/).
