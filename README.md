# Процессор диалоговых графов

Этот репозиторий предоставляет подробный обзор алгоритма, реализованного в программном компоненте для автоматического создания навыков диалога на основе сценариев с использованием моделей машинного и глубокого обучения. Компонент состоит из модулей для построения многотурнирного диалогового графа и ранжирования кандидатов для ответов в диалоге.

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
# Построение диалогового графа и модели предсказание следующей вершины.
python3 1_dialog_graph.py
# Фильтрация кандидатов для следующего ответа для продолжения диалога.
python3 2_ranking.py
# Построение векторного представления для персоны в диалоге.
python3 3_persona_embedding.py
```

Этот сервис обрабатывает диалоговые графы, предсказывает следующий ответ в диалоге на основе построенного графа, фильтрует кандидатов для следующего ответа, построение векторного представления для персоны в диалоге.

**Примечание:** Убедитесь, что у вас установлен Docker на вашей системе перед выполнением вышеуказанных команд. В противном случае, пожалуйста, обратитесь к официальному [руководству по установке Docker](https://docs.docker.com/get-docker/).
