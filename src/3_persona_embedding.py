# %% [markdown]
# # Пример использования: Построение векторного представления для персоны в диалоге
import os, torch
from tqdm import tqdm


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"
print(torch.cuda.device_count())

# %%
import time
from PersonaEmbeddings.persona_embeddings import PersonaEmbeddings

GRAPH_PATH = "./test_dir/dialog_graph"

persona_embs = PersonaEmbeddings(GRAPH_PATH)

dialog = {
    'utterance': ['Ух ты, это потрясающе!',
          'Есть здесь живописный вид?',
          'Какие предметы представлены на выставке?',
          'Что еще мы можем увидеть в парке?',
          'Когда парк открылся?'
    ],
    'speaker': [0, 0, 0, 0, 0]
}


start = time.time()
persona_embedding = persona_embs.create_representation(dialog)

end = time.time()
print(f"{persona_embedding=}")

print(f"Создано векторное представление персоны по диалогу. Время выполнения {end - start}s")