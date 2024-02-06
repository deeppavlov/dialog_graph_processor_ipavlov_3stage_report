FROM anibali/pytorch:2.0.1-cuda11.8

RUN pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
RUN pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
RUN pip install torch_geometric
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html 

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY src .