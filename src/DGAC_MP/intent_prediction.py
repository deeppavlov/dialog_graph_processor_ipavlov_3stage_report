import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from DGAC_MP.dgac_two_speakers import Clusters as ClustersTwoSpeakers
from DGAC_MP.dgac_one_speakers import Clusters as ClustersOneSpeaker
from DGAC_MP.data_function_one_partite import get_data as get_data_one_speaker
from DGAC_MP.data_function_two_partite import get_data as get_data_two_speakers
from DGAC_MP.dgl_graph_conctruction import get_dgl_graphs
from DGAC_MP.early_stopping_tools import LRScheduler, EarlyStopping
from DGAC_MP.GAT import GAT_model

from sentence_transformers import SentenceTransformer

class IntentPredictor:
    def __init__(self, data_path, embedding_file, language, num_speakers, num_clusters_per_stage, is_split = False):
        self.data_path = data_path
        self.embedding_file = embedding_file
        self.language = language
        self.num_speakers = num_speakers

        if len(num_clusters_per_stage) > 2 or len(num_clusters_per_stage) == 0:
            raise ValueError("Wrong number of clustering stages")

        self.second_stage_num_clusters = num_clusters_per_stage[-1]

        if len(num_clusters_per_stage) == 1:
            first_stage_num_clusters = -1
        else:
            first_stage_num_clusters = num_clusters_per_stage[0]
        
        if self.num_speakers > 2 or self.num_speakers < 1:
            raise ValueError("Wrong number of speakers")
        elif self.num_speakers == 1:
            self.clusters = ClustersOneSpeaker(
                self.data_path, self.embedding_file, language, first_stage_num_clusters, self.second_stage_num_clusters, is_split
            )
        else:
            self.clusters = ClustersTwoSpeakers(
                self.data_path, self.embedding_file, language, first_stage_num_clusters, self.second_stage_num_clusters, is_split
            )

    def dialog_graph_auto_construction(self):
        self.clusters.form_clusters()

    def dump_dialog_graph(self, PATH):
        pickle.dump(self.clusters, open(PATH, "wb"))

    def one_partite_dgl_graphs_preprocessing(self):
        train_x, train_y = get_data_one_speaker(
            self.clusters.train_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.cluster_train_df,
            np.array(self.clusters.train_embs.astype(np.float64, copy=False)),
        )
        valid_x, valid_y = get_data_one_speaker(
            self.clusters.valid_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.cluster_valid_df,
            np.array(self.clusters.valid_embs.astype(np.float64, copy=False)),
        )

        self.train_dataloader = get_dgl_graphs(train_x, train_y, self.top_k, self.batch_size, True)
        self.valid_dataloader = get_dgl_graphs(valid_x, valid_y, self.top_k, self.batch_size, True)
        
    def one_partite_test_dgl_graphs_preprocessing(self):
        test_x, test_y = get_data_one_speaker(
            self.clusters.test_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.cluster_test_df,
            np.array(self.clusters.test_embs.astype(np.float64, copy=False)),
        )
        self.test_dataloader = get_dgl_graphs(test_x, test_y, self.top_k, self.batch_size, False)

    def two_partite_dgl_graphs_preprocessing(self):
        user_train_x, user_train_y, sys_train_x, sys_train_y = get_data_two_speakers(
            self.clusters.train_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.train_user_df,
            self.clusters.train_system_df,
            np.array(self.clusters.train_user_embs).astype(np.float64, copy=False),
            np.array(self.clusters.train_system_embs).astype(np.float64, copy=False),
        )
        user_valid_x, user_valid_y, sys_valid_x, sys_valid_y = get_data_two_speakers(
            self.clusters.valid_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.valid_user_df,
            self.clusters.valid_system_df,
            np.array(self.clusters.valid_user_embs).astype(np.float64, copy=False),
            np.array(self.clusters.valid_system_embs).astype(np.float64, copy=False),
        )

        self.user_train_dataloader = get_dgl_graphs(user_train_x, user_train_y, self.top_k, self.batch_size, True)
        self.sys_train_dataloader = get_dgl_graphs(sys_train_x, sys_train_y, self.top_k, self.batch_size, True)

        self.user_valid_dataloader = get_dgl_graphs(user_valid_x, user_valid_y, self.top_k, self.batch_size, True)
        self.sys_valid_dataloader = get_dgl_graphs(sys_valid_x, sys_valid_y, self.top_k, self.batch_size, True)
        
    def two_partite_test_dgl_graphs_preprocessing(self):
        user_test_x, user_test_y, sys_test_x, sys_test_y = get_data_two_speakers(
            self.clusters.test_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.test_user_df,
            self.clusters.test_system_df,
            np.array(self.clusters.test_user_embs).astype(np.float64, copy=False),
            np.array(self.clusters.test_system_embs).astype(np.float64, copy=False),
        )

        self.user_test_dataloader = get_dgl_graphs(user_test_x, user_test_y, self.top_k, self.batch_size, False)
        self.sys_test_dataloader = get_dgl_graphs(sys_test_x, sys_test_y, self.top_k, self.batch_size, False)
        
    def dgl_graphs_preprocessing(self, top_k=10, batch_size=128):
        self.top_k = top_k
        self.batch_size = batch_size

        if self.num_speakers == 1:
            self.one_partite_dgl_graphs_preprocessing()
        else:
            self.two_partite_dgl_graphs_preprocessing()
            
    def train_MP_model(
        self,
        train_dataloader,
        valid_dataloader,
        model_file_name,
        early_stopping_steps=5,
        hidden_dim=512,
        num_heads=2,
        learning_rate=0.0001,
        num_epochs=100,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GAT_model(
            self.clusters.embs_dim, hidden_dim, num_heads, self.top_k, self.second_stage_num_clusters
        ).to(device)

        for param in model.parameters():
            param.requires_grad = True

        train_loss_values = []
        valid_loss_values = []

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = LRScheduler(optimizer)
        early_stopping = EarlyStopping(early_stopping_steps)

        for epoch in range(num_epochs):
            train_epoch_loss = 0

            for iter, (batched_graph, labels) in tqdm(enumerate(train_dataloader)):
                logits = model(batched_graph.to(device))
                loss = criterion(logits, labels.to(device))
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.detach().item()

            train_epoch_loss /= iter + 1
            train_loss_values.append(train_epoch_loss)

            valid_epoch_loss = 0
            with torch.no_grad():
                for iter, (batched_graph, labels) in enumerate(valid_dataloader):
                    logits = model(batched_graph.to(device))
                    loss = criterion(logits, labels.to(device))
                    valid_epoch_loss += loss.detach().item()

                valid_epoch_loss /= iter + 1
                valid_loss_values.append(valid_epoch_loss)

            print(f"Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}")

            lr_scheduler(valid_epoch_loss)
            early_stopping(valid_epoch_loss)

            if early_stopping.early_stop:
                break

        torch.save(model, model_file_name)
    
    def init_message_passing_model(self, MODEL_DIR_PATH, num_heads=2, hidden_dim=512):
        embs_dim = self.clusters.embs_dim

        if self.num_speakers == 1:
            self.train_MP_model(self.train_dataloader, self.valid_dataloader, MODEL_DIR_PATH + "/GAT")
        else:
            self.train_MP_model(self.user_train_dataloader, self.user_valid_dataloader, MODEL_DIR_PATH + "/GAT_user")
            self.train_MP_model(self.sys_train_dataloader, self.sys_valid_dataloader, MODEL_DIR_PATH + "/GAT_system")
                        
    def get_message_passing_metrics(self, MODEL_DIR_PATH):
        if self.num_speakers == 1:
            self.one_partite_test_dgl_graphs_preprocessing()
            self.one_stage_MP_metrics([MODEL_DIR_PATH + "/GAT"])
        else:
            self.two_partite_test_dgl_graphs_preprocessing()
            self.two_stage_MP_metrics([MODEL_DIR_PATH + "/GAT_user", MODEL_DIR_PATH + "/GAT_system"])

    def one_stage_MP_metrics(self, MODEL_PATHS):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        file = open(f"MP_one_stage.txt", "w")

        model = torch.load(MODEL_PATHS[0])
        model.eval()
        test_X, test_Y = map(list, zip(*self.test_dataloader))

        probs = []
        test = []

        for i in range(len(test_Y)):
            g = test_X[i].to(device)
            labels = test_Y[i]
            labels = labels.tolist()
            test += labels
            probs_Y = torch.softmax(model(g), 1).tolist()
            probs += probs_Y

        file.write("Accuracy metric\n")

        acc_1 = self.get_accuracy_one_stage_k(1, self.clusters.cluster_test_df, probs, self.clusters.test_dataset)
        acc_3 = self.get_accuracy_one_stage_k(3, self.clusters.cluster_test_df, probs, self.clusters.test_dataset)
        acc_5 = self.get_accuracy_one_stage_k(5, self.clusters.cluster_test_df, probs, self.clusters.test_dataset)
        acc_10 = self.get_accuracy_one_stage_k(10, self.clusters.cluster_test_df, probs, self.clusters.test_dataset)
        MAR = (acc_1 + acc_3 + acc_5 + acc_10) / 4
        
        file.write(f"Acc@1: {acc_1}\n")
        file.write(f"Acc@3: {acc_3}\n")
        file.write(f"Acc@5: {acc_5}\n")
        file.write(f"Acc@10: {acc_10}\n")
        file.write(f"MAR: {MAR}\n")

    def two_stage_MP_metrics(self, MODEL_PATHS):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        file = open(f"MP_two_stage.txt", "w")

        user_model = torch.load(MODEL_PATHS[0])
        user_model.eval()
        user_test_X, user_test_Y = map(list, zip(*self.user_test_dataloader))

        user_probs = []
        user_test = []

        for i in range(len(user_test_Y)):
            g = user_test_X[i].to(device)
            labels = user_test_Y[i]
            labels = labels.tolist()
            user_test += labels
            user_probs_Y = torch.softmax(user_model(g), 1).tolist()
            user_probs += user_probs_Y

        file.write("USER metric\n")

        file.write(f"Acc@1: {self.get_accuracy_k(1, self.clusters.test_user_df, user_probs, self.clusters.test_dataset, 0)}\n")
        file.write(f"Acc@3: {self.get_accuracy_k(3, self.clusters.test_user_df, user_probs, self.clusters.test_dataset, 0)}\n")
        file.write(f"Acc@5: {self.get_accuracy_k(5, self.clusters.test_user_df, user_probs, self.clusters.test_dataset, 0)}\n")
        file.write(f"Acc@10: {self.get_accuracy_k(10, self.clusters.test_user_df, user_probs, self.clusters.test_dataset, 0)}\n")

        system_model = torch.load(MODEL_PATHS[1])
        system_model.eval()
        system_test_X, system_test_Y = map(list, zip(*self.sys_test_dataloader))

        system_probs = []
        system_test = []

        for i in range(len(system_test_Y)):
            g = system_test_X[i].to(device)
            labels = system_test_Y[i]
            labels = labels.tolist()
            system_test += labels
            system_probs_Y = torch.softmax(system_model(g), 1).tolist()
            system_probs += system_probs_Y

        file.write("SYSTEM metric\n")

        file.write(f"Acc@1: {self.get_accuracy_k(1, self.clusters.test_system_df, system_probs, self.clusters.test_dataset, 1)}\n")
        file.write(f"Acc@3: {self.get_accuracy_k(3, self.clusters.test_system_df, system_probs, self.clusters.test_dataset, 1)}\n")
        file.write(f"Acc@5: {self.get_accuracy_k(5, self.clusters.test_system_df, system_probs, self.clusters.test_dataset, 1)}\n")
        file.write(f"Acc@10: {self.get_accuracy_k(10, self.clusters.test_system_df, system_probs, self.clusters.test_dataset, 1)}\n")

        file.write("ALL metric\n")

        acc_1 = self.get_all_accuracy_k(1, self.clusters.test_user_df, self.clusters.test_system_df, user_probs, system_probs, self.clusters.test_dataset)
        acc_3 = self.get_all_accuracy_k(3, self.clusters.test_user_df, self.clusters.test_system_df, user_probs, system_probs, self.clusters.test_dataset)
        acc_5 = self.get_all_accuracy_k(5, self.clusters.test_user_df, self.clusters.test_system_df, user_probs, system_probs, self.clusters.test_dataset)
        acc_10 = self.get_all_accuracy_k(10, self.clusters.test_user_df, self.clusters.test_system_df, user_probs, system_probs, self.clusters.test_dataset)
        MAR = (acc_1 + acc_3 + acc_5 + acc_10) / 4
        
        file.write(f"Acc@1: {acc_1}\n")
        file.write(f"Acc@3: {acc_3}\n")
        file.write(f"Acc@5: {acc_5}\n")
        file.write(f"Acc@10: {acc_10}\n")
        file.write(f"MAR: {MAR}\n")

    def one_stage_markov_chain_baseline(self):
        self.probs = np.zeros((self.second_stage_num_clusters + 1, self.second_stage_num_clusters))
    
        index = 0

        for obj in self.clusters.train_dataset:
            # кластер с номером, равным числу вершин, - это нулевой кластер
            pred_cluster = self.second_stage_num_clusters

            for j in range(len(obj)):
                cur_cluster = self.clusters.cluster_train_df["cluster"][index]
                index += 1

                self.probs[pred_cluster][cur_cluster] += 1
                pred_cluster = cur_cluster

        for i in range(self.second_stage_num_clusters + 1):
            sum_i_probs = sum(self.probs[i])

            if sum_i_probs != 0:
                self.probs[i] /= sum_i_probs
        
    def two_stage_markov_chain_baseline(self):
        self.probs_user_sys = np.zeros((self.second_stage_num_clusters + 1, self.second_stage_num_clusters))
        self.probs_sys_user = np.zeros((self.second_stage_num_clusters + 1, self.second_stage_num_clusters))

        ind_user = 0
        ind_system = 0

        for obj in self.clusters.train_dataset:
            # кластер с номером, равным числу вершин, - это нулевой кластер
            pred_cluster = self.second_stage_num_clusters

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    cur_cluster = self.clusters.train_user_df["cluster"][ind_user]

                    ind_user += 1

                    self.probs_sys_user[pred_cluster][cur_cluster] += 1
                    pred_cluster = cur_cluster
                else:
                    cur_cluster = self.clusters.train_system_df["cluster"][ind_system]

                    ind_system += 1

                    self.probs_user_sys[pred_cluster][cur_cluster] += 1
                    pred_cluster = cur_cluster

        for i in range(self.second_stage_num_clusters + 1):
            sum_i_user_sys = sum(self.probs_user_sys[i])
            sum_i_sys_user = sum(self.probs_sys_user[i])

            if sum_i_user_sys != 0:
                self.probs_user_sys[i] /= sum_i_user_sys

            if sum_i_sys_user != 0:    
                self.probs_sys_user[i] /= sum_i_sys_user
                
    def one_stage_markov_chain_metrics(self):
        file = open(f"MarkovChain_one_stage.txt", "w")

        test = []
        index = 0

        for obj in self.clusters.test_dataset:
            pred_cluster = self.second_stage_num_clusters

            for j in range(len(obj["utterance"])):
                cur_cluster = self.clusters.cluster_test_df["cluster"][index]
                test.append(self.probs[pred_cluster])
                index += 1
                pred_cluster = cur_cluster

        file.write("Accuracy metric\n")

        acc_1 = self.get_accuracy_one_stage_k(1, self.clusters.cluster_test_df, test, self.clusters.test_dataset)
        acc_3 = self.get_accuracy_one_stage_k(3, self.clusters.cluster_test_df, test, self.clusters.test_dataset)
        acc_5 = self.get_accuracy_one_stage_k(5, self.clusters.cluster_test_df, test, self.clusters.test_dataset)
        acc_10 = self.get_accuracy_one_stage_k(10, self.clusters.cluster_test_df, test, self.clusters.test_dataset)
        MAR = (acc_1 + acc_3 + acc_5 + acc_10) / 4
        
        file.write(f"Acc@1: {acc_1}\n")
        file.write(f"Acc@3: {acc_3}\n")
        file.write(f"Acc@5: {acc_5}\n")
        file.write(f"Acc@10: {acc_10}\n")
        file.write(f"MAR: {MAR}\n")
        
    def two_stage_markov_chain_metrics(self):
        file = open(f"MarkovChain_two_stage.txt", "w")

        sys_test = []
        user_test = []

        ind_user = 0
        ind_system = 0

        for obj in self.clusters.test_dataset:
            pred_cluster = self.second_stage_num_clusters

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    cur_cluster = self.clusters.test_user_df["cluster"][ind_user]
                    user_test.append(self.probs_sys_user[pred_cluster])
                    ind_user += 1
                    pred_cluster = cur_cluster
                else:
                    cur_cluster = self.clusters.test_system_df["cluster"][ind_system]
                    sys_test.append(self.probs_user_sys[pred_cluster])
                    ind_system += 1
                    pred_cluster = cur_cluster
        
        file.write("USER metric\n")

        file.write(f"Acc@1: {self.get_accuracy_k(1, self.clusters.test_user_df, user_test, self.clusters.test_dataset, 0)}\n")
        file.write(f"Acc@3: {self.get_accuracy_k(3, self.clusters.test_user_df, user_test, self.clusters.test_dataset, 0)}\n")
        file.write(f"Acc@5: {self.get_accuracy_k(5, self.clusters.test_user_df, user_test, self.clusters.test_dataset, 0)}\n")
        file.write(f"Acc@10: {self.get_accuracy_k(10, self.clusters.test_user_df, user_test, self.clusters.test_dataset, 0)}\n")

        file.write("SYSTEM metric\n")

        file.write(f"Acc@1: {self.get_accuracy_k(1, self.clusters.test_system_df, sys_test, self.clusters.test_dataset, 1)}\n")
        file.write(f"Acc@3: {self.get_accuracy_k(3, self.clusters.test_system_df, sys_test, self.clusters.test_dataset, 1)}\n")
        file.write(f"Acc@5: {self.get_accuracy_k(5, self.clusters.test_system_df, sys_test, self.clusters.test_dataset, 1)}\n")
        file.write(f"Acc@10: {self.get_accuracy_k(10, self.clusters.test_system_df, sys_test, self.clusters.test_dataset, 1)}\n")

        file.write("ALL metric\n")

        acc_1 = self.get_all_accuracy_k(1, self.clusters.test_user_df, self.clusters.test_system_df, user_test, sys_test, self.clusters.test_dataset)
        acc_3 = self.get_all_accuracy_k(3, self.clusters.test_user_df, self.clusters.test_system_df, user_test, sys_test, self.clusters.test_dataset)
        acc_5 = self.get_all_accuracy_k(5, self.clusters.test_user_df, self.clusters.test_system_df, user_test, sys_test, self.clusters.test_dataset)
        acc_10 = self.get_all_accuracy_k(10, self.clusters.test_user_df, self.clusters.test_system_df, user_test, sys_test, self.clusters.test_dataset)
        MAR = (acc_1 + acc_3 + acc_5 + acc_10) / 4
        
        file.write(f"Acc@1: {acc_1}\n")
        file.write(f"Acc@3: {acc_3}\n")
        file.write(f"Acc@5: {acc_5}\n")
        file.write(f"Acc@10: {acc_10}\n")
        file.write(f"MAR: {MAR}\n")
        
    def run_markov_chain_baseline(self):
        if self.num_speakers == 1:
            self.one_stage_markov_chain_baseline()
        else:
            self.two_stage_markov_chain_baseline()
            
    def get_markov_chain_metrics(self):
        if self.num_speakers == 1:
            self.one_stage_markov_chain_metrics()
        else:
            self.two_stage_markov_chain_metrics()        

    def get_accuracy_one_stage_k(self, k, test_df, probabilities, data):
        '''
            metric function
        '''
        index = 0
        metric = []

        for obj in data:
            utterance_metric = []

            for i in range(len(obj["utterance"])):
                cur_cluster = test_df["cluster"][index]

                top = []

                for j in range(len(probabilities[index][:])):
                    top.append((probabilities[index][j], j))

                top.sort(reverse=True)
                top = top[:k]

                if (probabilities[index][cur_cluster], cur_cluster) in top:
                    utterance_metric.append(1)
                else:
                    utterance_metric.append(0)
                index += 1

            metric.append(np.array(utterance_metric).mean()) 
        return np.array(metric).mean()
    
    def get_accuracy_k(self, k, test_df, probabilities, data, flag):
        '''
            metric function, flag: user - speaker 0, system - speaker 1
        '''
        index = 0
        metric = []

        for obj in data:
            utterance_metric = []

            for i in range(len(obj["utterance"])):
                if obj['speaker'][i] == flag:
                    cur_cluster = test_df["cluster"][index]

                    top = []

                    for j in range(len(probabilities[index][:])):
                        top.append((probabilities[index][j], j))

                    top.sort(reverse=True)
                    top = top[:k]

                    if (probabilities[index][cur_cluster], cur_cluster) in top:
                        utterance_metric.append(1)
                    else:
                        utterance_metric.append(0)
                    index += 1

            metric.append(np.array(utterance_metric).mean()) 
        return np.array(metric).mean()

    def get_all_accuracy_k(self, k, test_user_data, test_system_data, probs_sys_user, probs_user_sys, data):
        '''
            metric function for both speakers
        '''
        ind_user = 0
        ind_system = 0
        metric = []

        for obj in data:
            utterance_metric = []
            pred_cluster = -1

            for i in range(len(obj["utterance"])):
                if obj['speaker'][i] == 0:
                    cur_cluster = test_user_data["cluster"][ind_user]

                    top = []

                    for j in range(len(probs_sys_user[ind_user][:])):
                        top.append((probs_sys_user[ind_user][j], j))

                    top.sort(reverse=True)
                    top = top[:k]

                    if (probs_sys_user[ind_user][cur_cluster], cur_cluster) in top:
                        utterance_metric.append(1)
                    else:
                        utterance_metric.append(0)
                    pred_cluster = cur_cluster   
                    ind_user += 1
                else:
                    cur_cluster = test_system_data["cluster"][ind_system]

                    top = []

                    for j in range(len(probs_user_sys[ind_system][:])):
                        top.append((probs_user_sys[ind_system][j], j))

                    top.sort(reverse=True)
                    top = top[:k]

                    if (probs_user_sys[ind_system][cur_cluster],cur_cluster) in top:
                        utterance_metric.append(1)
                    else:
                        utterance_metric.append(0)
                    pred_cluster = cur_cluster  
                    ind_system += 1


            metric.append(np.array(utterance_metric).mean()) 
        return np.array(metric).mean()