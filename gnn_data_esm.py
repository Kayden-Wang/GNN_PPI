# This file is  used to generate the data for GNN model
# Author: Yifan Wang
# Date: 2020/12/15

import os
import json
import numpy as np
import copy
import torch
import random
import pandas as pd
import gc

from transformers import AutoTokenizer, EsmModel

from tqdm import tqdm

from utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("use: ", device)
class GNN_DATA:
    # data reading 
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=2000, skip_head=True, p1_index=0, p2_index=1, label_index=2, graph_undirection=True, bigger_ppi_path=None):
        '''
        ppi_path: 
            the path of ppi file 
            defult: "./data/9606.protein.actions.all_connected.txt"
        exclude_protein_path: 
            the path of protein file, which contains the protein that we want to exclude
            defult: None
        max_len: 
            the max length of acc number per protein
            defult: 2000
        skip_head: 
            whether to skip the head of ppi file
            defult: True
        p1_index: 
            the index of protein 1 in ppi file
            defult: 0
        p2_index: 
            the index of protein 2 in ppi file
            defult: 1
        label_index: 
            the index of label in ppi file (mode)
            defult: 2
        graph_undirection: 
            whether to make the graph undirection
            defult: True
        bigger_ppi_path: 
            the path of ppi file, which contains more ppi information
            defult: None
        '''
        self.ppi_list = [] 
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {} # no use
        self.protein_name = {}
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len # no use

        name = 0
        ppi_name = 0
        # maxlen = 0
        self.node_num = 0
        self.edge_num = 0
        if exclude_protein_path != None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p:i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}

        # This Loop is to get Graph Node and Edge (protein_name, ppi_dict, ppi_label_list)
        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')

            # Remove the protein that we want to exclude
            if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                continue

            # get node and node name
            # name: the number of node
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1
            
            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1

            # get edge and its label
            # ppi_name: the number of edge
            temp_data = ""
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            
            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label
        
        # This Loop is to get more ppi information (if any)
        if bigger_ppi_path != None:
            skip_head = True
            for line in tqdm(open(bigger_ppi_path)):
                if skip_head:
                    skip_head = False
                    continue
                line = line.strip().split('\t')

                if line[p1_index] not in self.protein_name.keys():
                    self.protein_name[line[p1_index]] = name
                    name += 1
                
                if line[p2_index] not in self.protein_name.keys():
                    self.protein_name[line[p2_index]] = name
                    name += 1
                
                temp_data = ""
                if line[p1_index] < line[p2_index]:
                    temp_data = line[p1_index] + "__" + line[p2_index]
                else:
                    temp_data = line[p2_index] + "__" + line[p1_index]
                
                if temp_data not in self.ppi_dict.keys():
                    self.ppi_dict[temp_data] = ppi_name
                    temp_label = [0, 0, 0, 0, 0, 0, 0]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list.append(temp_label)
                    ppi_name += 1
                else:
                    index = self.ppi_dict[temp_data]
                    temp_label = self.ppi_label_list[index]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list[index] = temp_label

        i = 0
        # This Loop is to get the ppi_list from ppi_dict
        # "9606.ENSP00000000233__9606.ENSP00000263025" -> [9606.ENSP00000000233, 9606.ENSP00000263025]
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        # Convert the protein name in ppi_list to IDs (name -> ID)
        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            # print(len(self.protein_name))
            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]
        
        # Add the reverse edge (if undirected graph)
        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                # if temp_ppi not in self.ppi_list:
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)
    
    def get_protein_aac(self, pseq_path):
        """
        aac: amino acid sequences
        pseq_path: 
            the path of protein sequences
            default: "./data/protein.STRING_all_connected.sequences.dictionary.tsv"
        """
        self.pseq_path = pseq_path
        self.pseq_dict = {}
        self.protein_len = []

        for line in tqdm(open(self.pseq_path)):
            line = line.strip().split('\t')
            if line[0] not in self.pseq_dict.keys():
                self.pseq_dict[line[0]] = line[1]
                self.protein_len.append(len(line[1]))
        
        print("protein num: {}".format(len(self.pseq_dict)))
        print("protein average length: {}".format(np.average(self.protein_len)))
        print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))
        print("protein 80% length: {}".format(np.percentile(self.protein_len, 80)))
        print("protein 90% length: {}".format(np.percentile(self.protein_len, 90)))
        print("protein 95% length: {}".format(np.percentile(self.protein_len, 95)))
        print("protein length over 1024: {:.2%}".format(sum([1 for l in self.protein_len if l>=1024])/len(self.protein_len)))
        

    def embed_normal(self, seq, dim):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        elif len(seq) < self.max_len:
            less_len = self.max_len - len(seq)
            return np.concatenate((seq, np.zeros((less_len, dim))))
        return seq
     
    def ESM_feature_extraction(self, model, batch_size=1, device="cuda"):
        model_name = model.split("/")[-1]
        print("use model:", model_name)
        if not os.path.exists(f'../autodl-tmp/{model_name}'):
            os.mkdir(f'../autodl-tmp/{model_name}')
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = EsmModel.from_pretrained(model).to(device)  # 将模型转移到GPU上
        for i in tqdm(range(0, len(self.pseq_dict_df), batch_size)):
            batch_aac = self.pseq_dict_df['aac'].iloc[i:i+batch_size]
            inputs = tokenizer(batch_aac.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
            inputs = inputs.to(device)  # 将数据转移到GPU上
            with torch.no_grad():  # 禁用梯度计算，以加快推理速度
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            batch_vecs = last_hidden_states.cpu().numpy()  # 将结果移回CPU上
            
            np.save(f'../autodl-tmp/{model_name}/batch_{i}.npy', batch_vecs)

    def vectorize(self, model, regenerate):
        """
            model_name : use which model
            内存优化
        """
        model_name = model.split("/")[-1]
        self.pseq_dict_df = pd.DataFrame(list(self.pseq_dict.items()), columns=['id', 'aac'])
        # 1. 在 self.pseq_dict_df 中预先为 esm_1b 列创建一个空列表：
        # 通过这种方式，我们可以确保在循环之前分配好内存，避免在循环中频繁分配内存。
        self.pseq_dict_df['esm_1b'] = [None] * len(self.pseq_dict_df)
        
        # 2. 在 self.pseq_dict_df 中预先为 esm_1b 列创建一个空列表：
        # 通过这种方式，我们可以确保在循环之前分配好内存，避免在循环中频繁分配内存。
        batch_size = 64
        if regenerate == "True":
            self.ESM_feature_extraction(model=model, batch_size=batch_size, device=device)
        # 3. 在循环中适时释放内存：通过使用 del 和 gc.collect() 释放不再使用的变量内存，我们可以确保内存得到有效回收，从而减小内存占用。
        for i in range(0, len(self.pseq_dict_df), batch_size):
            batch_file = f'../autodl-tmp/{model_name}/batch_{i}.npy'
            print(f"Load: batch_{i}.npy")
            if os.path.exists(batch_file):
                batch_data = np.load(batch_file)
                # print(f'batch_data shape: {batch_data.shape}')
                batch_datas_list = [np.asarray(batch_data[i]) for i in range(batch_data.shape[0])]
                # print(i * batch_size,":",(i + 1) * batch_size - 1)
                # print(f'DataFrame shape: {self.pseq_dict_df.loc[i * batch_size:(i + 1) * batch_size - 1, "esm_1b"].shape}')
                self.pseq_dict_df.loc[i : i + batch_size - 1, 'esm_1b'] = batch_datas_list
                del batch_data, batch_datas_list
                gc.collect()
            else:
                break
        
        print("Load over")
        self.dim = None
        if self.dim is None:
            self.dim = self.pseq_dict_df['esm_1b'][0].shape

        print("esm acid vector dimension: {}".format(self.dim))
        self.pvec_dict = self.pseq_dict_df.set_index('id')['esm_1b'].to_dict()

#         self.pvec_dict = {}
        
#         # 使用生成器进行迭代：通过使用生成器 pvec_generator，我们可以在每次迭代时返回一个键值对，
#         # 避免一次性加载大量数据，从而减小内存占用。
#         def pvec_generator():
#             for index, row in self.pseq_dict_df.iterrows():
#                 protein_id = row['id']
#                 esm_1b = row['esm_1b']
#                 yield protein_id, esm_1b
#                 del row
#                 gc.collect()

#         for key, value in pvec_generator():
#             self.pvec_dict[key] = value

    # protein vectorize 
    def get_feature_origin(self, pseq_path, model="facebook/esm_msa1b_t12_100M_UR50S", regenerate="False"):
        self.get_protein_aac(pseq_path)

        self.vectorize(model, regenerate)

        self.protein_dict = {}
        for name in tqdm(self.protein_name.keys()):
            self.protein_dict[name] = self.pvec_dict[name]

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    # generate pyg data 
    def generate_data(self):
        self.get_connected_num()

        print("Connected domain num: {}".format(self.ufs.count))

        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
        self.x = []
        i = 0
        # for name in self.protein_name:
        #     assert self.protein_name[name] == i
        #     i += 1
        #     self.x.append(self.protein_dict[name])  
        # self.x = np.array(self.x)
        # self.x = torch.tensor(self.x, dtype=torch.float)
        def protein_generator():
            i = 0
            for name in self.protein_name:
                assert self.protein_name[name] == i
                i += 1
                yield self.protein_dict[name]

        self.x = torch.tensor(list(protein_generator()), dtype=torch.float)
        self.data = Data(x=self.x, edge_index=self.edge_index.T, edge_attr_1=self.edge_attr)
        del self.x
        gc.collect()
        print("I am ok")
    # Data partition 
    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        if random_new:
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1-test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1-test_size)) :]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                print("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)
                
                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                
                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()
            
            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_valid_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
                f.close()
    