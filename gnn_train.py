import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from gnn_data import GNN_DATA
from gnn_model import GIN_Net2
from utils import Metrictor_PPI, print_file
import datetime
# from tensorboardX import SummaryWriter

import wandb
os.environ["WANDB_API_KEY"] = "def7b111fbd26d9cf31c22e6c421e3e334d5b7dd"
wandb.login()

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--project_name', default=None, type=str,
                    help='wandb project name')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--split_new', default=None, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=None, type=boolean_string,
                    help="train use learning rate scheduler or not")
parser.add_argument('--save_path', default=None, type=str,
                    help='model save path')
parser.add_argument('--tensorboard_path', default=None, type=str,
                    help='tfboard save path')                  
parser.add_argument('--graph_only_train', default=None, type=boolean_string,
                    help='train ppi graph conctruct by train or all(train with test)')
parser.add_argument('--batch_size', default=None, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=None, type=int,
                    help='train epoch number')

def train(model, graph, ppi_list, loss_fn, optimizer, device,
        result_file_path, summary_writer, save_path,
        batch_size=512, epochs=1000, scheduler=None, 
        got=False):
    
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    truth_edge_num = graph.edge_index.shape[1] // 2

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)
        '''
            函数 train 中，当 got=True 时，将使用 train_mask_got 作为训练集的样本 ID,从而训练一个 GO-terms 模型。
            这个模型的目标是根据蛋白质之间的相似性，预测它们在哪些 GO-terms 中具有共同的功能。
            GO-terms 是 Gene Ontology (GO) 词汇集合中的一个子集，用于描述生物学过程、分子功能和细胞成分。
        '''
        for step in range(steps):
            if step == steps-1:
                if got:
                    train_edge_id = graph.train_mask_got[step*batch_size:]
                else:
                    train_edge_id = graph.train_mask[step*batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step*batch_size : step*batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step*batch_size : step*batch_size + batch_size]
            
            if got:
                output = model(graph.x, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output = model(graph.x, graph.edge_index, train_edge_id)
                label = graph.edge_attr_1[train_edge_id]
            
            label = label.type(torch.FloatTensor).to(device)

            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data)

            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            # summary_writer.add_scalar('train/loss', loss.item(), global_step)
            # summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            # summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            # summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                        .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))
        
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                    os.path.join(save_path, 'gnn_model_train.ckpt'))
        
        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps-1:
                    valid_edge_id = graph.val_mask[step*batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step*batch_size : step*batch_size + batch_size]
                
                output = model(graph.x, graph.edge_index, valid_edge_id)
                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
        
        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']), save_file_path=result_file_path)
        
        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict()},
                        os.path.join(save_path, 'gnn_model_best.ckpt'))
        
        # summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        # summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        # summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        # summary_writer.add_scalar('valid/loss', valid_loss, global_step)
        
        print_file("epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                    .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1, global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)
        
        # use wandb format print these information
        wandb.log({'epoch':epoch, 
                    'train_loss': loss, 
                    'train_recall': recall, 
                    'train_precision': precision, 
                    'train_F1': f1, 
                    'valid_loss': valid_loss, 
                    'valid_recall': metrics.Recall, 
                    'valid_precision': metrics.Precision, 
                    'valid_F1': metrics.F1, 
                    'best_valid_F1': global_best_valid_f1, 
                    'best_valid_F1_epoch': global_best_valid_f1_epoch})

    wandb.finish()
def main():
    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)

    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(args.split_new))
    if args.split_new:
        print("use {} method to split".format(args.split_mode))
    ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data
    print(graph.x.shape)

    ppi_list = ppi_data.ppi_list

    # graph.train_mask 是一个布尔型的一维张量，用于指示哪些节点在训练时应该被包含
    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))
    
    # 将 graph 的原始边索引 edge_index 中训练集所包含的部分赋值给 edge_index_got 属性，并且将其中的起点和终点顺序反转，再与原有的边索引拼接在一起。
    graph.edge_index_got = torch.cat((graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    # 将 graph 的边属性 edge_attr_1 中训练集所包含的部分赋值给 edge_attr_got 属性，并将其复制一份，再拼接在一起。
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]), dim=0)
    # 是一个列表，包含了一个从 0 开始的连续整数序列，长度与 graph.train_mask 相同, 这个列表可以被用来指定在训练过程中要使用的样本 ID
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    graph.to(device)

    model = GIN_Net2(in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1).to(device)

    # Add wandb to script
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=args.project_name, config = args.__dict__, name = nowtime, save_code=True)
    model.run_id = wandb.run.id

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)

    with open(config_path, 'w') as f:
        args_dict = args.__dict__
        for key in args_dict:
            f.write("{} = {}".format(key, args_dict[key]))
            f.write('\n')
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    # summary_writer = SummaryWriter(args.tensorboard_path)
    summary_writer = None
    
    # log the dataset
    GNN_PPI_dataset = wandb.Artifact('PPI', type='dataset')
    GNN_PPI_dataset.add_file(f'{args.ppi_path}')
    GNN_PPI_dataset.add_file(f'{args.pseq_path}')
    GNN_PPI_dataset.add_file(f'{args.vec_path}')
    GNN_PPI_dataset.add_file(f'{args.train_valid_index_path}')
    wandb.log_artifact(GNN_PPI_dataset)
    
    # log the code
    GNN_PPI_code = wandb.Artifact('Python_code', type='code')
    GNN_PPI_code.add_file("./gnn_train.py")
    GNN_PPI_code.add_file("./gnn_data.py")
    GNN_PPI_code.add_file("./gnn_model.py")
    GNN_PPI_code.add_file("./run.py")
    wandb.log_artifact(GNN_PPI_code)

    train(model, graph, ppi_list, loss_fn, optimizer, device,
        result_file_path, summary_writer, save_path,
        batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler, 
        got=args.graph_only_train)
    
    # log the model
    GNN_PPI_model = wandb.Artifact('GNN_PPI_model', type='model')
    GNN_PPI_model.add_file(f'{save_path}/gnn_model_best.ckpt')
    wandb.log_artifact(GNN_PPI_model)

    # summary_writer.close()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    seed_everything(seed=42)
    main()