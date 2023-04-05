import os

def run_func(description, project_name, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, tensorboard_path, graph_only_train, 
            batch_size, epochs):
    os.system("python -u gnn_train.py \
            --description={} \
            --project_name={}\
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --tensorboard_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} \
            ".format(description, project_name, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, tensorboard_path, graph_only_train, 
                    batch_size, epochs))

if __name__ == "__main__":
    description = "test_string_bfs"

    project_name = "GNN-PPI"

    use_dataset = "SHS27k" 
    # use_dataset = "SHS148k"
#     use_dataset = "STRING"

    ppi_path = f"./data/protein.actions.{use_dataset}.STRING.txt"
    pseq_path = f"./data/protein.{use_dataset}.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"
    
    split_new = "False"
    # split_mode = "dfs" # OR "bfs"
    split_mode = "dfs"
#     split_mode = "bfs"
    # split_mode = "random"
    train_valid_index_path = f"./new_train_valid_index_json/{use_dataset}.{split_mode}.fold.json"

    use_lr_scheduler = "True"
    save_path = "./save_model/"
    graph_only_train = "False"
    tensorboard_path = "../tf-logs/"

    batch_size = 2048
    # batch_size = 4096 # to make full use of the GPU memory
    epochs = 300
    # epochs = 80 # for test

    run_func(description, project_name, ppi_path, pseq_path, vec_path, 
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, tensorboard_path, graph_only_train, 
            batch_size, epochs)