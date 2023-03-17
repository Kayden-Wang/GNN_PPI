import os

def run_func(description, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, tensorboard_path, graph_only_train, 
            batch_size, epochs):
    os.system("python -u gnn_train.py \
            --description={} \
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
            ".format(description, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, tensorboard_path, graph_only_train, 
                    batch_size, epochs))

if __name__ == "__main__":
    description = "test_string_bfs"

    ppi_path = "./data/9606.protein.actions.all_connected.txt"
    pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"

    split_new = "False"
    split_mode = "bfs"
    train_valid_index_path = "./new_train_valid_index_json/string.bfs.fold1.json"

    use_lr_scheduler = "True"
    save_path = "./save_model/"
    graph_only_train = "False"
    tensorboard_path = "../tf-logs/"

    batch_size = 2048
    # batch_size = 4096 # to make full use of the GPU memory
    # epochs = 300
    epochs = 80 # for test

    run_func(description, ppi_path, pseq_path, vec_path, 
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, tensorboard_path, graph_only_train, 
            batch_size, epochs)