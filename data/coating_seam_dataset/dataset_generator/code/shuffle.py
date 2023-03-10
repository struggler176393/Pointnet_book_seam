import numpy as np
import json

def shuffle_data(num):
    txt_num = int(num)

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    train_num = int(train_ratio * txt_num)
    val_num = int(val_ratio * txt_num)
    test_num = int(test_ratio * txt_num)

    txt_list = np.array(range(1,txt_num+1))
    np.random.shuffle(txt_list)

    train_list = []
    val_list = []
    test_list = []
    for i in txt_list[:train_num]:
        train_list.append("shape_data/12345678/"+str(i))
    for i in txt_list[train_num:train_num+val_num]:
        val_list.append("shape_data/12345678/"+str(i))
    for i in txt_list[train_num+val_num:txt_num]:
        test_list.append("shape_data/12345678/"+str(i))


    filename='data/coating_seam_dataset/train_test_split/shuffled_train_file_list.json'
    with open(filename,'w') as file_obj:
        json.dump(train_list,file_obj)

    filename='data/coating_seam_dataset/train_test_split/shuffled_val_file_list.json'
    with open(filename,'w') as file_obj:
        json.dump(val_list,file_obj)

    filename='data/coating_seam_dataset/train_test_split/shuffled_test_file_list.json'
    with open(filename,'w') as file_obj:
        json.dump(test_list,file_obj)