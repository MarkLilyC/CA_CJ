import os
import yaml
import shutil
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
import math
import torch.utils as utils
import random
from torch.utils.data import Dataset, DataLoader
import argparse
from case import Simulation, SimulationCase
import random
import pickle

LOGCOLOR = {
    0:'\033[0;30;41m',
    1:'\033[0;30;42m',
    2:'\033[0;30;43m'
}

FEATURES = {
    "ped"
}

class cstr():
    def __init__(self, input_str, mode = 1):
        self.LOGCOLOR = {
                0:'\033[0;30;41m',
                1:'\033[0;30;42m',
                2:'\033[0;30;43m'
            }
        self.__str = self.LOGCOLOR[mode] + str(input_str) + '\033[0m'
    def __str__(self) -> str:
        return self.__str

class GCNDataset(Dataset):
    def __init__(self, adj, feature, time):
        self.adj = adj
        self.feature = feature
        self.time = time
    
    def __len__(self) -> int:
        return len(self.time)
    
    def __getitem__(self, index):
        adj_index = index // 20
        adj = self.adj[adj_index]
        feature = self.feature[index]
        label = self.time[index]
        return adj, feature, label

class GCNDataset2(Dataset):
    def __init__(self, sims:list, device:str):
        self.sims = sims
        self.device = device
    
    def __len__(self) -> int:
        return len(self.sims)
    
    def __getitem__(self, index):
        sim = self.sims[index]
        adj, feature, time = sim.get(device = self.device)
        return  adj, feature, time
        

def check_path(filepath:str) -> bool:
    if os.path.exists(filepath):
        return True
    else: return False

def path_normalize(path:str) -> str:
    '''将路径中的'/'转换为'//'

    Args:
        path (str): 原始路径

    Returns:
        str: 转换后的路径
    '''
    res = ""
    for i in path:
        if i == '/':res += "//"
        else: res += i
    if res.endswith('//'): return res
    else: return res + "//"

def load_adj(data_path_:str) -> sp._csc.csc_matrix:
    '''加载npz文件保存的adj矩阵数据

    Args:
        data_path_ (str): 保存adj矩阵的npz文件路径

    Returns:
        sp._csc.csc_matrix: adj稀疏矩阵
    '''
    adj = sp.load_npz(data_path_)
    adj.tocsc()
    # 房间节点数
    vertex_num = adj.shape[0]
    return adj, vertex_num

def calculate_adj_num(adj:sp._csc.csc_matrix) -> list:
    '''计算adj矩阵中每个节点的连接个数与总连接值
    [[连通个数], [连通值]]

    Args:
        adj (sp._csc.csc_matrix): 通过load_adj方法读取的adj邻接矩阵

    Returns:
        list: 矩阵中节点的连接个数与总连接值
    '''
    # 0 转换为密集矩阵
    adj = adj.todense() # numpy.matrix
    res = [[],[]]
    # 迭代所有的行
    rows = adj.shape[0]
    for row in range(rows):
        # 定义一个adj个数和adj值
        adj_num, adj_value = 0, 0
        # 取出当前位置的行转换为list
        cur_row = adj[row].tolist()[0]
        # 迭代行里的值
        for val in cur_row:
            # 如果值>0 则将adj个数+1 将adj值加上该值
            if val > 0:
                adj_num += 1
                adj_value += val
            else:pass
        # 迭代结束后将adj个数和adj值添加到res
        res[0].append(adj_num)
        res[1].append(adj_value)
    return res

def load_yaml(file_path_:str):
    '''加载yaml文件 返回一个可迭代对象

    Args:
        file_path_ (str): yaml文件绝对路径

    Returns:
        _type_: _description_
    '''
    assert file_path_.endswith('.yml'), "The target File should be a '.yml' file, while current {}".format('.' + file_path_.split('.')[-1])
    with open(file=file_path_) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data

def dump_yaml(_file_path:str, _data, _allow_overwriting = False):
    assert _file_path.endswith('.yml'), "The target File should be a '.yml' file, while current {}".format('.' + file_path_.split('.')[-1])
    if os.path.exists(_file_path):
        if _allow_overwriting:
            with open(_file_path, 'w') as _file:
                yaml.dump(_data, _file)
        else:
            raise FileExistsError(f"file exists and not allowed to overwriting")
    else:
        with open(_file_path, 'w') as _file:
                yaml.dump(_data, _file)

def get_file_names(folder_path_:str, suffix_:str, startswith_ = None) ->list:
    '''根据后缀名 获得指定目录下的文件名 
    * 无法识别嵌套目录下的文件
    Args:
        folder_path_ (str): 指定目录绝对路径 
            * 如果后续需要紧接着使用函数get_file_path进行文件名与路径的拼接 则需要本参数以'/'皆为
        suffix_ (str): 筛选文件的后缀名 
            * 以'.'开头
            * 如果不指定文件后缀 则选出所有文件
    Returns:
        list: 返回文件名
    '''
    res_ = []
    file_name_ = os.listdir(folder_path_)
    # 检查输入的文件后缀是否以'.'开头
    if suffix_ is not None:
        assert suffix_[0] == '.', "Suffix should start with '.'"
        # 如果指定了文件后缀
        for filename in file_name_:
            if os.path.splitext(filename)[1] == suffix_:
                if startswith_ is None:
                    res_.append(filename)
                else:
                    if filename.startswith(startswith_): res_.append(filename)
                    else:pass
    # 如果没有指定文件后缀
    else:
        for filename in file_name_:
            if startswith_ is None:
                res_.append(filename)
            else:
                if filename.startswith(startswith_): res_.append(filename)
                else:pass
    return res_

def get_file_path(folder_path:str, suffix:str) -> list:
    '''读取给定文件夹路径中的所有文件名，并将文件名与文件夹路径组合为文件路径

    Args:
        folder_path (str): 文件夹路径
        suffix (str): 文件后缀

    Returns:
        list: 文件名路径列表
    '''
    folder_path = path_normalize(folder_path)
    filenames = get_file_names(folder_path_=folder_path, suffix_=suffix)
    res = []
    for name in filenames:res.append(folder_path + name)
    return res
    
def save_file_path(folder_path_:str, suffix_:str, dump_path_:str, dump_data_type_= dict, _overw = True) -> dict:
    '''根据给定文件后缀找出给定路径下的所有文件 并且进行文件名与路径的拼接

    Args:
        folder_path_ (str): 指定目录绝对路径 
            * 如果后续需要紧接着使用函数get_file_path进行文件名与路径的拼接 则需要本参数以'/'皆为
        suffix_ (str): 筛选文件的后缀名 
            * 以'.'开头
            * 如果不指定文件后缀 则选出所有文件
        dump_path_ (str): 将找出的文件路径写入文件的路径
            * '.yml'文件
        dump_data_type_ (type): 保存到本地文件的数据类型
            * 默认为dict 可选为dict list
    Returns:
        dict: 文件:文件路径组成的字典
    '''
    tmp = get_file_names(folder_path_=folder_path_, suffix_=suffix_)
    if dump_data_type_ == dict:
        res_ = {}
        for i in tmp:
            res_[i] = folder_path_ + i
        if dump_path_ is None:
            pass
        else:
            dump_yaml(dump_path_, res_, _overw)
        return res_
    elif dump_data_type_ == list:
        res_ = []
        for i in tmp:
            res_.append(folder_path_ + i)
        if dump_path_ is None:
            pass
        else:
            dump_yaml(dump_path_, res_, _overw)
        return res_

def get_subfolder_name(path_:str) -> list:
    '''读取一个文件夹内的所有文件夹名

    Args:
        path_ (str): 母文件夹路径

    Returns:
        list: 子文件夹名
    '''
    assert os.path.exists(path_), cstr(f"{path_} Not Exists", 0)
    #定义一个列表，用来存储结果
    list_ = []
    #获取该目录下的所有文件或文件夹目录
    files = os.listdir(path_)
    for file in files:
        #得到该文件下所有目录的路径
        m = os.path.join(path_,file)
        #判断该路径下是否是文件夹
        if (os.path.isdir(m)):
            h = os.path.split(m)
            list_.append(h[1])
    return list_

def read_file(path_:str, strip_ = False) -> list:
    '''按行读取一个给定路径的文件

    Args:
        path_ (str): 文件路径
        strip_ (bool, optional): 是否需要去掉每一行最后的换行符号. Defaults to False.

    Returns:
        list: 文件内容 按行组成列表
    '''
    if os.path.exists(path=path_):pass
    else:
        print(cstr(path_ + " Not Exist", 0))
        return None
    with open(path_, 'r') as file_:
        lines  =  file_.readlines()
    if strip_:lines = list(map(lambda x : x.strip(), lines))
    return lines

def sort_list(data_:list, head_ = None, tail_ = None, start_ = 1) -> list:
    '''对一个给定列表（通常是从文件夹内读取到的文件名列表）进行重新排序

    Args:
        data_ (list): 需要排序的列表
        head_ (str, optional): 列表内元素的同一开头. Defaults to None.
        tail_ (str, optional): 列表内元素的统一结尾. Defaults to None.
        start_ (int, optional): 列表内元素的最开始编号. Defaults to 1.

    Returns:
        list: _description_
    '''
    if head_ is not None and tail_ is None:
        res = []
        err_ = []
        for i in range(start_, len(data_) + 1):
            tar_ = head_ + str(i)
            if tar_ in data_: res.append(tar_)
            else: err_.append(tar_)
        return res, err_
    elif head_ is None and tail_ is not None:
        res = []
        err_ = []
        for i in range(start_, len(data_) + 1):
            tar_ = str(i) + head_
            if tar_ in data_: res.append(tar_)
            else: err_.append(tar_)
        return res, err_
    else: return None

def wirte_file(path_:str, data_, mode_ = 'w'):
    '''将数据按行写入文件

    Args:
        path_ (str): 待写入文件路径
        data_ (_type_): 待写入数据
        mode_ (str, optional): _description_. Defaults to 'w'.
    '''
    with open(path_, mode_) as file_:
        # 如果数据为字符串
        if isinstance(data_,str):
            file_.write(data_)
        elif isinstance(data_, int) or isinstance(data_, float):
            file_.write(str(data_))
        elif isinstance(data_, list) or isinstance(data_, tuple):
            for i in data_:
                file_.write(str(i) + "\n")

def load_data_singular(data_path_:str) -> list:
    '''单图模型的数据加载

    Args:
        data_path_ (str): 数据路径

    Returns:
        list: _description_
    '''
    original_data = np.loadtxt(data_path_, delimiter=',')   
    # 切片取出所有的features
    features = original_data[0:, 0:-1]
    # 切片取出所有的疏散时间
    true_values = original_data[0:,-1]
    # 数据总数
    total_adta_length = len(true_values)
    return features, true_values, total_adta_length

def concat_features(original_features_:np.ndarray, add_features_:np.ndarray) -> np.ndarray:
    res = []
    for ori_feature in original_features_:
        tmp_ = ori_feature
        if len(add_features_.shape) == 2:
            for add_feature in add_features_:
                tmp_ = np.vstack((tmp_, add_feature))
        elif  len(add_features_.shape) == 1:
            tmp_ = np.vstack((tmp_, add_features_))
        else:
            raise Exception
        res.append(tmp_.T.tolist())
    features = np.array(res)
    return features

def cal_gso(adj:sp._csc.csc_matrix):
    # 首先保证传入的adj是对称的
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 重归一化 在实际操作中不常使用D^(-1/2)AD^(-1/2),而是用D(-1)A代替
    # step1 添加自连边
    adj_selfcon = adj + sp.eye(adj.shape[0])
    # step2 创建度矩阵
    row_sum = np.array(adj_selfcon.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    gso_selfcon = r_mat_inv.dot(adj_selfcon)
    # 不存在自连边
    row_sum = np.array(adj.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    gso = r_mat_inv.dot(adj)
    return gso_selfcon

def split_data_by_case(total_case_length:int, val_and_test_rate:float):
    len_val = int(math.floor(total_case_length * val_and_test_rate))
    len_test = int(math.floor(total_case_length * val_and_test_rate))
    len_train = int(total_case_length - len_val - len_test)
    print(f"Total Data:{total_case_length}, Train:{len_train}, Validation:{len_val}, Test:{len_test}")
    return len_train, len_test, len_val

def split_data_by_demo(total_demo_length:int, val_and_test_rate:int, simulation_time:int):
    '''根据demo个数来划分训练 测试 验证集的长度

    Args:
        total_demo_length (int): 总的demo个数
        val_and_test_rate (int): 验证集与测试集占总demo个数的比例
        simulation_time (int): 每一个demo的模拟次数
    Returns:
        len_demo_train (int): 训练集所需的demo个数
        len_demo_val_and_test (int): 验证集与测试集所需的demo个数
        len_case_train (int): 训练集所需的case个数
        len_case_val (int): 验证集与测试集所需的demo个数
    '''
    # 验证集与测试集所需的demo个数 向下取整数
    len_demo_val_and_test = int(math.floor(total_demo_length * val_and_test_rate))
    # 训练集中所需的demo个数 在上一步向下取整数的基础上相当于这一步实际上已经向上取整数了
    len_demo_train = total_demo_length - len_demo_val_and_test * 2
    # 训练集case个数
    len_case_train = len_demo_train * simulation_time
    # 验证集与测试集所需的case个数
    len_case_val = len_demo_val_and_test * simulation_time
    return len_demo_train, len_demo_val_and_test, len_case_train, len_case_val

def encapsulate_data_singular(features:np.ndarray, true_values:np.ndarray, len_train:int, len_val:int, device, batch_size):
    assert len(features) == len(true_values), f"The length of Features:{len(features)} unequal to True_values{len(true_values)}"
    train_features = torch.Tensor(features[0:len_train]).to(device)
    train_true_vars = torch.Tensor(true_values[0:len_train]).to(device)
    train_data = utils.data.TensorDataset(train_features, train_true_vars)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_features = torch.Tensor(features[len_train:len_train + len_val]).to(device)
    val_true_vars = torch.Tensor(true_values[len_train:len_train + len_val]).to(device)
    val_data = utils.data.TensorDataset(val_features, val_true_vars)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_features = torch.Tensor(features[len_train + len_val:]).to(device)
    test_true_vars = torch.Tensor(true_values[len_train + len_val:]).to(device)
    test_data = utils.data.TensorDataset(test_features, test_true_vars)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return train_iter, val_iter, test_iter

def encapsulate_data_plural(adj:torch.tensor, feature:torch.tensor, time:torch.tensor, len_demo_train:int, len_demo_val:int, len_case_train:int, len_case_val, device:str, batch_size:int):
    '''对concat_data拼接好的数据进行封装

    Args:
        adj (torch.tensor): 存放adj邻接矩阵 第一维为demo的个数
        feature (torch.tensor): 存放adj邻接矩阵及房间的feature矩阵 第一维为case的个数
        time (torch.tensor): 存放每一个case的疏散时间 第一维为case个数
        len_demo_train (int): 训练集中需要的demo个数
        len_demo_val (int): 测试与验证集需要的demo个数  
        len_case_train (int): 训练集需要的case个数
        len_case_val (_type_): 测试与验证集需要的case个数
        device (str): _description_
        batch_size (int): _description_

    Returns:
        train_iter (torch.utils.data.dataloader.DataLoader):
        val_iter (torch.utils.data.dataloader.DataLoader):
        test_iter (torch.utils.data.dataloader.DataLoader):
    '''
    
    # 根据demos将所有数据分段
    train_case = GCNDataset(adj = adj[0:len_demo_train], 
                             feature=feature[0:len_case_train], 
                             time=time[0:len_case_train])
    train_iter = utils.data.DataLoader(dataset=train_case, batch_size=batch_size, shuffle=True)

    val_case = GCNDataset(adj=adj[len_demo_val : len_demo_train + len_demo_val ],
                           feature=feature[len_case_train:len_case_train + len_case_val], 
                           time=time[len_case_train:len_case_train + len_case_val])
    val_iter = utils.data.DataLoader(dataset=val_case, batch_size=batch_size, shuffle=False)

    test_case = GCNDataset(adj=adj[len_demo_train + len_demo_val:],
                            feature=feature[len_case_train + len_case_val:],
                            time=time[len_case_train + len_case_val:])
    test_iter = utils.data.DataLoader(dataset=test_case, batch_size=batch_size, shuffle=False)
    
    
    # 根据demos将所有数据分段
    '''
    train_case = GCNDataset(adj = adj[len_demo_val + len_demo_val : ], 
                             feature=feature[len_case_val + len_case_val: ], 
                             time=time[len_case_val + len_case_val : ])
    train_iter = utils.data.DataLoader(dataset=train_case, batch_size=batch_size, shuffle=True)

    val_case = GCNDataset(adj=adj[0 : len_demo_val ],
                           feature=feature[0 : len_case_val], 
                           time=time[0 : len_case_val])
    val_iter = utils.data.DataLoader(dataset=val_case, batch_size=batch_size, shuffle=False)

    test_case = GCNDataset(adj=adj[len_demo_val : len_demo_val + len_demo_val],
                            feature=feature[len_case_val : len_case_val + len_case_val],
                            time=time[len_case_val : len_case_val + len_case_val])
    test_iter = utils.data.DataLoader(dataset=test_case, batch_size=batch_size, shuffle=False)
    '''

    return train_iter, val_iter, test_iter

def encapsulate_data_by_simulation_case(sims:list, len_train, len_val, batch_size, device):
    train_case = GCNDataset2(sims=sims[0:len_train], device=device)
    train_iter = utils.data.DataLoader(dataset=train_case, batch_size=batch_size, shuffle=False)
    val_case = GCNDataset2(sims=sims[len_train : len_train + len_val], device=device)
    val_iter = utils.data.DataLoader(dataset=val_case, batch_size=batch_size, shuffle=False)
    test_case = GCNDataset2(sims=sims[len_train + len_val:], device=device)
    test_iter = utils.data.DataLoader(dataset=test_case, batch_size=batch_size, shuffle=False)
    return train_iter, val_iter, test_iter

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_all_adj(npz_dataset_path:str, filenames:list, device = None) -> dict:
    # 检查用于存放npz文件的文件夹是否存在
    assert os.path.exists(npz_dataset_path), cstr(npz_dataset_path + " Not Exists", 0)
    # 定义一个返回值
    res = {}
    # 迭代所有的filename
    for name in filenames:
        npz_path = npz_dataset_path + name + '\\adj.npz'
        # 检查当前npz文件是否存在
        assert os.path.exists(npz_path), cstr(npz_path + " Not Exists", 0)
        # 加载npz文件内的adj矩阵
        adj, _ = load_adj(npz_path)
        # 计算gso
        gso = cal_gso(adj)
        # 将gso转换为np array
        gso = gso.toarray().astype(dtype=np.float32)
        # 将gso转换为tensor
        # gso = torch.from_numpy(gso).to(device)
        # 将存储为tensor的gso添加到函数返回值
        res[name] = gso
    return res

def load_all_ped_time(npz_dataset_path:str, filenames:list) -> dict:
    '''加载所有的ped_time.csv文件 该文件中存放了每一个case的人员分布和疏散时间

    Args:
        npz_dataset_path (str): _description_
        filenames (list): _description_

    Returns:
        dict: _description_
    '''
    # 检查用于存放npz文件的文件夹是否存在
    assert os.path.exists(npz_dataset_path), cstr(npz_dataset_path + " Not Exists", 0)
    # 定义一个返回值
    res = {}
    # 迭代所有的filename
    for name in filenames:
        npz_path = npz_dataset_path + name + '\\ped_time.csv'
        # 检查当前csv文件是否存在
        assert os.path.exists(npz_path), cstr(npz_path + " Not Exists", 0)
        res[name] = np.array(pd.read_csv(npz_path, header=None)).tolist()
    return res

def load_all_csv_data_by_filename(dataset:str, demos:list, target_dataname:str) -> dict:
    # 检查dataset是否存在
    assert os.path.exists(dataset)
    res ={}
    load_completely = True
    for demo in demos:
        # 定义当前demo内该文件的路径
        filepath = dataset + demo + "\\" + target_dataname
        # 确认文件存在
        if check_path(filepath=filepath):
            res[demo] = np.array(pd.read_csv(filepath, header=None))
        else:
            load_completely = False
            print(cstr(f"{filepath} Not Exists", 0))
    return res, load_completely
       
def load_all_time(time_dataset_path:str, demonames:list) -> dict:
    # 检查用于存放npz文件的文件夹是否存在
    assert os.path.exists(time_dataset_path), cstr(time_dataset_path + " Not Exists", 0)
    # 定义一个返回值
    res = {}
    # 迭代所有的filename
    for name in demonames:
        npz_path = time_dataset_path + name + '\\time.npz'
        # 检查当前npz文件是否存在
        assert os.path.exists(npz_path), cstr(npz_path + " Not Exists", 0)

def expand_adj(adj):pass

def load_all_area(area_dataset_path:str, filenames:list) -> dict:
    # 检查用于存放npz文件的文件夹是否存在
    assert os.path.exists(area_dataset_path), cstr(area_dataset_path + " Not Exists", 0)
    # 定义一个返回值
    res = {}
    # 迭代所有的filename
    for name in filenames:
        area_path = area_dataset_path + name + '\\area.csv'
        # 检查当前csv文件是否存在
        assert os.path.exists(area_path), cstr(area_path + " Not Exists", 0)
        res[name] = np.array(pd.read_csv(area_path, header=None)).tolist()
    return res

def calculate_adj_num(adj:sp._csc.csc_matrix) -> list:
    # 0 转换为密集矩阵
    adj = adj.todense() # numpy.matrix
    res = [[],[]]
    # 迭代所有的行
    rows = adj.shape[0]
    for row in range(rows):
        # 定义一个adj个数和adj值
        adj_num, adj_value = 0, 0
        # 取出当前位置的行转换为list
        cur_row = adj[row].tolist()[0]
        # 迭代行里的值
        for val in cur_row:
            # 如果值>0 则将adj个数+1 将adj值加上该值
            if val > 0:
                adj_num += 1
                adj_value += val
            else:pass
        # 迭代结束后将adj个数和adj值添加到res
        res[0].append(adj_num)
        res[1].append(adj_value)
    return res, adj

def compare_list(list1:list, list2:list) -> bool:
    '''比较两个列表是否完全一致

    Args:
        list1 (list): _description_
        list2 (list): _description_

    Returns:
        bool: _description_
    '''
    if len(list1) == len(list2):
        for i in range(len(list1)):
            if list1[i] == list2[i]:pass
            else:return False
        return True
    else:return False

def load_data(dataset:str, target_datanames:list, ):
    # 确认dataset存在
    assert os.path.exists(dataset), cstr(f"dataset: {dataset} Not Exists", 0)
    # 加载dataset内的所有demo名 即子文件夹名
    demonames = get_subfolder_name(dataset)
    # 加载所有的adj邻接矩阵 adj.npz文件
    adjs = load_all_adj(dataset, demonames)
    # 加载所有的目标数据
    data = {'adj': adjs}
    for filename in target_datanames:
        loaded_data, loaded_completely =load_all_csv_data_by_filename(dataset=dataset, demos=demonames, target_dataname=filename)
        data[filename.split('.')[0]] = loaded_data
        if loaded_completely:print(cstr((f"{filename} Loaded Completely.."), 1))
        else: print(cstr(f"{filename} Missing File..", 0))
    return data

    # return train_iter, val_iter, test_iter, gsos, len_train, len_val, len_test, max_vertex



def concat_data(data:dict, device:str):
    '''将从本地读取到的数据如人数 面积 连通个数 进行拼接

    Args:
        data (dict): _description_
    '''
    # 取出adj数据
    adj = data['adj']
    # 首先找出最大的节点个数
    '''
     vertexes = []
    for key, item in adj.items():
        vertexes.append(item.shape[0])
    max_v = max(vertexes)
    '''
    max_v = 512
    # 迭代所有的demos
    demos = adj.keys()
    res_adj = []
    res_feature = []
    res_time = []
    for demo in demos:
        # 获取原始的原始的adj矩阵
        original_adj = adj[demo]
        # 原始adj矩阵的节点个数
        original_v = original_adj.shape[0]
        # 计算需要扩充的节点个数
        pad_v = max_v - original_v
        # 对原始adj矩阵进行pad
        pad_adj =  np.pad(original_adj,(0,pad_v),'constant') 
        res_adj.append(pad_adj)
        # 对ped数据进行扩充 首先取出当前的ped数据
        original_ped = data['ped_t'][demo]  # 行数为节点个数
        pad_ped = np.vstack((original_ped,np.zeros((pad_v, original_ped.shape[1]))))    # 在原始ped数据下方添加一个0矩阵 使得行数统一
        # 取出原始的面积数据   
        original_area = data['area'][demo]   # 行数为节点个数
        pad_area= np.vstack((original_area,np.zeros((pad_v, original_area.shape[1]))))
        # 取出原始的矩阵feature
        original_adj_feature = data['adj_feature'][demo]    # 行数为节点个数
        pad_adj_feature = np.vstack((original_adj_feature,np.zeros((pad_v, original_adj_feature.shape[1]))))
        # 将面积 adj_feature 和不同的人数分布拼接到一起
        for i in range(pad_ped.shape[1]):
            try:
                tmp = np.hstack((pad_area, pad_adj_feature, pad_ped[:,i].reshape((max_v, 1))))
                if tmp.shape[1] != 5:print(demo)
                res_feature.append(tmp)
            except:print(cstr(f"{demo}", 0))
        # 取出原始的time
        res_time = res_time + data['time_c'][demo][:,0].tolist()
    res_adj = torch.Tensor(res_adj).to(device)
    res_feature = torch.Tensor(res_feature).to(device)
    res_time = torch.Tensor(res_time).to(device)
    return res_adj, res_feature, res_time

def concat_data_case(data:dict, device:str):
    # 取出adj数据
    adj = data['adj']
    # 首先找出最大的节点个数
    '''vertexes = []
    for key, item in adj.items():
        vertexes.append(item.shape[0])
    max_v = max(vertexes)'''
    max_v = 512
    # 迭代所有的demos
    demos = adj.keys()
    sim = []
    for demo in demos:
        # 获取原始的原始的adj矩阵
        original_adj = adj[demo]
        # 取出ped数据
        original_ped = data['ped_t'][demo]  # 行数为节点个数
        # 取出area数据
        original_area = data['area'][demo]   # 行数为节点个数
        # 取出原始的矩阵feature
        original_adj_feature = data['adj_feature'][demo]    # 行数为节点个数
        # 取出时间数据
        res_time = data['time_c'][demo][:,0].tolist()
        assert len(res_time) == original_ped.shape[1], cstr(f"{demo} Has Wrong Simulation Times")
        # 迭代所有的ped分布
        for i in range(original_ped.shape[1]):
            case = SimulationCase(f"{demo}_{i}", original_adj, original_adj_feature, original_ped[:,i], res_time[i], original_area)
            case.pad(max_v)
            sim.append(case)
    random.shuffle(sim)
    return sim
        
def load_pkl_data(dataset:str, max_v = 512):
    check_path(dataset)
    filenames = get_file_names(dataset, suffix_='.pkl')
    sim = []
    for file in filenames:
        path = dataset + file
        data = pickle.load(open(path, 'rb'))
        # 将case数据封装为simulationcase类对象
        case = SimulationCase(id=None, adj=data['adj'], feature=data['adj_feature'], ped=data['ped'], time=data['time'], area=data['area'])
        case.pad(max_v)
        sim.append(case)
    random.shuffle(sim)
    return sim
        
def contain(original:list|tuple, target):
    # 如果传入的是可迭代对象
    if hasattr(target, '__iter__') and isinstance(target, str) is not True:
        # 如果传入对象中还有元素
        if len(target) > 0:
            # 则取出最后一个对象传入本函数进行判断
            flag = contain(original=original, target=target[-1])    
            if flag:    # 如果上一步返回true 则将传入对象去掉最后一个元素后继续传入本函数
                target.pop()
                
                tmp = contain(original=original, target=target)
                return tmp
            else: return False  # 如果上一步返回false 则直接本函数返回false
        else:   # 如果已经完成迭代 则返回true
            return True
    # 如果传入的是单一对象
    else:
        for i in original:
            if target == i:return True
            else: pass
        return False
        
        

def duplicata_file(original_filepath:str, op_path = None) -> bool:
    # 原文件是否存在
    if os.path.exists(original_filepath):
        filename = original_filepath.split('\\')[-1].split('.')[0]
        op_path = op_path if op_path is not None else original_filepath.replace(filename, filename + '_c')
        shutil.copy(original_filepath, op_path)
    else: 
        print(cstr(f"{original_filepath} Not Exists", 0))
        return False

def remove_item(original_list:list, remove_target) -> bool:
    # 尝试去除目标元素
    try:
        original_list.remove(remove_target)
        return True
    # 去除失败
    except:
        return False


def remove_items(original_list:list, remove_target:list) -> bool:
    for target in remove_target:
        remove_item(original_list=original_list, remove_target=target)


if __name__ == "__main__":
    dataset = "A:\\PyTorch\\MyGcn\\Data\\Dataset\\Part3\\0\\"
    sim = load_pkl_data(dataset=dataset)


    '''
    dataset="A:\\PyTorch\\MyGcn\\Data\\Dataset\\"
    demos = get_subfolder_name(path_=dataset)
    res = load_data(dataset=dataset, target_datanames=['adj_feature.csv', 'ped_t.csv', 'area.csv', 'time.csv'])
    '''
    