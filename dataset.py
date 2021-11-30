import os.path
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import dgl
import torch

class BaseDataset(object):
    def __init__(self, trainpath, testpath, statpath, validpath):
        """base Dataset. Read data files and preprocess.
        Args:
            trainpath: File path of train Data;
            testpath: File path of test data;
            statpath: File path of entities num and relatioins num;
            validpath: File path of valid data
        """
        self.trainQuadruples = self.load_quadruples(trainpath)  # 训练集四元组，List
        self.testQuadruples = self.load_quadruples(testpath)  # 测试集四元组，List
        self.validQuadruples = self.load_quadruples(validpath)  # 验证集四元组，List
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        self.num_e, self.num_r = self.get_total_number(statpath)  # number of entities, number of relations
        self.skip_dict = self.get_skipdict(self.allQuadruples)   # (s, r, t) -> 正确答案实体集合

        self.train_entities = set()  # Entities that have appeared in the training set
        for query in self.trainQuadruples:
            self.train_entities.add(query[0])
            self.train_entities.add(query[2])

        self.train_snapshots = self.split_by_time(self.trainQuadruples)  # 训练snapshots, list, 每个元素包含，（图三元组，时间戳）
        self.valid_snapshots = self.split_by_time(self.validQuadruples)  # 验证集同上
        self.test_snapshots = self.split_by_time(self.testQuadruples)   # 测试集同上

    def get_all_timestamps(self):
        """Get all the timestamps in the dataset.
        return:
            timestamps: a set of timestamps.
        """
        timestamps = set()
        for ex in self.allQuadruples:
            timestamps.add(ex[3])
        return timestamps

    def get_skipdict(self, quadruples):
        """Used for time-dependent filtered metrics.
        return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
        """
        filters = defaultdict(set)
        for src, rel, dst, time in quadruples:
            filters[(src, rel, time)].add(dst)
            filters[(dst, rel+self.num_r+1, time)].add(src)
        return filters

    @staticmethod
    def load_quadruples(inpath):
        """train.txt/valid.txt/test.txt reader
        inpath: File path. train.txt, valid.txt or test.txt of a dataset;
        return:
            quadrupleList: A list
            containing all quadruples([subject/headEntity, relation, object/tailEntity, timestamp]) in the file.
        """
        with open(inpath, 'r') as f:
            quadrupleList = []
            for line in f:
                try:
                    line_split = line.split()
                    head = int(line_split[0])
                    rel = int(line_split[1])
                    tail = int(line_split[2])
                    time = int(line_split[3])
                    quadrupleList.append([head, rel, tail, time])
                except:
                    print(line)
        return quadrupleList

    @staticmethod
    def get_total_number(statpath):
        """stat.txt reader
        return:
            (number of entities -> int, number of relations -> int)
        """
        with open(statpath, 'r') as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])

    @staticmethod
    def split_by_time(data):
        # 时间快照图，返回list, 每个元素是个元组，包括图、时间戳
        snapshot_list = []
        snapshot = []
        latest_t = 0
        for i in range(len(data)):
            t = data[i][3]
            train = data[i]
            # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
            if latest_t != t:  # 同一时刻发生的三元组
                if len(snapshot):
                    snapshot_list.append((np.array(snapshot).copy(), latest_t))
                snapshot = []
                latest_t = t
            snapshot.append(train[:3])
        # 加入最后一个shapshot
        if len(snapshot) > 0:
            snapshot_list.append((np.array(snapshot).copy(), latest_t))
        return snapshot_list

    @staticmethod
    def sanity_check(snapshot_list):
        nodes = []
        rels = []
        for snapshot, timestamp in snapshot_list:
            uniq_v, edges = np.unique((snapshot[:, 0], snapshot[:, 2]), return_inverse=True)  # relabel
            uniq_r = np.unique(snapshot[:, 1])
            nodes.append(len(uniq_v))
            rels.append(len(uniq_r) * 2)
        print(
            "# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}"
                .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list),
                        max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list])))


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, snapshots, start_idx, n_ent, n_rel, history_len, device):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.history_len = history_len  # 用于建模的历史快照长度
        self.start_idx = start_idx  # 用于预测的开始时间戳idx
        self.snapshots_num = len(snapshots)
        self.snapshots = snapshots
        self.device = device
        self.dgl_graphs = [self.build_sub_graph(n_ent, n_rel, g, device) for g, _ in snapshots]

    def __len__(self):
        return self.snapshots_num - self.start_idx

    def __getitem__(self, idx):
        predict_snapshots, timestamp = self.snapshots[idx + 1]
        if idx - self.history_len < 0:
            graph_list = self.dgl_graphs[:idx]
        else:
            graph_list = self.dgl_graphs[idx - self.history_len: idx]
        query_entites = torch.tensor(predict_snapshots[:, 0], device=self.device)
        query_relations = torch.tensor(predict_snapshots[:, 1], device=self.device)
        query_timestamps = torch.ones_like(query_entites) * timestamp
        answers = torch.tensor(predict_snapshots[:, 2], device=self.device)
        return graph_list, query_entites, query_relations, query_timestamps, answers

    def build_sub_graph(self, num_nodes, num_rels, triples, device):
        # 针对每一个snapshot, 构建dgl图
        src, rel, dst = triples.transpose()
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + num_rels)) # 加入取反边
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(src, dst)
        norm = self.comp_deg_norm(g)
        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        g.edata['type'] = torch.LongTensor(rel)
        return g.to(device)

    def comp_deg_norm(self, g):
        # 计算图中节点的度正则项
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm


if __name__ == '__main__':
    ####  基本数据处理测试
    data = 'data/ICEWS14'
    trainpath = os.path.join(data, 'train.txt')
    validpath = os.path.join(data, 'valid.txt')
    testpath = os.path.join(data, 'test.txt')
    statpath = os.path.join(data, 'stat.txt')
    baseDataset = BaseDataset(trainpath, testpath, statpath, validpath)
    baseDataset.sanity_check(baseDataset.train_snapshots)
    baseDataset.sanity_check(baseDataset.test_snapshots)
    baseDataset.sanity_check(baseDataset.valid_snapshots)

    # dataset 测试
    train_graphdataset = GraphDataset(baseDataset.train_snapshots, 0, baseDataset.num_e, baseDataset.num_r, 10)
    print('-----train-----')
    for graph_list, query_entities, query_relations, query_timestamps, answers in train_graphdataset:
        print(len(graph_list))
        print(query_entities.shape)
        print(query_relations.shape)
        print(query_timestamps.shape)
        print(answers.shape)

    time_span = 24
    print('-----valid-----')
    valid_start_idx = baseDataset.valid_snapshots[0][1] // time_span
    valid_graphdataset = GraphDataset(baseDataset.train_snapshots, valid_start_idx, baseDataset.num_e, baseDataset.num_r, 10)
    for graph_list, query_entities, query_relations, query_timestamps, answers in valid_graphdataset:
        print(len(graph_list))
        print(query_entities.shape)
        print(query_relations.shape)
        print(query_timestamps.shape)
        print(answers.shape)

    time_span = 24
    print('-----test-----')
    test_start_idx = baseDataset.test_snapshots[0][1] // time_span
    test_graphdataset = GraphDataset(baseDataset.train_snapshots, test_start_idx, baseDataset.num_e, baseDataset.num_r, 10)
    for graph_list, query_entities, query_relations, query_timestamps, answers in test_graphdataset:
        print(len(graph_list))
        print(query_entities.shape)
        print(query_relations.shape)
        print(query_timestamps.shape)
        print(answers.shape)
