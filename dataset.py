import os.path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
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
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
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
    def get_reverse_quadruples_array(quadruples, num_r):
        quads = np.array(quadruples)
        quads_r = np.zeros_like(quads)
        quads_r[:, 1] = num_r + quads[:, 1]
        quads_r[:, 0] = quads[:, 2]
        quads_r[:, 2] = quads[:, 0]
        quads_r[:, 3] = quads[:, 3]
        return np.concatenate((quads, quads_r))

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


class DGLGraphDataset(object):
    def __init__(self, snapshots, n_ent, n_rel):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.snapshots_num = len(snapshots)
        self.snapshots = snapshots
        self.dgl_graphs = [self.build_sub_graph(n_ent, n_rel, g, time) for g, time in snapshots]
        self.dgl_graphs.insert(0, self.build_sub_graph(n_ent, n_rel, np.array([]), 0))  # 添加PAD Graph, 方便后面操作

    def build_sub_graph(self, num_nodes, num_rels, triples, time):
        # 针对每一个snapshot, 构建dgl图
        if triples.size != 0:
            src, rel, dst = triples.transpose()
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + num_rels))  # 加入取反边
        else:
            src, rel, dst = np.array([]), np.array([]), np.array([])
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(src, dst)
        norm = self.comp_deg_norm(g)
        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        g.edata['type'] = torch.LongTensor(rel)
        g.edata['timestamp'] = torch.LongTensor(torch.ones_like(g.edata['type']) * time)
        return g

    def comp_deg_norm(self, g):
        # 计算图中节点的度正则项
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm


class QuadruplesDataset(Dataset):
    def __init__(self, quadruples, history_len, time_span):
        self.quadruples = quadruples  # 四元组数组，np.array, [quad_num, 4] (sub, rel, obj, time)
        self.history_len = history_len  # 预测答案依据的历史序列长度
        self.time_span = time_span  # TKG的时间跨度，ICEWS是24

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        quad = self.quadruples[idx]
        sub, rel, obj, time = quad[0], quad[1], quad[2], quad[3]
        time = time // self.time_span
        # 选择最近历史图的idx, 这里可以用其他的sample方法代替, 因为Graph list头部添加了一个空白图，所以Graph idx数值小于等于预测时间戳即可
        history_list = [max(0, time - i) for i in range(self.history_len)]
        return torch.tensor(sub), torch.tensor(rel), torch.tensor(obj), torch.tensor(time), torch.tensor(history_list)

    @staticmethod
    def collate_fn(data):
        sub = torch.stack([_[0] for _ in data], dim=0)
        rel = torch.stack([_[1] for _ in data], dim=0)
        obj = torch.stack([_[2] for _ in data], dim=0)
        time = torch.stack([_[3] for _ in data], dim=0)
        history_list = torch.stack([_[4] for _ in data], dim=0)
        uniq_history, history_align = np.unique(history_list.numpy(), return_inverse=True)  # 独立的历史
        history_align = history_align.reshape([sub.shape[0], -1])
        # graphs = [dgl_graphs[idx] for idx in uniq_history]
        return sub, rel, obj, time, torch.tensor(history_align), uniq_history


class QuadsInputByTimesDataset(Dataset):
    def __init__(self, timeIDs, dglDataset, seq_len, num_sample_triples=0):
        self.timeIDs = timeIDs  # 用于训练的时间戳ID
        self.dglDataset = dglDataset
        self.snapshots = [(self.get_reverse_triples(self.dglDataset.n_rel, g), t) for g, t in self.dglDataset.snapshots]
        self.seq_len = seq_len
        self.num_sample_triples = num_sample_triples

    def __len__(self):
        return len(self.timeIDs)

    def __getitem__(self, idx):
        timeid = self.timeIDs[idx]
        predict_triples, predict_t = self.snapshots[timeid]
        seq_graphs = [self.dglDataset.dgl_graphs[max(0, timeid-self.seq_len+1+i)] for i in range(self.seq_len)]
        if self.num_sample_triples:
            predict_triples = self.triples_sample(predict_triples)
        predict_s = torch.tensor(predict_triples[:, 0])
        predict_p = torch.tensor(predict_triples[:, 1])
        predict_o = torch.tensor(predict_triples[:, 2])
        predict_t = torch.ones_like(predict_s) * predict_t
        return predict_s, predict_p, predict_o, predict_t, seq_graphs

    def get_reverse_triples(self, num_r, triples):
        triples_r = np.zeros_like(triples)
        triples_r[:, 1] = num_r + triples[:, 1]
        triples_r[:, 0] = triples[:, 2]
        triples_r[:, 2] = triples[:, 0]
        return np.concatenate((triples, triples_r))

    def triples_sample(self, triples):
        # 对三元组采样
        if self.num_sample_triples < len(triples):
            index = np.random.choice(range(len(triples)), self.num_sample_triples, False)
            triples = triples[index]
        return triples

    @staticmethod
    def collate_fn(data):
        predict_s, predict_p, predict_o, predict_t, seq_graphs = data[0]
        return predict_s, predict_p, predict_o, predict_t, seq_graphs


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

    dglGraphDataset = DGLGraphDataset(baseDataset.train_snapshots + baseDataset.valid_snapshots + baseDataset.test_snapshots,
                                      baseDataset.num_e, baseDataset.num_r)
    trainTimes = list(range(len(baseDataset.train_snapshots)))
    quadsInputByTimeDataset = QuadsInputByTimesDataset(trainTimes, dglGraphDataset, 10)

    dataloader = DataLoader(
        quadsInputByTimeDataset,
        shuffle=True,
        batch_size=1,
        num_workers=2,
        collate_fn=QuadsInputByTimesDataset.collate_fn,
    )

    for predict_s, predict_p, predict_o, predict_t, seq_graphs in dataloader:
        break

    testTimes = list(range(len(baseDataset.train_snapshots + baseDataset.valid_snapshots),
                           len(baseDataset.train_snapshots + baseDataset.valid_snapshots + baseDataset.test_snapshots)))
    testQuadsInputByTimeDataset = QuadsInputByTimesDataset(testTimes, dglGraphDataset, 5)

    testDataLoader = DataLoader(
        testQuadsInputByTimeDataset,
        batch_size=1,
        num_workers=2,
        collate_fn=QuadsInputByTimesDataset.collate_fn,
    )

    for predict_s, predict_p, predict_o, predict_t, seq_graphs in testDataLoader:
        print(predict_t[0])
        print(seq_graphs[-1].edata['timestamp'][0])
