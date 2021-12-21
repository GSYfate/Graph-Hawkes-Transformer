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

        self.train_snapshots = self.split_by_time(self.trainQuadruples)  # 训练snapshots, , 每个元素包含，（图三元组，时间戳）
        self.valid_snapshots = self.split_by_time(self.validQuadruples)  # 验证集同上
        self.test_snapshots = self.split_by_time(self.testQuadruples)   # 测试集同上

        self.time_inverted_index_dict = self.get_time_inverted_index_dict(self.allQuadruples)

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
            filters[(dst, rel+self.num_r, time)].add(src)
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

    def get_time_inverted_index_dict(self, quadruples):
        """获取实体出现过的时间的倒排索引索引表，(e, r)出现过的时间索引表, 时间从小到大排序"""
        index_dict = defaultdict(set)
        for quad in quadruples:
            index_dict[quad[0]].add(quad[3])
            index_dict[quad[2]].add(quad[3])
            index_dict[(quad[0], quad[1])].add(quad[3])
            index_dict[(quad[2], quad[1] + self.num_r)].add(quad[3])
        for k, v in index_dict.items():
            index_dict[k] = sorted(list(v))

        return index_dict


class DGLGraphDataset(object):
    def __init__(self, snapshots, n_ent, n_rel):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.snapshots_num = len(snapshots)
        self.snapshots = snapshots
        # dgl_graph_dict: key是时间，value是带所有节点的时间图，-1表示空图， dgl_graphs是它的list版本, 首元素为空图
        self.dgl_graph_dict, self.dgl_graphs = self.get_dglGraph_dict(snapshots)

    def get_dglGraph_dict(self, snapshots):
        dgl_graph_dict = {}
        dgl_graph = []
        for (g, time) in snapshots:
            graph = self.build_sub_graph(self.n_ent, self.n_rel, g, time)
            dgl_graph_dict[time] = graph
            dgl_graph.append(graph)
        PAD_graph = self.build_sub_graph(self.n_ent, self.n_rel, np.array([]), 0)
        dgl_graph_dict[-1] = PAD_graph
        dgl_graph.insert(0, PAD_graph)
        return dgl_graph_dict, dgl_graph

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

        uniq_r, r_len, r_to_e = self.r2e(triples)
        g.uniq_r = uniq_r
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
        g.r_len = r_len
        return g

    def comp_deg_norm(self, g):
        # 计算图中节点的度正则项
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    def get_nhop_subgraph(self, time, nodes, n=2):
        """获取某组节点在某个时间的n步邻居子图
        time: 时间，绝对值
        nodes: 节点id, list
        n: N步邻居
        """
        g = self.dgl_graph_dict[time]  # time时刻的图
        total_nodes = set(nodes)
        for i in range(n):
            step_nodes = total_nodes.copy()
            for node in step_nodes:
                neighbor_n, _ = g.in_edges(node)
                neighbor_n = set(neighbor_n.tolist())
                total_nodes |= neighbor_n
        sub_g = g.subgraph(list(total_nodes))
        sub_g.ndata['norm'] = self.comp_deg_norm(sub_g).view(-1, 1)
        sub_g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        return sub_g

    def get_nhop_neighbor(self, time, nodes, n=2):
        g = self.dgl_graph_dict[time]  # time时刻的图
        total_nodes = set(nodes)
        for i in range(n):
            step_nodes = total_nodes.copy()
            for node in step_nodes:
                neighbor_n, _ = g.in_edges(node)
                neighbor_n = set(neighbor_n.tolist())
                total_nodes |= neighbor_n
        return total_nodes

    def r2e(self, triplets):
        if triplets.size != 0:
            src, rel, dst = triplets.transpose()
        else:
            src, rel, dst = np.array([]), np.array([]), np.array([])
        # get all relations
        uniq_r = np.unique(rel)
        uniq_r = np.concatenate((uniq_r, uniq_r + self.n_rel))
        # generate r2e
        r_to_e = defaultdict(set)
        for j, (src, rel, dst) in enumerate(triplets):
            r_to_e[rel].add(src)
            r_to_e[rel].add(dst)
            r_to_e[rel + self.n_rel].add(src)
            r_to_e[rel + self.n_rel].add(dst)
        r_len = []
        e_idx = []
        idx = 0
        for r in uniq_r:
            r_len.append((idx, idx + len(r_to_e[r])))
            e_idx.extend(list(r_to_e[r]))
            idx += len(r_to_e[r])
        return uniq_r, r_len, e_idx


class QuadruplesDataset(Dataset):
    def __init__(self, quadruples, history_len, dglGraphs, timeInvDict, history_mode='sub_rel', nhop=2):
        self.quadruples = quadruples  # 四元组数组，np.array, [quad_num, 4] (sub, rel, obj, time)
        self.history_len = history_len  # 预测答案依据的历史序列长度
        self.dglGraphs = dglGraphs  # DGLGraphDataset类
        self.timeInvDict = timeInvDict  # 时间倒排表，key是实体/（实体，关系），value是从小到大的时间list
        self.nhop = nhop  # 用于取子图的时的nhop
        self.history_mode = history_mode

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        quad = self.quadruples[idx]
        sub, rel, obj, time = quad[0], quad[1], quad[2], quad[3]

        if self.history_mode == 'sub_rel':
            times = self.timeInvDict[(sub, rel)]  # 出现过的时间
            history_times = times[:times.index(time)]
            history_times = history_times[max(-self.history_len, -len(history_times)):]  # 按照历史时间长度取时间
        elif self.history_mode == 'both':
            times1 = self.timeInvDict[(sub, rel)]
            times2 = self.timeInvDict[sub]
            history_times1 = times1[:times1.index(time)]
            history_times1 = history_times1[max(-(self.history_len//2), -len(history_times1)):]
            history_times2 = times2[:times2.index(time)]
            history_times2 = history_times2[max(-(self.history_len // 2), -len(history_times2)):]
            history_times = sorted(list(set(history_times1 + history_times2)))
        else:
            times = self.timeInvDict[sub]
            history_times = times[:times.index(time)]
            history_times = history_times[max(-self.history_len, -len(history_times)):]  # 按照历史时间长度取时间
        if len(history_times) < self.history_len:
            # 如果小于则补-1
            history_times = [-1] * (self.history_len - len(history_times)) + history_times

        history_graphs = []
        node_ids = []
        for t in history_times:
            sub_graph = self.dglGraphs.get_nhop_subgraph(t, [sub], self.nhop)
            sub_graph.edata['query_rel'] = torch.ones_like(sub_graph.edata['type']) * rel
            sub_graph.edata['query_ent'] = torch.ones_like(sub_graph.edata['type']) * sub
            sub_graph.edata['query_time'] = torch.ones_like(sub_graph.edata['type']) * time
            history_graphs.append(sub_graph)
            node_ids.append(sub_graph.ndata['id'].squeeze(1).tolist().index(sub))

        return torch.tensor(sub), torch.tensor(rel), torch.tensor(obj), torch.tensor(time), \
              history_graphs, torch.tensor(history_times), torch.tensor(node_ids)

    @staticmethod
    def collate_fn(data):
        sub = torch.stack([_[0] for _ in data], dim=0)
        rel = torch.stack([_[1] for _ in data], dim=0)
        obj = torch.stack([_[2] for _ in data], dim=0)
        time = torch.stack([_[3] for _ in data], dim=0)
        history_graphs = []
        for item in data:
            gs = item[4]
            history_graphs = history_graphs +  gs
        history_graphs = dgl.batch(history_graphs)
        history_times = torch.stack([_[5] for _ in data], dim=0)
        batch_node_ids = torch.cat([_[6] for _ in data], dim=0)
        batchgraph_nodes_num = history_graphs.batch_num_nodes()
        graph_num = batchgraph_nodes_num.size(0)
        offset_node_ids = batchgraph_nodes_num.unsqueeze(0).repeat(graph_num, 1)
        offset_mask = torch.tril(torch.ones(graph_num, graph_num), diagonal=-1).long()
        offset_node_ids = offset_node_ids * offset_mask
        offset_node_ids = torch.sum(offset_node_ids, dim=1)
        batch_node_ids += offset_node_ids
        return sub, rel, obj, time, history_graphs, history_times, batch_node_ids

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

    dglGraphDataset = DGLGraphDataset(baseDataset.train_snapshots + baseDataset.valid_snapshots + baseDataset.test_snapshots,
                                      baseDataset.num_e, baseDataset.num_r)

    trainQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.trainQuadruples, baseDataset.num_r)
    trainQuadDataset = QuadruplesDataset(trainQuadruples, 3, dglGraphDataset, baseDataset.time_inverted_index_dict, 2)
    trainDataLoader = DataLoader(
        trainQuadDataset,
        shuffle=True,
        batch_size=2,
        collate_fn=trainQuadDataset.collate_fn
    )

    from torch import nn
    from models.GraphEncoder import *

    ent_dim = 100
    ent_embeds = nn.Embedding(baseDataset.num_e, ent_dim)
    graphEncoder = RGCNEncoder(ent_dim, baseDataset.num_r * 2, ent_dim // 4, 0.0)

    for sub, rel, obj, time, history_graphs, history_times, batch_node_ids in trainDataLoader:
        history_graphs.ndata['h'] = ent_embeds(history_graphs.ndata['id']).view(-1, ent_dim)
        print(ent_embeds(sub))
        print( history_graphs.ndata['h'][batch_node_ids])
        break

    testQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.testQuadruples, baseDataset.num_r)
    testQuadDataset = QuadruplesDataset(testQuadruples, 3, dglGraphDataset,
                                        baseDataset.time_inverted_index_dict, 2)
    testDataLoader = DataLoader(
        testQuadDataset,
        batch_size=2,
        collate_fn=testQuadDataset.collate_fn,
    )

    for sub, rel, obj, time, history_graphs, history_times, batch_node_ids in testDataLoader:
        history_graphs.ndata['h'] = ent_embeds(history_graphs.ndata['id']).view(-1, ent_dim)
        print(ent_embeds(sub))
        print(history_graphs.ndata['h'][batch_node_ids])
        break

