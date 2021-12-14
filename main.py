import argparse
import torch
import os
from tqdm import tqdm
from dataset import *
from model import TKGraphormer
import logging
from collections import namedtuple
from torch.utils.data import DataLoader
from utils import set_logger

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Reasoning Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_root', type=str, default='data', help='数据储存的根路径')
    parser.add_argument('--output_root', type=str, default='output', help='输出信息的根路径')
    parser.add_argument('--model_name', type=str, default='baseline', help='模型名称，用于输出信息储存路径')
    parser.add_argument('--batch_size', type=int, default=0, help='输入batch_size, 0表示输入一整个snapshot, N表示从这个snapshot中采样N个')

    parser.add_argument('--num_works', type=int, default=8, help='dataloader使用的cpu works数量')

    parser.add_argument('--grad_norm', type=float, default=1.0, help='梯度裁剪norm')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='优化器的weight decay参数')

    parser.add_argument('--ent_dim', default=100, type=int, help='实体嵌入维度')
    parser.add_argument('--rel_dim', default=100, type=int, help='关系嵌入维度')
    parser.add_argument('--lstm_hidden_dim', default=100, type=int, help='lstm隐藏层维度')
    parser.add_argument('--data', default='ICEWS14', type=str, help='使用的数据集名称')
    parser.add_argument('--max_epochs', default=400, type=int, help='最大训练epoch数量')
    parser.add_argument('--lr', default=0.003, type=float, help='学习率')
    parser.add_argument('--do_train', action='store_true', help='执行训练过程')
    parser.add_argument('--do_test', action='store_true', help='执行测试过程')
    parser.add_argument('--valid_epoch', default=3, type=int, help='训练过程中验证的频次，每N个epoch进行验证')
    parser.add_argument('--history_len', default=10, type=int, help='使用的历史信息长度')

    parser.add_argument('--graphEncoder', default='RGCNEncoder', type=str, help='图编码模块，[RGCNEncoder, RGTEncoder]')
    parser.add_argument('--sequenceEncoder', default='LSTMEncoder', type=str, help='序列编码模块，[LSTMEncoder, TransformerEncoder]')
    parser.add_argument('--decoder', default='MLPCLFDecoder', type=str, help='解码模块，[MLPCLFDecoder, DistMultDecoder]')

    return parser.parse_args(args)

def test(model, testloader, skip_dict, device):
    """测试函数
    model: 输入模型
    testloader: 测试集数据DataLoader
    skip_dict: 用于Time Aware Filter, 在BaseDataset中获取
    device: 使用的设备，cuda or cpu
    """
    model.eval()
    logs = []
    with torch.no_grad():
        for query_entities, query_relations, answers, query_timestamps, seq_history in testloader:
            query_entities = query_entities.to(device)
            query_relations = query_relations.to(device)
            answers = answers.to(device)
            query_timestamps = query_timestamps.to(device)
            graph_list = [g.to(device) for g in seq_history]

            score = model(graph_list, query_entities, query_relations, query_timestamps)
            for i in range(score.shape[0]):
                src = query_entities[i].item()
                rel = query_relations[i].item()
                dst = answers[i].item()
                time = query_timestamps[i].item()

                predict_score = score[i].tolist()
                answer_prob = predict_score[dst]
                for e in skip_dict[(src, rel, time)]:
                    if e != dst:
                        predict_score[e] = -1e6
                predict_score.sort(reverse=True)
                rank = predict_score.index(answer_prob) + 1

                logs.append({
                    'MRR': 1.0 / rank,
                    'HITS@1': 1.0 if rank <= 1 else 0.0,
                    'HITS@3': 1.0 if rank <= 3 else 0.0,
                    'HITS@10': 1.0 if rank <= 10 else 0.0,
                })
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics

def train_epoch(args, model, traindataloader, optimizer, device):
    model.train()
    with tqdm(total=len(traindataloader), unit='ex') as bar:
        bar.set_description('Train')
        for idx, (query_entities, query_relations, answers, query_timestamps, seq_history) in enumerate(traindataloader):
            query_entities = query_entities.to(device)
            query_relations = query_relations.to(device)
            answers = answers.to(device)
            query_timestamps = query_timestamps.to(device)
            graph_list = [g.to(device) for g in seq_history]

            score = model(graph_list, query_entities, query_relations, query_timestamps)
            loss = model.loss(score, answers)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

            bar.update(1)
            bar.set_postfix(loss='%.4f' % loss)

def main(args):
    # 设置路径，logger
    output_path = os.path.join(args.output_root, '{0}_{1}'.format(args.data, args.model_name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file = os.path.join(output_path, 'log.txt')
    set_logger(log_file)

    logging.info(args)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # 数据集处理
    data_path = os.path.join(args.data_root, args.data)
    trainpath = os.path.join(data_path, 'train.txt')
    validpath = os.path.join(data_path, 'valid.txt')
    testpath = os.path.join(data_path, 'test.txt')
    statpath = os.path.join(data_path, 'stat.txt')
    baseDataset = BaseDataset(trainpath, testpath, statpath, validpath)

    if 'ICEWS' in args.data:
        time_span = 24
    else:
        time_span = 1

    dglGraphDataset = DGLGraphDataset(
        baseDataset.train_snapshots + baseDataset.valid_snapshots + baseDataset.test_snapshots,
        baseDataset.num_e, baseDataset.num_r)

    trainTimes = list(range(len(baseDataset.train_snapshots)))
    trainQuadsInputByTimeDataset = QuadsInputByTimesDataset(trainTimes, dglGraphDataset, args.history_len, args.batch_size)
    trainDataLoader = DataLoader(
        trainQuadsInputByTimeDataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.num_works,
        collate_fn=QuadsInputByTimesDataset.collate_fn,
    )

    testTimes = list(range(len(baseDataset.train_snapshots + baseDataset.valid_snapshots),
                           len(baseDataset.train_snapshots + baseDataset.valid_snapshots + baseDataset.test_snapshots)))
    testQuadsInputByTimeDataset = QuadsInputByTimesDataset(testTimes, dglGraphDataset, args.history_len)

    testDataLoader = DataLoader(
        testQuadsInputByTimeDataset,
        batch_size=1,
        num_workers=args.num_works,
        collate_fn=QuadsInputByTimesDataset.collate_fn,
    )

    # 模型创建
    Config = namedtuple('config', ['n_ent', 'ent_dim', 'n_rel', 'rel_dim', 'lstm_hidden_dim',
                                   'graphEncoder', 'sequenceEncoder', 'decoder'])
    config = Config(n_ent=baseDataset.num_e,
                    ent_dim=args.ent_dim,
                    n_rel=baseDataset.num_r*2,
                    rel_dim=args.rel_dim,
                    lstm_hidden_dim=args.lstm_hidden_dim,
                    graphEncoder=args.graphEncoder,
                    sequenceEncoder=args.sequenceEncoder,
                    decoder=args.decoder)
    model = TKGraphormer(config)
    model.to(device)

    if args.do_train:
        logging.info('Start Training......')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for i in range(args.max_epochs):
            train_epoch(args, model, trainDataLoader, optimizer, device)
            if i % args.valid_epoch == 0 and i != 0:
                metrics = test(model, testDataLoader, baseDataset.skip_dict, device)
                for mode in metrics.keys():
                    logging.info('Test {} : {}'.format(mode, metrics[mode]))

    if args.do_test:
        logging.info('Start Testing......')
        metrics = test(model, testDataLoader, baseDataset.skip_dict, device)
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))

if __name__ == '__main__':
    args = parse_args()
    main(args)

