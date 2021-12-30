import argparse
import torch
import os
from tqdm import tqdm
from dataset import *
from model import TemporalTransformerHawkesGraphModel
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
    parser.add_argument('--batch_size', type=int, default=256, help='输入batch_size, 0表示输入一整个snapshot, N表示从这个snapshot中采样N个')

    parser.add_argument('--num_works', type=int, default=8, help='dataloader使用的cpu works数量')

    parser.add_argument('--grad_norm', type=float, default=1.0, help='梯度裁剪norm')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='优化器的weight decay参数')

    parser.add_argument('--d_model', default=100, type=int, help='实体嵌入维度')
    parser.add_argument('--data', default='ICEWS14', type=str, help='使用的数据集名称')
    parser.add_argument('--max_epochs', default=400, type=int, help='最大训练epoch数量')
    parser.add_argument('--lr', default=0.003, type=float, help='学习率')
    parser.add_argument('--do_train', action='store_true', help='执行训练过程')
    parser.add_argument('--do_test', action='store_true', help='执行测试过程')
    parser.add_argument('--valid_epoch', default=1, type=int, help='训练过程中验证的频次，每N个epoch进行验证')
    parser.add_argument('--history_len', default=10, type=int, help='使用的历史信息长度')
    parser.add_argument('--dropout', default=0.2, type=float)

    parser.add_argument('--seqTransformerLayerNum', default=1, type=int, help='序列编码中Transformer的层数')
    parser.add_argument('--seqTransformerHeadNum', default=1, type=int, help='序列编码中Transformer注意力头个数')

    parser.add_argument('--load_model_path', default='output', type=str, help='加载模型')

    parser.add_argument('--history_mode', default='delta_t_windows', type=str)
    parser.add_argument('--nhop', default=1, type=int)
    parser.add_argument('--forecasting_t_win_size', default=1, type=int)

    return parser.parse_args(args)

def test(model, testloader, skip_dict, device):
    """测试函数
    model: 输入模型
    testloader: 测试集数据DataLoader
    skip_dict: 用于Time Aware Filter, 在BaseDataset中获取
    device: 使用的设备，cuda or cpu
    """
    model.eval()
    ranks = []
    logs = []
    TimeMSE = 0.
    with torch.no_grad():
        for sub, rel, obj, time, history_graphs, history_times, batch_node_ids in tqdm(testloader):
            sub = sub.to(device)
            rel = rel.to(device)
            obj = obj.to(device)
            time = time.to(device)
            history_graphs = history_graphs.to(device)
            history_times = history_times.to(device)
            batch_node_ids = batch_node_ids.to(device)

            intes, type = model(sub, rel, history_graphs, history_times, batch_node_ids)
            score = model.predict_e(intes, type)

            # 时间预测
            last_time, _ = torch.max(history_times, 1)
            dur_last = time - last_time
            dur_last = torch.where(last_time > 0, dur_last, torch.zeros_like(dur_last))
            estimate_dt, dur_last = model.predict_t(obj, dur_last)
            tp_loss = model.time_prediction_loss(estimate_dt, dur_last)

            TimeMSE = TimeMSE + tp_loss * sub.size(0)

            _, rank_idx = score.sort(dim=1, descending=True)
            rank = torch.nonzero(rank_idx == obj.view(-1, 1))[:, 1].view(-1)
            ranks.append(rank)

            for i in range(score.shape[0]):
                src_i = sub[i].item()
                rel_i = rel[i].item()
                dst_i = obj[i].item()
                time_i = time[i].item()

                predict_score = score[i].tolist()
                answer_prob = predict_score[dst_i]
                for e in skip_dict[(src_i, rel_i, time_i)]:
                    if e != dst_i:
                        predict_score[e] = -1e6
                predict_score.sort(reverse=True)
                filter_rank = predict_score.index(answer_prob) + 1

                logs.append({
                    'Time-aware Filter MRR': 1.0 / filter_rank,
                    'Time-aware Filter HITS@1': 1.0 if filter_rank <= 1 else 0.0,
                    'Time-aware Filter HITS@3': 1.0 if filter_rank <= 3 else 0.0,
                    'Time-aware Filter HITS@10': 1.0 if filter_rank <= 10 else 0.0,
                })

    metrics = {}
    ranks = torch.cat(ranks)
    ranks += 1
    mrr = torch.mean(1.0 / ranks.float())
    metrics['Raw MRR'] = mrr
    for hit in [1, 3, 10]:
        avg_count = torch.mean((ranks <= hit).float())
        metrics['Raw Hit@{}'.format(hit)] = avg_count

    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    metrics['Time MSE'] = TimeMSE / len(testloader.dataset)
    return metrics


def train_epoch(args, model, traindataloader, optimizer, device, epoch):
    model.train()
    with tqdm(total=len(traindataloader), unit='ex') as bar:
        bar.set_description('Train')
        total_loss = 0
        total_num = 0
        for sub, rel, obj, time, history_graphs, history_times, batch_node_ids in traindataloader:
            sub = sub.to(device)
            rel = rel.to(device)
            obj = obj.to(device)
            time = time.to(device)
            history_graphs = history_graphs.to(device)
            history_times = history_times.to(device)
            batch_node_ids = batch_node_ids.to(device)

            score, type = model(sub, rel, history_graphs, history_times, batch_node_ids)
            lp_loss = model.link_prediction_loss(score, type, obj)

            # 时间预测
            last_time, _ = torch.max(history_times, 1)
            dur_last = time - last_time
            dur_last = torch.where(last_time > 0, dur_last, torch.zeros_like(dur_last))

            estimate_dt, dur_last = model.predict_t(obj, dur_last)
            tp_loss = model.time_prediction_loss(estimate_dt, dur_last)

            loss = lp_loss + 0.1 * tp_loss

            loss.backward()

            total_loss += loss
            total_num += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

            bar.update(1)
            bar.set_postfix(loss='%.4f' % loss, lp_loss='%.4f' % lp_loss, tp_loss='%.4f' % tp_loss)

        logging.info('Epoch {} Train Loss: {}'.format(epoch, total_loss/total_num))

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

    trainQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.trainQuadruples, baseDataset.num_r)
    trainQuadDataset = QuadruplesDataset(trainQuadruples, args.history_len, dglGraphDataset, baseDataset,
                                         args.history_mode, args.nhop, args.forecasting_t_win_size, time_span)
    trainDataLoader = DataLoader(
        trainQuadDataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=lambda x: trainQuadDataset.collate_fn(x, baseDataset.num_e),
        num_workers=args.num_works,
    )

    testQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.testQuadruples, baseDataset.num_r)
    testQuadDataset = QuadruplesDataset(testQuadruples, args.history_len, dglGraphDataset,
                                         baseDataset, args.history_mode, args.nhop, args.forecasting_t_win_size, time_span)
    testDataLoader = DataLoader(
        testQuadDataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=lambda x: testQuadDataset.collate_fn(x, baseDataset.num_e),
        num_workers=args.num_works,
    )

    # 模型创建
    Config = namedtuple('config', ['n_ent', 'd_model', 'n_rel', 'dropout','seqTransformerLayerNum', 'seqTransformerHeadNum'])
    config = Config(n_ent=baseDataset.num_e + 1,  # 最后一个为PAD Entity
                    n_rel=baseDataset.num_r * 2,
                    d_model=args.d_model,
                    dropout=args.dropout,
                    seqTransformerLayerNum=args.seqTransformerLayerNum,
                    seqTransformerHeadNum=args.seqTransformerHeadNum)
    model = TemporalTransformerHawkesGraphModel(config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    if args.do_train:
        logging.info('Start Training......')

        for i in range(args.max_epochs):
            if i % args.valid_epoch == 0 and i != 0:
                metrics = test(model, testDataLoader, baseDataset.skip_dict, device)
                model_save_path = os.path.join(output_path, 'model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_save_path)

                for mode in metrics.keys():
                    logging.info('Test {} : {}'.format(mode, metrics[mode]))

            train_epoch(args, model, trainDataLoader, optimizer, device, i)

    if args.do_test:
        logging.info('Start Testing......')
        metrics = test(model, testDataLoader, baseDataset.skip_dict, device)
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))

if __name__ == '__main__':
    args = parse_args()
    main(args)

