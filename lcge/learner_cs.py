import argparse
from typing import Dict
import torch
from torch import optim
from datasets_lcge import TemporalDataset
from optimizers_cs import TKBCOptimizer, IKBCOptimizer
from models_cs import LCGE
from regularizers import N3, Lambda3


# 适用于 Wikidata12k
# 设置了一系列的参数，可以在运行脚本时通过命令行来指定
parser = argparse.ArgumentParser(description="Commonsense-Guided Temporal KGE")
parser.add_argument("--dataset", type=str, help="Dataset name")
models = ["LCGE"]
parser.add_argument("--model", choices=models, help="Model in {}".format(models))
parser.add_argument("--max_epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument(
    "--valid_freq", default=5, type=int, help="Number of epochs between each valid."
)
parser.add_argument("--rank", default=100, type=int, help="Factorization rank.")
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate")
parser.add_argument(
    "--emb_reg", default=0.0, type=float, help="Embedding regularizer strength"
)
parser.add_argument(
    "--time_reg", default=0.0, type=float, help="Timestamp regularizer strength"
)
parser.add_argument(
    "--no_time_emb",
    default=False,
    action="store_true",
    help="Use a specific embedding for non temporal relations",
)
parser.add_argument(
    "--rule_reg", default=0.0, type=float, help="Rule regularizer strength"
)
parser.add_argument(
    "--weight_static", default=0.0, type=float, help="Weight of static score"
)

args = parser.parse_args()

# 初始化和设置训练模型及其参数
dataset = TemporalDataset(args.dataset)
sizes = dataset.get_shape()

print("sizes of dataset is:\t", sizes)

model = {
    "LCGE": LCGE(sizes, args.rank, args.weight_static, no_time_emb=args.no_time_emb),
}[args.model]
model = model.cuda()

# 设置优化器
opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

# 设置正则化器
emb_reg = N3(args.emb_reg)
time_reg = Lambda3(args.time_reg)

# 初始化变量
best_mrr = 0.0
best_hit = 0.0
early_stopping = 0

# 开始训练模型
for epoch in range(args.max_epochs):
    # 加载训练数据
    examples = torch.from_numpy(dataset.get_train().astype("int64"))
    # print("\nexamples:\n", examples.size())

    # 设置模型为训练模式
    model.train()

    # 这两行分别初始化了不同的优化器，传入模型、正则化器、PyTorch的优化器对象、数据集和批次大小，并开始训练
    # 如果数据集有时间间隔，则使用IKBCOptimizer
    # 否则使用TKBCOptimizer
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset, batch_size=args.batch_size
        )
        optimizer.epoch(examples)
    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt, batch_size=args.batch_size
        )
        optimizer.epoch(examples)

    # 定义一个函数，用于计算MRR和hits@n
    # 这个函数的作用是将左右两边的MRR和hits@n取平均
    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs["lhs"] + mrrs["rhs"]) / 2.0
        h = (hits["lhs"] + hits["rhs"]) / 2.0
        return {"MRR": m, "hits@[1,3,10]": h}

    # 每隔一定的epoch，计算一次验证集、测试集和训练集的MRR和hits@n
    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        # 评估模型
        # 如果数据集有时间间隔，则使用eval函数
        # 否则使用avg_both函数
        if dataset.has_intervals():
            valid, test, train = [
                dataset.eval(model, split, -1 if split != "train" else 50000)
                for split in ["valid", "test", "train"]
            ]
            print("epoch: ", epoch + 1)
            print("valid: ", valid)
            print("test: ", test)
            print("train: ", train)

            print("test hits@n:\t", test["hits@_all"])

            if test["MRR_all"] > best_mrr:
                best_mrr = test["MRR_all"]
                best_hit = test["hits@_all"]
                early_stopping = 0
            else:
                early_stopping += 1

            if early_stopping > 10:
                print("early stopping!")
                break
        else:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != "train" else 50000))
                for split in ["valid", "test", "train"]
            ]

            print("epoch: ", epoch + 1)
            print("valid: ", valid["MRR"])
            print("test: ", test["MRR"])
            print("train: ", train["MRR"])

            print("test hits@n:\t", test["hits@[1,3,10]"])

            # 如果测试集的MRR大于最好的MRR，则更新最好的MRR和hits@n
            if test["MRR"] > best_mrr:
                best_mrr = test["MRR"]
                best_hit = test["hits@[1,3,10]"]
                early_stopping = 0
            else:
                early_stopping += 1

            # 如果连续多个周期没有性能提升，则停止训练
            if early_stopping > 10:
                print("early stopping!")
                break

print("The best test mrr is:\t", best_mrr)
print("The best test hits@1,3,10 are:\t", best_hit)
