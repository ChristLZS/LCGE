# Copyright (c) Facebook, Inc. and its affiliates.

import pkg_resources
import os
import errno
from pathlib import Path
import pickle

import numpy as np

from collections import defaultdict

DATA_PATH = pkg_resources.resource_filename("lcge", "data/")


# 处理数据集，将实体、关系、时间戳映射到id，并创建相应的文件夹
def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(timestamp)\n
    Maps each entity and relation to a unique id, create corresponding folder name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and rel_id / ent_id for analysis.
    """

    # 从 src_data 中读取实体、关系、时间戳，并将其映射到集合中
    files = ["train", "valid", "test"]
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, "r", encoding="utf-8")
        for line in to_read.readlines():
            lhs, rel, rhs, timestamp = line.strip().split("\t")
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(timestamp)
        to_read.close()

    # 将实体、关系、时间戳在集合中排序，并映射到id
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print(
        "{} entities, {} relations over {} timestamps".format(
            len(entities), len(relations), len(timestamps)
        )
    )
    n_relations = len(relations)
    n_entities = len(entities)

    # 创建 lcge 包中的文件夹
    os.makedirs(os.path.join(DATA_PATH, name))
    # 把实体、关系、时间戳映射到id的字典写入文件，文件名分别为ent_id, rel_id, ts_id
    for dic, f in zip(
        [entities_to_id, relations_to_id, timestamps_to_id],
        ["ent_id", "rel_id", "ts_id"],
    ):
        ff = open(os.path.join(DATA_PATH, name, f), "w+", encoding="utf-8")
        for x, i in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # 用 id 替换实体、关系、时间戳，并序列化到 lcge 包中
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, "r", encoding="utf-8")
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs, ts = line.strip().split("\t")
            try:
                examples.append(
                    [
                        entities_to_id[lhs],
                        relations_to_id[rel],
                        entities_to_id[rhs],
                        timestamps_to_id[ts],
                    ]
                )
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + ".pickle"), "wb")
        pickle.dump(np.array(examples).astype("uint64"), out)
        out.close()

    print("creating filtering lists")

    # 创建过滤列表，把对称的关系对应的实体对过滤掉，直接跳过
    to_skip = {"lhs": defaultdict(set), "rhs": defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + ".pickle"), "rb"))
        for lhs, rel, rhs, ts in examples:
            to_skip["lhs"][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip["rhs"][(lhs, rel, ts)].add(rhs)

    to_skip_final = {"lhs": {}, "rhs": {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / "to_skip.pickle", "wb")
    pickle.dump(to_skip_final, out)
    out.close()

    # 计算实体的概率，probas 是概率
    examples = pickle.load(open(Path(DATA_PATH) / name / "train.pickle", "rb"))
    counters = {
        "lhs": np.zeros(n_entities),
        "rhs": np.zeros(n_entities),
        "both": np.zeros(n_entities),
    }

    for lhs, rel, rhs, _ts in examples:
        counters["lhs"][lhs] += 1
        counters["rhs"][rhs] += 1
        counters["both"][lhs] += 1
        counters["both"][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)  # 计算每个实体的概率
    out = open(Path(DATA_PATH) / name / "probas.pickle", "wb")
    pickle.dump(counters, out)
    out.close()


#  ICEW 数据集的处理
if __name__ == "__main__":
    datasets = ["ICEWS14", "ICEWS05-15"]
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "src_data", d
                ),
                d,
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise
