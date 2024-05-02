# Copyright (c) Facebook, Inc. and its affiliates.

import pkg_resources
import os
import errno
import math
from pathlib import Path
import pickle
import sys

import numpy as np


DATA_PATH = pkg_resources.resource_filename("lcge", "data/")


# 处理时间，提取年份转换为整数，并将####转换为-inf和inf
def get_be(begin, end):
    begin = begin.strip().split("-")[0]
    end = end.strip().split("-")[0]
    if begin == "####":
        begin = (-math.inf, 0, 0)
    else:
        begin = (int(begin), 0, 0)
    if end == "####":
        end = (math.inf, 0, 0)
    else:
        end = (int(end), 0, 0)

    return begin, end


def prepare_dataset_rels(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(type)\t(timestamp)\n
    Maps each entity, relation+type and timestamp to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    # 从 src_data 中读取实体、关系、时间戳，并将其映射到集合中
    files = ["train", "valid", "test"]
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, "r")
        for line in to_read.readlines():
            v = line.strip().split("\t")
            lhs, rel, rhs, begin, end = v

            begin, end = get_be(begin, end)

            timestamps.add(begin)
            timestamps.add(end)
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)

        to_read.close()

    # 将实体、关系、时间戳在集合中排序，并映射到id，key为实体、关系、时间戳，value为id
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}

    # 对时间戳进行排序，并去掉-inf和inf
    all_ts = sorted(timestamps)[1:-1]
    timestamps_to_id = {x: i for (i, x) in enumerate(all_ts)}

    print(
        "{} entities, {} relations over {} timestamps".format(
            len(entities), len(relations), len(timestamps)
        )
    )
    n_relations = len(relations)
    n_entities = len(entities)

    # 创建 lcge 包中的文件夹
    try:
        os.makedirs(os.path.join(DATA_PATH, name))
    except OSError as e:
        r = input(f"{e}\nContinue ? [y/n]")
        if r != "y":
            sys.exit()

    # 把实体、关系、时间戳映射到id的字典写入文件，文件名分别为ent_id, rel_id, ts_id
    for dic, f in zip(
        [entities_to_id, relations_to_id, timestamps_to_id],
        ["ent_id", "rel_id", "ts_id"],
    ):
        ff = open(os.path.join(DATA_PATH, name, f), "wb")
        pickle.dump(dic, ff)
        ff.close()

    # 计算时间戳之间的差异，并将这些差异数据存储到文件中
    ts_to_int = [x[0] for x in all_ts]  # 只取出年份
    ts = np.array(ts_to_int, dtype="float")
    # 提取时间戳之间的差值
    diffs = ts[1:] - ts[:-1]
    out = open(os.path.join(DATA_PATH, name, "ts_diffs.pickle"), "wb")
    pickle.dump(diffs, out)
    out.close()

    # 用 id 替换实体、关系、时间戳，并序列化到 lcge 包中
    event_list = {
        "all": [],
    }
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, "r")
        examples = []
        ignore = 0
        total = 0
        full_intervals = 0
        half_intervals = 0
        point = 0
        for line in to_read.readlines():
            v = line.strip().split("\t")
            lhs, rel, rhs, begin, end = v
            begin_t, end_t = get_be(begin, end)
            total += 1

            begin = begin_t
            end = end_t

            if begin_t[0] == -math.inf:  # 处理 -inf，将其映射到第一个时间戳
                begin = all_ts[0]
                if not end_t[0] == math.inf:  # 半时间区+1
                    half_intervals += 1
            if end_t[0] == math.inf:  # 处理 inf，将其映射到最后一个时间戳
                end = all_ts[-1]
                if not begin_t[0] == -math.inf:  # 半时间区+1
                    half_intervals += 1

            if begin_t[0] > -math.inf and end_t[0] < math.inf:  # 处理完整时间戳
                if begin_t[0] == end_t[0]:
                    point += 1
                else:
                    full_intervals += 1  # 完整时间区+1

            # 将时间戳映射到id
            begin = timestamps_to_id[begin]
            end = timestamps_to_id[end]

            if begin > end:
                ignore += 1
                continue

            lhs = entities_to_id[lhs]
            rel = relations_to_id[rel]
            rhs = entities_to_id[rhs]

            # 添加事件到 event_list 中，event_list 是总的事件列表
            event_list["all"].append((begin, -1, (lhs, rel, rhs)))
            event_list["all"].append((end, +1, (lhs, rel, rhs)))

            # 添加到 examples 中
            try:
                examples.append([lhs, rel, rhs, begin, end])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + ".pickle"), "wb")
        pickle.dump(np.array(examples).astype("uint64"), out)
        out.close()
        print(f"Ignored {ignore} events.")
        print(
            f"Total : {total} // Full : {full_intervals} // Half : {half_intervals} // Point : {point}"
        )

    # 将 event_list 中的事件序列化到文件中，包含所有事件、开始事件、结束事件
    for k, v in event_list.items():
        out = open(Path(DATA_PATH) / name / ("event_list_" + k + ".pickle"), "wb")
        print("Dumping all events", len(v))
        pickle.dump(sorted(v), out)
        out.close()


if __name__ == "__main__":
    datasets = ["wikidata12k"]
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset_rels(
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
