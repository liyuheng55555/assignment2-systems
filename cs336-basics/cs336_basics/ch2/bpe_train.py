import json
import logging
import os
import pickle
import time
from collections import deque
from multiprocessing import Process, Queue
from time import perf_counter

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries
from ch2.BucketMaxSD import BucketMaxSD
from ch2.Vocab import Vocab
from ch2.type_define import Index, Connection, Num, TokenList, GB, GPT2_PAT
from ch2.utils import (
    bytes_to_bytes_list,
    merge_by_one_rule,
    bytes_list_to_connections,
)
from tests.conftest import vocab_size


def find_best_connection(
    connections_num_map: BucketMaxSD,
    connections_contrib_map: dict[Connection, set[Index]],
) -> tuple[Connection, set[Index]]:
    max_connection, _ = connections_num_map.max_item()
    contributors_index = connections_contrib_map[max_connection]
    return max_connection, contributors_index


def remove_old_connection(
    old_token_list: TokenList,
    nums: Num,
    i: int,
    connections_num_map: BucketMaxSD,
    connections_contrib_map: dict[Connection, set[Index]],
):
    old_connections = bytes_list_to_connections(old_token_list)
    for connection in old_connections:
        if connection in connections_num_map:
            connections_num_map.decr(connection, nums)
            if (
                connection in connections_contrib_map
                and i in connections_contrib_map[connection]
            ):
                connections_contrib_map[connection].remove(i)
                if len(connections_contrib_map[connection]) == 0:
                    connections_contrib_map.pop(connection)


def update_by_new_connection(
    new_bytes_list: TokenList,
    nums: Num,
    i: int,
    connections_num_map: BucketMaxSD,
    connections_contrib_map: dict[Connection, set[Index]],
    all_bytes: list[tuple[TokenList, int]],
):
    new_connections = bytes_list_to_connections(new_bytes_list)
    for connection in new_connections:
        connections_num_map.incr(connection, nums)
        if connection not in connections_contrib_map:
            connections_contrib_map[connection] = set()
        connections_contrib_map[connection].add(i)
    all_bytes[i] = (new_bytes_list, nums)


def update(
    all_bytes: list[tuple[TokenList, int]],
    last_contributors_index: list[Index],
    connections_num_map: BucketMaxSD,
    connections_contrib_map: dict[Connection, set[Index]],
    merge_rule: Connection,
):
    if not last_contributors_index:
        last_contributors_index = list(range(len(all_bytes)))
    for i in last_contributors_index:
        token_list: TokenList = all_bytes[i][0]
        nums: int = all_bytes[i][1]
        # 根据最新的一条合并规则进行合并
        new_bytes_list: TokenList
        if merge_rule is not None:
            new_bytes_list = merge_by_one_rule(token_list, merge_rule)
        else:
            new_bytes_list = token_list
        if new_bytes_list is None:
            new_bytes_list = token_list
        # 旧的connection从统计中清除
        if merge_rule is not None:
            remove_old_connection(
                token_list,
                nums,
                i,
                connections_num_map,
                connections_contrib_map,
            )
        # 新的connection加入统计
        update_by_new_connection(
            new_bytes_list,
            nums,
            i,
            connections_num_map,
            connections_contrib_map,
            all_bytes,
        )


def pre_tokenize_worker(
    queue: Queue,
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    logging.info(f"start {start} to {end}")

    pattern = "|".join(re.escape(token) for token in special_tokens)
    contents = [c for c in re.split(pattern, chunk) if c]

    print(
        f"特殊符号切分完成, 共{len(contents)}段",
    )

    all_words: dict[bytes, int] = {}  # "xxx" -> nums
    for content in contents:
        for match in re.finditer(GPT2_PAT, content):
            word = match.group()
            byte = word.encode("utf-8")
            all_words[byte] = all_words.get(byte, 0) + 1

    print(f"正则解析完成")

    queue.put(all_words)


def pre_tokenize(
    input_path: str, special_tokens: list[str], concurrency: int = 8
) -> list[tuple[TokenList, Num]]:
    queue: Queue = Queue()
    workers: deque[Process] = deque()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            concurrency,
            list(token.encode("utf-8") for token in special_tokens),
            max_memory_in_bytes=3 * GB,
        )
        logging.info(f"文件切分为{len(boundaries)-1}块")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            process = Process(
                target=pre_tokenize_worker,
                args=(queue, input_path, start, end, special_tokens),
            )
            workers.append(process)

    logging.info("子进程准备完毕")

    finished_worker_count = 0
    worker_count = len(workers)

    for _ in range(min(concurrency, worker_count)):
        workers.pop().start()
        time.sleep(0.5)

    logging.info(f"首批{concurrency}个进程启动")

    merged_result: dict[bytes, Num] = {}

    while finished_worker_count < worker_count:
        # 每结束一个进程，就开一个新进程
        d: dict[bytes, int] = queue.get()
        finished_worker_count += 1
        logging.info(f"第 {finished_worker_count} 个任务完成")
        if len(workers) != 0:
            workers.pop().start()
        for k, v in d.items():
            merged_result[k] = merged_result.get(k, 0) + v

    logging.info("结果汇总完毕")

    return list((bytes_to_bytes_list(k), v) for k, v in merged_result.items())


def _save_cache(
    cache_filename: str,
    special_tokens: list[str],
    all_bytes: list[tuple[TokenList, Num]],
):
    """保存预分词结果到缓存文件"""
    try:
        with open(cache_filename, "wb") as f:
            pickle.dump(
                {"special_tokens": special_tokens, "all_bytes": all_bytes}, f
            )
        logging.info(f"预分词结果已缓存到: {cache_filename}")
    except Exception as save_error:
        logging.warning(f"保存缓存失败: {save_error}")


def _do_pretokenize_and_cache(
    input_path: str, special_tokens: list[str], cache_filename: str
) -> list[tuple[TokenList, Num]]:
    """执行预分词并缓存结果"""
    all_bytes = pre_tokenize(input_path, special_tokens)

    _save_cache(cache_filename, special_tokens, all_bytes)
    return all_bytes


def bpe_train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Connection]]:
    vocab = Vocab()
    merge_rules: list[Connection] = []

    for i in range(0, 256):
        vocab.put(bytes([i]))
    for token in special_tokens:
        vocab.put(token.encode("utf-8"))

    pre_tokenize_start = perf_counter()

    # 生成缓存文件名
    cache_filename = f"{input_path}.pretokenize_cache.pkl"

    # 尝试加载缓存
    if os.path.exists(cache_filename):
        logging.info(f"发现预分词缓存文件: {cache_filename}")
        try:
            with open(cache_filename, "rb") as f:
                cached_data = pickle.load(f)
                if cached_data["special_tokens"] == special_tokens:
                    logging.info("缓存文件有效，直接使用")
                    all_bytes = cached_data["all_bytes"]

                else:
                    logging.info("特殊token不匹配，重新预分词")
                    raise ValueError("Token mismatch")
        except Exception as e:
            logging.warning(f"读取缓存失败: {e}，重新预分词")
            all_bytes = _do_pretokenize_and_cache(
                input_path, special_tokens, cache_filename
            )
    else:
        logging.info("未发现缓存文件，开始预分词")
        all_bytes = _do_pretokenize_and_cache(
            input_path, special_tokens, cache_filename
        )
    pre_tokenize_end = perf_counter()

    idx = 0
    last_contributors_index: list[Index] = []
    connections_num_map = BucketMaxSD()
    connections_contrib_map: dict[Connection, set[Index]] = {}
    while len(vocab) < vocab_size:
        idx += 1
        # calculate connections
        update(
            all_bytes,
            last_contributors_index,
            connections_num_map,
            connections_contrib_map,
            merge_rules[-1] if len(merge_rules) > 0 else None,
        )

        # find best connection
        best_connection, contributors = find_best_connection(
            connections_num_map, connections_contrib_map
        )
        last_contributors_index = list(contributors)
        merge_rules.append(best_connection)

        # update vocab
        vocab.put(best_connection[0] + best_connection[1])

        if idx % 16 == 0:
            logging.info(f"idx {idx}")
            connections_num_map.message()

    calculate_end = perf_counter()

    print(f"预分词耗时: {pre_tokenize_end - pre_tokenize_start:.6f} 秒")
    print(f"计算耗时：{calculate_end - pre_tokenize_end:.6f} 秒")

    return vocab.build_dict(), merge_rules


def load_bpe_json(json_filepath: str) -> tuple[dict[int, bytes], list[Connection], list[str] | None]:
    """
    从由本文件写出的合并 JSON（同名 .json）加载 vocab 与 merges。

    JSON 结构示例：
    {
      "input_path": ".../TinyStoriesV2-GPT4-train.txt",
      "vocab_size": 10000,
      "special_tokens": ["<|endoftext|>"],
      "vocab": {"0": [116, 101, ...], ...},
      "merges": [ [[110,...],[101,...]], ... ]
    }

    返回值：
    - vocab: dict[int, bytes]
    - merges: list[tuple[bytes, bytes]]
    - special_tokens: list[str] | None
    """
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_vocab = data.get("vocab", {})
    vocab: dict[int, bytes] = {int(k): bytes(v) for k, v in raw_vocab.items()}

    raw_merges = data.get("merges", [])
    merges: list[Connection] = [(bytes(a), bytes(b)) for a, b in raw_merges]

    special_tokens = data.get("special_tokens")
    return vocab, merges, special_tokens


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_path = "/data/cs336/data/owt_train.txt"
    vocab_size = 32000

    main_start = perf_counter()
    # 测试缓存功能 - 使用小数据集
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/bpe_example.txt", 270, ["<|endoftext|>"])
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 512, ["<|endoftext|>"])
    special_tokens = ["<|endoftext|>"]
    vocab, merges = bpe_train(
        input_path,
        vocab_size,
        special_tokens,
    )
    # 序列化 vocab 与 merges 到与 input_path 同名的 .json 文件
    try:
        out_json_path = os.path.splitext(input_path)[0] + ".json"

        # 使用整数数组表示 bytes，避免 base64，保持 JSON 可读、可无损还原
        serialized = {
            "input_path": input_path,
            "vocab_size": vocab_size,
            "special_tokens": special_tokens,
            "vocab": {str(k): list(v) for k, v in vocab.items()},  # bytes -> [int]
            "merges": [[list(a), list(b)] for (a, b) in merges],    # (bytes, bytes) -> [[int],[int]]
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False)
        logging.info(f"已保存 BPE 结果到: {out_json_path}")
    except Exception as e:
        logging.error(f"保存 BPE 结果到 JSON 失败: {e}")
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/owt_valid.txt", 32000, ["<|endoftext|>"])
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/owt_train.txt", 32000, ["<|endoftext|>"])
    # print(merge)
    main_end = perf_counter()
    print(f"总耗时: {main_end - main_start:.6f} 秒")



    # 2025-08-24 21:52:04,794 - INFO - 预分词结果已缓存
