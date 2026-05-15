import json
import codecs
import logging
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from collections import deque
from multiprocessing import Process
from multiprocessing import Queue
from typing import Iterator, Iterable

from ch2.type_define import TokenList, Connection
from ch2.utils import (
    chunk_split,
    bytes_to_bytes_list,
    bytes_list_to_connections,
    merge_by_one_rule,
)


class Tokenizer:


    def __init__(
        self,
        vocab: dict[int, bytes],
        merge_rules: list[Connection],
        special_tokens=None,
    ):
        self.vocab: dict[int, bytes] = vocab
        self.reverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_rules: list[Connection] = merge_rules
        self.merge_rules_index: dict[Connection, int] = {
            t: i for i, t in enumerate(merge_rules)
        }
        self.MERGE_RULE_NOT_FOUND = 9999999
        self.special_tokens: list[str] = special_tokens
        self.special_tokens_set: set[str] = (
            set(special_tokens) if special_tokens is not None else set()
        )
        self.logger = logging.getLogger("Tokenizer")

    @classmethod
    def from_json(cls, json_filepath: str):
        """从合并的 JSON 文件构建 Tokenizer。

        该 JSON 文件由 bpe_train.py 写出，包含 vocab（bytes 以整型数组表示）、
        merges（两端 bytes 同样为整型数组）以及可选的 special_tokens。
        """
        from ch2.bpe_train import load_bpe_json

        vocab, merges, special_tokens = load_bpe_json(json_filepath)
        return cls(vocab, merges, special_tokens)

    def encode_impl(self, words: Iterable[str]) -> list[int]:
        result: list[int] = []
        for word in words:
            if word not in self.special_tokens_set:
                token_list: TokenList = bytes_to_bytes_list(
                    word.encode("utf-8")
                )
                while True:
                    connections: list[Connection] = bytes_list_to_connections(
                        token_list
                    )
                    chosen_merge_rule = None
                    min_index = self.MERGE_RULE_NOT_FOUND
                    for connection in connections:
                        if connection in self.merge_rules_index:
                            if self.merge_rules_index[connection] < min_index:
                                min_index = self.merge_rules_index[connection]
                                chosen_merge_rule = connection
                    if min_index != self.MERGE_RULE_NOT_FOUND:
                        token_list = merge_by_one_rule(
                            token_list, chosen_merge_rule
                        )
                    else:
                        break
                for token in token_list:
                    result.append(self.reverse_vocab[token])
            else:
                result.append(self.reverse_vocab[word.encode("utf-8")])
        return result


    def encode_worker(self, queue: Queue, index: int, words: Iterable[str]):
        queue.put((index, self.encode_impl(words)))

    def encode_parallel(self, text: str) -> list[int]:

        logging.info(f"encode开始")

        word_list: list[str] = chunk_split(
            text, special_tokens=self.special_tokens
        )

        THRESHOLD = len(word_list) // 6 + 1

        split_word_list = [word_list[i:i+THRESHOLD] for i in range(0, len(word_list), THRESHOLD)]

        queue = Queue()
        workers: deque[Process] = deque()
        for i,words in enumerate(split_word_list):
            worker = Process(target=self.encode_worker, args=(queue, i, words))
            workers.append(worker)

        worker_count = len(workers)

        logging.info(f"{worker_count} worker 生成完毕")

        for _ in range(min(6, worker_count)):
            workers.pop().start()

        result_list: list[list[int]] = [[]] * worker_count
        finished_worker_count = 0
        while finished_worker_count < worker_count:
            i, words = queue.get()
            finished_worker_count += 1
            logging.info(f"{finished_worker_count} worker 执行完毕")
            result_list[i] = words
            if len(workers) != 0:
                workers.pop().start()

        result: list[int] = []
        for x in result_list:
            result.extend(x)

        return result


    def encode_single(self, text: str) -> list[int]:
        word_list: list[str] = chunk_split(
            text, special_tokens=self.special_tokens
        )

        return self.encode_impl(word_list)


    def encode(self, text: str) -> list[int]:
        return self.encode_single(text)
        # return self.encode_parallel(text)


    def encode_to_file(self, text: str, dir: Path) -> None:
        dir.mkdir(exist_ok=True)
        words: Iterable[str] = chunk_split(
            text, special_tokens=self.special_tokens
        )
        buffer: list[int] = []
        id = 0
        token_count = 0
        shard_paths = []
        for word in words:
            if word not in self.special_tokens_set:
                token_list: TokenList = bytes_to_bytes_list(
                    word.encode("utf-8")
                )
                while True:
                    connections: list[Connection] = bytes_list_to_connections(
                        token_list
                    )
                    chosen_merge_rule = None
                    min_index = self.MERGE_RULE_NOT_FOUND
                    for connection in connections:
                        if connection in self.merge_rules_index:
                            if self.merge_rules_index[connection] < min_index:
                                min_index = self.merge_rules_index[connection]
                                chosen_merge_rule = connection
                    if min_index != self.MERGE_RULE_NOT_FOUND:
                        token_list = merge_by_one_rule(
                            token_list, chosen_merge_rule
                        )
                    else:
                        break
                for token in token_list:
                    buffer.append(self.reverse_vocab[token])
            else:
                buffer.append(self.reverse_vocab[word.encode("utf-8")])
            if len(buffer) >= 8*1024*1024:
                path = dir / f"{id:06d}.npy"
                shard_paths.append(path)
                np.save(path, np.array(buffer, dtype=np.uint16))
                id += 1
                token_count += len(buffer)
                buffer = []
                logging.info(id)
        path = dir / f"{id:06d}.npy"
        shard_paths.append(path)
        np.save(path, np.array(buffer, dtype=np.uint16))
        id += 1
        token_count += len(buffer)

        final = np.lib.format.open_memmap(
            dir/"result.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(token_count,),
        )
        offset = 0
        for p in shard_paths:
            x = np.load(p, mmap_mode="r")
            n = len(x)
            final[offset:offset+n] = x
            offset += n
        final.flush()


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for code in self.encode(text):
                yield code

    def encode_file(self, input_path: str)  -> Iterator[int]:
        pass

    @staticmethod
    def save_encoded_ids(
        token_ids: list[int],
        output_path: str,
        dtype: np.dtype = np.uint16,
    ) -> None:
        arr = np.asarray(token_ids, dtype=dtype)
        np.save(output_path, arr)

    @staticmethod
    def load_encoded_ids(
        input_path: str,
        mmap_mode: str | None = None,
    ) -> npt.NDArray:
        return np.load(input_path, mmap_mode=mmap_mode)

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            buf.extend(self.vocab[i])
        return bytes(buf).decode("utf-8", errors="replace")

    def decode_one(self, id: int) -> bytes:
        return self.vocab[id]
