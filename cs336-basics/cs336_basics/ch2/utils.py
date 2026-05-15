import logging
from typing import Iterator

from ch2.type_define import Connection, GPT2_PAT
import regex as re


def connection_to_str(tp: tuple[bytes, bytes]) -> str:
    return "(" + tp[0].decode("utf-8") + ", " + tp[1].decode("utf-8") + ")"


def max_connection(
    connections: list[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    return max(
        connections, key=lambda x: (x[0].decode("utf-8"), x[1].decode("utf-8"))
    )


# def split_then_merge(word: bytes, merge_rules: list[tuple[bytes,bytes]]) -> list[bytes]:
#     bytes_list = list(bytes([x]) for x in word)
#     while True:
#         merged_bytes_list = merge_once(bytes_list, merge_rules)
#         if merged_bytes_list is None:
#             return bytes_list
#         else:
#             bytes_list = merged_bytes_list


def bytes_to_bytes_list(word: bytes) -> list[bytes]:
    return list(bytes([x]) for x in word)


def merge_once(
    bytes_list: list[bytes], merge_rules: list[tuple[bytes, bytes]]
) -> list[bytes] | None:
    for merge_rule in merge_rules:
        merge_result: list[bytes] = []
        idx = 1
        merged = False
        while idx < len(bytes_list):
            if (bytes_list[idx - 1], bytes_list[idx]) == merge_rule:
                merge_result.append(bytes_list[idx - 1] + bytes_list[idx])
                idx += 2
                merged = True
            else:
                merge_result.append(bytes_list[idx - 1])
                idx += 1
        if merged:
            return merge_result
    return None


def merge_by_one_rule(
    bytes_list: list[bytes], merge_rule: tuple[bytes, bytes]
) -> list[bytes] | None:
    merge_result: list[bytes] = []
    idx = 0
    merged = False

    while idx + 1 < len(bytes_list):
        if (bytes_list[idx], bytes_list[idx + 1]) == merge_rule:
            # 可合并
            merge_result.append(bytes_list[idx] + bytes_list[idx + 1])
            idx += 2
            merged = True
        else:
            # 不可合并
            merge_result.append(bytes_list[idx])
            idx += 1

    if idx < len(bytes_list):
        # 处理尾部
        merge_result.append(bytes_list[-1])

    return merge_result if merged else None


def bytes_list_to_connections(bytes_list: list[bytes]) -> list[Connection]:
    result: list[Connection] = []
    for i in range(1, len(bytes_list)):
        result.append((bytes_list[i - 1], bytes_list[i]))
    return result


def chunk_split(
    chunk: str,
    special_tokens: list[str] | None,
) -> Iterator[str]:
    contents: list[str]
    if special_tokens is not None:
        special_tokens.sort(key=lambda x: len(x), reverse=True)
        pattern = "|".join(re.escape(token) for token in special_tokens)
        pattern = "(" + pattern + ")"
        contents = [c for c in re.split(pattern, chunk) if c]
        # logging.info("special tokens split")
    else:
        contents = [chunk]
    special_tokens_set: set[str] = (
        set(special_tokens) if special_tokens is not None else set()
    )
    pattern = re.compile(GPT2_PAT)
    count = 0
    for content in contents:
        if content not in special_tokens_set:
            for match in pattern.finditer(content):
                yield match.group()
                count += 1
                if count % 65536 == 0:
                    logging.info(count)
        else:
            yield content
