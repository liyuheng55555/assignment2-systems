import logging
from codecs import getincrementaldecoder
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from time import perf_counter

import numpy as np

from cs336_basics.pretokenization_example import find_chunk_boundaries
from ch2.Tokenizer import Tokenizer


def _encode_range_worker(
    queue: Queue,
    worker_id: int,
    tokenizer_json_path: str,
    input_path: Path,
    start: int,
    end: int,
    worker_dir: Path,
    read_chunk_bytes: int = 128 * 1024 * 1024,
    shard_token_threshold: int = 8 * 1024 * 1024,
) -> int:
    worker_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_json(tokenizer_json_path)

    decoder = getincrementaldecoder("utf-8")()
    carry = ""
    buffer: list[int] = []
    token_count = 0
    shard_id = 0
    shard_paths: list[Path] = []

    def flush_buffer() -> None:
        nonlocal shard_id, token_count, buffer
        if not buffer:
            return
        path = worker_dir / f"{shard_id:06d}.npy"
        shard_paths.append(path)
        np.save(path, np.array(buffer, dtype=np.uint16))
        token_count += len(buffer)
        shard_id += 1
        buffer = []

    with open(input_path, "rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            raw = f.read(min(read_chunk_bytes, remaining))
            if not raw:
                break
            remaining -= len(raw)

            text = carry + decoder.decode(raw, final=False)
            lines = text.splitlines(keepends=True)
            carry = ""
            if lines and not text.endswith(("\n", "\r")):
                carry = lines.pop()

            for line in lines:
                buffer.extend(tokenizer.encode(line))
                if len(buffer) >= shard_token_threshold:
                    flush_buffer()

        tail = carry + decoder.decode(b"", final=True)
        if tail:
            buffer.extend(tokenizer.encode(tail))
    flush_buffer()

    final = np.lib.format.open_memmap(
        worker_dir / "result.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(token_count,),
    )

    offset = 0
    for shard in shard_paths:
        x = np.load(shard, mmap_mode="r")
        n = len(x)
        final[offset : offset + n] = x
        offset += n
    final.flush()

    queue.put((worker_id, token_count))


def encode_file_multiprocess(
    tokenizer_json_path: Path,
    input_path: Path,
    output_dir: Path,
    num_workers: int | None = None,
    read_chunk_bytes: int = 128 * 1024 * 1024,
    shard_token_threshold: int = 8 * 1024 * 1024,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_for_boundaries = Tokenizer.from_json(str(tokenizer_json_path))
    special_tokens = tokenizer_for_boundaries.special_tokens or []
    del tokenizer_for_boundaries
    split_markers = [t.encode("utf-8") for t in special_tokens] or [b"\n"]

    if num_workers is None:
        num_workers = min(8, cpu_count())
    num_workers = max(1, num_workers)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            file=f,
            desired_num_chunks=num_workers,
            split_special_tokens=split_markers,
        )
    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    if not ranges:
        raise RuntimeError("No chunk ranges generated for input file.")

    queue: Queue = Queue()
    workers: list[Process] = []
    worker_dirs: list[Path] = []
    for i, (start, end) in enumerate(ranges):
        worker_dir = output_dir / f"worker_{i:02d}"
        worker_dirs.append(worker_dir)
        p = Process(
            target=_encode_range_worker,
            args=(
                queue,
                i,
                str(tokenizer_json_path),
                input_path,
                start,
                end,
                worker_dir,
                read_chunk_bytes,
                shard_token_threshold,
            ),
        )
        workers.append(p)

    for p in workers:
        p.start()

    token_count_by_worker = [0] * len(workers)
    finished = 0
    while finished < len(workers):
        worker_id, token_count = queue.get()
        token_count_by_worker[worker_id] = token_count
        finished += 1
        logging.info("worker %s finished (%s/%s)", worker_id, finished, len(workers))

    for p in workers:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker pid={p.pid} exited with code {p.exitcode}")

    total_tokens = sum(token_count_by_worker)
    final = np.lib.format.open_memmap(
        output_dir / "result.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    offset = 0
    for i, worker_dir in enumerate(worker_dirs):
        worker_result = np.load(worker_dir / "result.npy", mmap_mode="r")
        n = len(worker_result)
        if n != token_count_by_worker[i]:
            raise RuntimeError(
                f"worker {i} token count mismatch: expected {token_count_by_worker[i]}, got {n}"
            )
        final[offset : offset + n] = worker_result
        offset += n
    final.flush()

    return total_tokens


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_path = Path("/data/cs336/data/owt_train.txt")
    tokenizer_file_path = Path("/data/cs336/data/owt_train.json")
    output_dir = Path("/data/cs336/data/owt_train_tokenized")

    start = perf_counter()
    total_tokens = encode_file_multiprocess(
        tokenizer_json_path=tokenizer_file_path,
        input_path=input_path,
        output_dir=output_dir,
        num_workers=8,
    )
    t2 = perf_counter()

    result = np.load(output_dir / "result.npy", mmap_mode="r")
    total_time = t2 - start

    print(f"tokens: {total_tokens}")
    print(f"result shape: {result.shape}, dtype: {result.dtype}")
    print("Timing (seconds):")
    print("  mode          : multiprocess")
    print("  workers       : 8")
    print(f"  total         : {total_time:.3f}")
