class Vocab:
    def __init__(self):
        self._vocab: dict[int, bytes] = {}
        self._counter: int = 0  # 从 0 开始计数

    def put(self, token: bytes) -> int:
        key = self._counter
        self._vocab[key] = token
        self._counter += 1
        return key

    def build_dict(self):
        return self._vocab

    def __len__(self) -> int:
        return len(self._vocab)

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)}, next_id={self._counter})"

    def __str__(self) -> str:
        return str(self._vocab)
