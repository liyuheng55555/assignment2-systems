Connection = tuple[bytes, bytes]
Index = int
Num = int
TokenList = list[bytes]
GB: int = 1024 * 1024 * 1024
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
