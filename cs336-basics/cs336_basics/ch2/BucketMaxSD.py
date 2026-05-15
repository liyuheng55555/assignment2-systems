import logging

from sortedcontainers import SortedDict

from ch2.type_define import Connection, Num


class BucketMaxSD:
    def __init__(self):
        self.counts: dict[Connection, Num] = {}
        self.buckets = SortedDict()  # Num->Set(Connection)
        self.set_count = 0
        self.bucket_create_destroy_count = 0

    def _add_to_bucket(self, num: Num, connection: Connection):
        if num not in self.buckets:
            self.buckets[num] = set()
            self.bucket_create_destroy_count += 1
        self.buckets[num].add(connection)

    def _delete_from_bucket(self, num: Num, connection: Connection):
        self.buckets[num].remove(connection)
        if len(self.buckets[num]) == 0:
            del self.buckets[num]
            self.bucket_create_destroy_count += 1

    def set(self, connection: Connection, num: Num):
        self.set_count += 1
        if connection not in self.counts:
            self.counts[connection] = num
            self._add_to_bucket(num, connection)
            return
        # delete old one
        old_num = self.counts[connection]
        self._delete_from_bucket(old_num, connection)
        # add new one
        self.counts[connection] = num
        self._add_to_bucket(num, connection)

    def incr(self, conn, d=1):
        self.set(conn, self.counts.get(conn, 0) + d)

    def decr(self, conn, d=1):
        self.set(conn, self.counts.get(conn, 0) - d)

    def max_item(self) -> tuple[Connection, Num]:
        num: Num
        s: set[Connection]
        num, s = self.buckets.peekitem(-1)
        return max(s), num

    def density(self) -> float:
        if len(self.counts) == 0:
            return -1
        return len(self.buckets) / len(self.counts)

    def average_bucket_size(self) -> float:
        if len(self.buckets) == 0:
            return 0
        return len(self.counts) / len(self.buckets)

    def churn(self) -> float:
        return self.bucket_create_destroy_count / self.set_count

    def message(self):
        logging.info(
            f"map size: {len(self.counts)}, bucket average size: {self.average_bucket_size()}, churn: {self.churn()}"
        )

    def __len__(self):
        return len(self.counts)

    def __contains__(self, item):
        if not isinstance(item, tuple):
            return False
        return item in self.counts
