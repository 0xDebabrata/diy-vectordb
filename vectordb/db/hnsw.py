import hnswlib


class HnswIndex:
    _index: hnswlib.Index

    def __init__(self, space="cosine", dim=1536, max_elements=1000, M=64, ef_construction=200, allow_replace_deleted=False):
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements, M, ef_construction, allow_replace_deleted)

    def add_items(self, data, ids, replace_deleted=False):
        max_elements = self._index.max_elements
        curr_elements = self._index.element_count

        # Increase index size
        if curr_elements + len(data) > max_elements:
            new_size = max(curr_elements + len(data), 2 * max_elements)
            self._index.resize_index(new_size)

        self._index.add_items(data, ids, replace_deleted)

    def knn_query(self, query_embeddings, k=1, filter_function=None):
        labels, distances = self._index.knn_query(query_embeddings, k, filter=filter_function)
        return labels, distances

    def mark_deleted(self, label: int):
        self._index.mark_deleted(label)
