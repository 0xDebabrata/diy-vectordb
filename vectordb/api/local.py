import os
import json
from typing import Dict, List, Optional
from numpy import float32
from numpy._typing import NDArray
import shutil

from vectordb.db.hnsw import HnswIndex
from vectordb.db.sqlite import SQLiteDB


class LocalAPI:
    _indices: Dict[str, HnswIndex] 
    _SQLClient: SQLiteDB
    _TEMP_DIRECTORY = "citrus_temp"

    def __init__(self):
        self._indices = {}

        if os.path.isdir(self._TEMP_DIRECTORY):
            # Cleanup previous sqlite data
            shutil.rmtree(self._TEMP_DIRECTORY)

        self._SQLClient = SQLiteDB(self._TEMP_DIRECTORY)

    def create_index(
        self,
        name: str,
        dimension: int = 1536,
        max_elements: int = 1000,
        M: int = 64,
        ef_construction: int = 200,
        allow_replace_deleted: bool = False,
    ):
        if not(self._SQLClient.get_index_details(name)):
            self._SQLClient.create_index(
                name,
                max_elements,
                M,
                ef_construction,
                allow_replace_deleted,
                dimensions=dimension
            )

        self._indices[name] = HnswIndex(
            max_elements=max_elements,
            M=M,
            ef_construction=ef_construction,
            allow_replace_deleted=allow_replace_deleted,
            dim=dimension
        )

    def add(
        self,
        index: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Insert embeddings/text documents

        index: Name of index
        ids: Unique ID for each element
        documents: List of strings to index
        embeddings: List of embeddings to index
        metadatas: Additional metadata for each vector
        """

        if embeddings is None and documents is None:
            raise ValueError("Please provide either embeddings or documents.")

        index_details = self._SQLClient.get_index_details(index)
        if index_details is None:
            raise ValueError(f"Index with name '{index}' does not exist.")

        if embeddings is not None:
            embedding_dim = len(embeddings[0])
            index_id = index_details[0]
            index_dim = index_details[2]
            replace_deleted = True if index_details[7] else False

            # Check whether the dimensions are equal
            if embedding_dim != index_dim:
                raise ValueError(
                        f"Embedding dimension ({embedding_dim}) and index "
                        + f"dimension ({index_dim}) do not match."
                        )

            # Ensure no of ids = no of embeddings
            if len(ids) != len(embeddings):
                raise ValueError(f"Number of embeddings" + " and ids are different.")

            data = []
            for i in range(len(ids)):
                row = (
                    ids[i],
                    index_id,
                    None if documents is None else documents[i],
                    json.dumps(embeddings[i]),
                    None if metadatas is None else json.dumps(metadatas[i])
                )
                data.append(row + row)

            # Insert data into DB
            hnsw_labels = self._SQLClient.insert_to_index(data)

            # Index vectors
            self._indices[index].add_items(
                embeddings,
                hnsw_labels,
                replace_deleted,
            )

    def delete_vectors(
        self,
        index: str,
        ids: list[str], 
    ):
        index_details = self._SQLClient.get_index_details(index)
        if index_details is None:
            raise ValueError(f"Could not find index: {index}")

        index_id = index_details[0]
        hnsw_labels = self._SQLClient.delete_vectors_from_index(
            index_id=index_id,
            ids=ids
        )

        for id in hnsw_labels:
            self._indices[index].mark_deleted(id)

    def query(
        self,
        index: str,
        query_embeddings: list[list[float]],
        k=1,
        filters: Optional[List[Dict]] = None,
        include: List[str] = []
    ):
        allowed_ids = []
        if filters is not None:
            allowed_ids = self._SQLClient.filter_vectors(index, filters)

        filter_function = lambda label: label in allowed_ids

        included_columns = {"id": True, "document": False, "metadata": False}
        if "document" in include:
            included_columns["document"] = True
        if "metadata" in include:
            included_columns["metadata"] = True

        flag = 1
        for key in self._indices.keys():
            if key == index:
                flag = 0

                results, distances = self._indices[key].knn_query(
                    query_embeddings,
                    k=k,
                    filter_function=None if filters is None else filter_function
                )
                elements = self._SQLClient.get_vector_ids_of_results(
                    name=index,
                    results=results,
                    include=included_columns
                )
                for i, rows in enumerate(elements):
                    for j, row in enumerate(rows):
                        row["distance"] = distances[i][j]
                return elements
        if flag:
            raise ValueError(f"Could not find index: {index}")
