import logging
from contextlib import contextmanager
from typing import Any, Type
from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import LanceConfig

import pyarrow as pa
import vectordb_bench.backend.clients.lance.lance as lance
import numpy as np
import lance


log = logging.getLogger(__name__)
INDEX_NAME = "index"                              # Vector Index Name
TABLE_NAME = "vector_table"

class Lance(VectorDB):
    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: DBCaseConfig,
            drop_old: bool = False,
            **kwargs
        ):

        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = TABLE_NAME
        self.dim = dim
        self._scalar_field = "id"
        self.schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim))
        ])
        self.url=self.db_config["url"]
        self.first_insert = False

    @contextmanager
    def init(self) -> None:
        yield

    def ready_to_search(self) -> bool:
        pass

    def ready_to_load(self) -> bool:
        pass

    def optimize(self) -> None:
        ds = lance.dataset(self.url)
        ds.create_scalar_index('id', index_type='BTREE')
        ds.create_index(column='vector', index_type='IVF_PQ', num_partitions=256, num_sub_vectors=96, replace=True)
        ds.optimize.compact_files()
        ds.optimize.optimize_indices()
        ds.cleanup_old_versions()
        print("Index created.")

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        """Insert embeddings into the database.
        Should call self.init() first.
        """
        metadata_arr = np.array(metadata)
        embeddings_arr = np.array(embeddings)
        try:
            data = [{"vector": np.array(embeddings_arr[i]).astype(np.float32), "id": m}
                for i, m in enumerate(metadata_arr)]
            import pandas as pd
            if self.first_insert:
                lance.write_dataset(pd.DataFrame(data), self.url, schema= self.schema, mode='overwrite', data_storage_version = 'stable')
            else:
                lance.write_dataset(pd.DataFrame(data), self.url, schema= self.schema, mode='append', data_storage_version = 'stable')
        except Exception as e:
            return 0, e
        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> (list[int]):
        ds = lance.dataset(self.url)
        if filters:
            expr = f"{self._scalar_field} {filters.get('metadata')}"
            table = ds.scanner(
                    columns=['id'],
                    nearest=dict(
                        column="vector",
                        q=np.array(query).astype(np.float32),
                        k=k,
                        nprobes=20,
                        refine_factor=10,
                    ),
                    limit=k,
                    filter=expr,
                    prefilter=True
            ).to_table()
            return table.column(0).to_pylist()
        else:
            table = ds.scanner(
                    columns=['id'],
                    nearest=dict(
                        column="vector",
                        q=np.array(query).astype(np.float32),
                        k=k,
                        nprobes=20,
                        refine_factor=10,
                    ),
                    limit=k,
                    prefilter=True
            ).to_table()
            return table.column(0).to_pylist()
