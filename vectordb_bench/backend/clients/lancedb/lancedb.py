import logging
from contextlib import contextmanager
from typing import Any, Type
from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import LancedbConfig

import pyarrow as pa
import lancedb
import numpy as np


log = logging.getLogger(__name__)
INDEX_NAME = "index"                              # Vector Index Name
TABLE_NAME = "vector_table"

class Lancedb(VectorDB):
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

        # Create a redis connection, if db has password configured, add it to the connection here and in init():
        self.url=self.db_config["url"]
        db =  lancedb.connect(self.url)

        if drop_old:
            for table in db.table_names():
                if table == self.table_name:
                    db.drop_table(self.table_name)
        custom_schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim))
        ])

        db.create_table(self.table_name, exist_ok=True, schema=custom_schema)
        db = None

    def make_index(self, vector_dimensions: int, db: lancedb.DBConnection):
        db =  lancedb.connect(self.url)
        table = db.open_table(self.table_name)
        table.create_scalar_index('id')
        table.create_index(vector_column_name='vector', metric = 'cosine', num_partitions=256, num_sub_vectors=96)

    @contextmanager
    def init(self) -> None:
        db =  lancedb.connect(self.url)
        db.open_table(self.table_name)
        yield

    def ready_to_search(self) -> bool:
        db =  lancedb.connect(self.url)
        self.make_index(self.dim, db)
        return True

    def ready_to_load(self) -> bool:
        pass

    def optimize(self) -> None:
        db =  lancedb.connect(self.url)
        table = db.open_table(self.table_name)
        table.compact_files()
        table.cleanup_old_versions()
        self.make_index(self.dim, db)
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
            db =  lancedb.connect(self.url)
            table = db.open_table(self.table_name)
            data = [{"vector": np.array(embeddings_arr[i]).astype(np.float32), "id": m}
                for i, m in enumerate(metadata_arr)]
            table.add(data=data)
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
        db =  lancedb.connect(self.url)
        table = db.open_table(self.table_name)
        if filters:
            expr = f"{self._scalar_field} {filters.get('metadata')}"
            df = table.search(np.array(query).astype(np.float32)).nprobes(20).refine_factor(10).where(expr, prefilter=True).limit(k).select(["id"]).to_pandas()
            return df['id'].tolist()
        else:
            df = table.search(np.array(query).astype(np.float32)).limit(k).nprobes(20).refine_factor(10).select(["id"]).to_pandas()
            return df['id'].tolist()
