from pydantic import SecretStr, BaseModel
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

class LancedbConfig(DBConfig):
    url: SecretStr

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
        }


class LancedbIndexConfig(BaseModel):
    """Base config for milvus"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""
        return self.metric_type.value

class LancedbHNSWConfig(LancedbIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }
