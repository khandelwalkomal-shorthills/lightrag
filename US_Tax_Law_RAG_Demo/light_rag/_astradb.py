from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.exceptions import DataAPIException
from dataclasses import dataclass
from dotenv import load_dotenv
from lightrag.base import BaseVectorStorage, BaseKVStorage
from logging import getLogger
import asyncio
import numpy as np
import os, json

load_dotenv()
f_VECTOR = "$vector"
logger = getLogger("astra-db")

@dataclass
class AstraDBVectorStorage(BaseVectorStorage):

    def __post_init__(self):
        self.astradb_application_token = os.getenv("ASTRADB_APPLICATION_TOKEN") or self.global_config["vector_db_storage_cls_kwargs"]["ASTRADB_APPLICATION_TOKEN"]
        self.astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT") or self.global_config["vector_db_storage_cls_kwargs"]["ASTRADB_API_ENDPOINT"]
        self.embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS") or self.global_config["vector_db_storage_cls_kwargs"]["EMBEDDING_DIMENSIONS"]

        if not all(
            [
                self.astradb_application_token,
                self.astradb_api_endpoint,
                self.embedding_dimensions,
            ]
        ):
            raise ValueError(
                "AstraDB application token, API endpoint, and embedding dimensions are required fields"
            )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self.storage = None
        self._client = self.get_or_create_collection()

    def get_client(self):
        try:
            client = DataAPIClient(token=self.astradb_application_token)
            return client
        except DataAPIException as e:
            self.logger.error(e)
            raise e

    def get_database(self):
        db = self.get_client().get_database(self.astradb_api_endpoint)
        return db

    def get_or_create_collection(self):
        if not all([self.namespace, self.embedding_dimensions]):
            raise ValueError(
                "Error getting collection. Namespace and dimensions are required fields"
            )
        database = self.get_database()
        collection = database.get_collection(self.namespace)
        if collection is None:
            collection = database.create_collection(
                namespace="default_keyspace",
                name=self.namespace,
                dimension=self.embedding_dimensions,
                metric=VectorMetric.COSINE,
                check_exists=False,
            )
        return collection

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            self.logger.warning("You insert an empty data to vector DB")
            return []
        _data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in _data]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(_data):
            d[f_VECTOR] = embeddings[i]

        self.storage = _data
        return self.storage

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]

        results = self._client.find(
            vector=embedding, limit=top_k, include_similarity=True
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["$similarity"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        if self.storage is not None:
            self._client.insert_many(documents=self.storage)


@dataclass
class AstraDBKVStorage(BaseKVStorage):

    def __post_init__(self):
        self.astradb_application_token = os.getenv("ASTRADB_APPLICATION_TOKEN") or self.global_config["vector_db_storage_cls_kwargs"]["ASTRADB_APPLICATION_TOKEN"]
        self.astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT") or self.global_config["vector_db_storage_cls_kwargs"]["ASTRADB_API_ENDPOINT"]
        self.embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS") or self.global_config["vector_db_storage_cls_kwargs"]["EMBEDDING_DIMENSIONS"]

        if not all(
            [
                self.astradb_application_token,
                self.astradb_api_endpoint,
                self.embedding_dimensions,
            ]
        ):
            raise ValueError(
                "AstraDB application token, API endpoint, and embedding dimensions are required fields"
            )

        self._max_batch_size = self.global_config["embedding_batch_num"]
        self.storage = None
        self._client = self.get_or_create_collection()

    def get_client(self):
        try:
            client = DataAPIClient(token=self.astradb_application_token)
            return client
        except DataAPIException as e:
            self.logger.error(e)
            raise e

    def get_database(self):
        db = self.get_client().get_database(self.astradb_api_endpoint)
        return db

    def get_or_create_collection(self):
        if not all([self.namespace, self.embedding_dimensions]):
            raise ValueError(
                "Error getting collection. Namespace and dimensions are required fields"
            )
        database = self.get_database()
        collection = database.get_collection(self.namespace)
        if collection is None:
            collection = database.create_collection(
                namespace="default_keyspace",
                name=self.namespace,
                dimension=self.embedding_dimensions,
                metric=VectorMetric.COSINE,
                check_exists=False,
            )
        return collection

    async def all_keys(self) -> list[str]:
        try:
            data = self._client.find({}, limit=1)
            if data is not None:
                return list(data.keys())
        except Exception as e:
            self.logger.error(e)
            return []


    async def get_by_id(self, id):
        try:
            data = self._client.find_one({"__id__": id})
            return data or None
        except Exception as e:
            self.logger.error(e)
            return None
        
    async def get_by_ids(self, ids, fields=None):
        pass


    async def filter_keys(self, data: list[str]) -> set[str]:
        s = set()
        for chunk_id in data:
            result = self._client.find_one({"__id__": chunk_id})
            if result is None:
                s.add(chunk_id)

        return s
    
    async def upsert(self, data: dict[str, dict]):
        pass

    async def drop(self):
        pass

    async def index_done_callback(self):
        pass