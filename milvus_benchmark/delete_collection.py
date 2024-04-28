import time
import sys
import random
from pathlib import Path
import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema, utility

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

nb = 50000
dim = 128
auto_id = True
INDEX_TYPE = "IVF_SQ8"

if __name__ == '__main__':
    host = "127.0.0.1" # host address
    shards = 1          # shards number
    insert_times = 20   # insert times

    port = 19530
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')   
    collection_name = f"random_1m_{INDEX_TYPE}"
    id = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
    age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[id, age_field, embedding_field],
                              auto_id=auto_id, primary_field=id.name,
                              description="my collection")
    print(utility.has_collection(collection_name))
    collection = Collection(name=collection_name, schema=schema, shards_num=shards)
    collection.drop()
    print(utility.has_collection(collection_name))
