import time
import sys
import random
from pathlib import Path
import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema

# Define log format and date format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

# Define constants for collection setup
num_entities = 50000
embedding_dim = 128
auto_generate_id = True
INDEX_TYPE = "IVF_SQ8"
index_params = {"index_type": INDEX_TYPE, "params": {"nlist": 8}, "metric_type": "L2"}

if __name__ == '__main__':
    # Extract host address from command line arguments
    host_address = sys.argv[1]

    # Define other parameters
    shards_number = 1
    insert_iterations = 20

    # Set up connection to Milvus server
    port = 19530
    connections.add_connection(default={"host": host_address, "port": port})
    connections.connect('default')

    # Set up logging
    log_name = "collection_setup"
    logs_path = Path(f"./logs/{INDEX_TYPE}_2")
    if not logs_path.exists():
        logs_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=f"./logs/{INDEX_TYPE}_2/{log_name}_{INDEX_TYPE}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info("Starting collection setup process...")

    # Define collection name and schema
    collection_name = f"random_1m_{INDEX_TYPE}"
    field_id = FieldSchema(name="id", dtype=DataType.INT64, description="Auto-generated primary ID")
    field_age = FieldSchema(name="age", dtype=DataType.INT64, description="Age of the individual")
    field_embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    schema = CollectionSchema(fields=[field_id, field_age, field_embedding],
                              auto_id=auto_generate_id, primary_field=field_id.name,
                              description="Collection containing random embeddings")

    # Create the collection
    collection = Collection(name=collection_name, schema=schema, shards_num=shards_number)
    logging.info(f"Collection '{collection_name}' created successfully")

    # Insert data into the collection
    for i in range(insert_iterations):
        # Prepare random data for insertion
        age_data = [random.randint(1, 100) for _ in range(num_entities)]
        data_embeddings = [[random.random() for _ in range(embedding_dim)] for _ in range(num_entities)]
        data = [age_data, data_embeddings]

        # Insert data and log insertion time
        t0 = time.time()
        collection.insert(data)
        insert_time = round(time.time() - t0, 3)
        logging.info(f"Insertion {i} completed in {insert_time} seconds")

    # Flush data and log the number of entities in the collection
    conn = collection._get_connection()
    conn.flush([collection.name])
    logging.info(f"Number of entities in collection: {collection.num_entities}")

    # Build index for the embedding field
    t0 = time.time()
    collection.create_index(field_name=field_embedding.name, index_params=index_params)
    index_build_time = round(time.time() - t0, 3)
    logging.info(f"Index build completed in {index_build_time} seconds")

    # Load the collection and mark setup process as completed
    collection.load()
    logging.info("Collection setup completed")
