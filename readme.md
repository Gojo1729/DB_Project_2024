- Run the docker compose file with the command
    docker-compose up -d
- Install pymilvus using
     pip install pymilvus==2.1.1
- Create the collections using the following command
    python3 create_collection.py 127.0.0.1
- Run the benchmark using
    python3 benchmark.py 127.0.0.1 ./benchmark
- To delete the collection 
    python3 delete_collection.py

Note:
For any new collection and index type you use in the create_collection.py, you need to update benchmark.py and delete_collection.py accordingly.