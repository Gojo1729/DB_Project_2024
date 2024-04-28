import subprocess
import sys
import os
import random
import json

# Mapping of supported index types
INDEX_MAPPING = {
    "FLAT": "FLAT",
    "IVF_FLAT": "IVF_FLAT",
    "IVF_SQ8": "IVF_SQ8",
    "HNSW": "HNSW",
}

def run_sub_process(params: list):
    # Function to run a subprocess and return its stderr output
    process = subprocess.Popen(params, stderr=subprocess.PIPE)
    return process.communicate()[1].decode('utf-8')

def generate_search_embeddings(n_queries, dimension):
    # Function to generate random vectors for search
    return [[random.random() for _ in range(dimension)] for _ in range(n_queries)]

def write_to_json_file(vectors, json_file_path=""):
    # Function to write vectors to a JSON file
    if not os.path.isfile(json_file_path):
        print("[write_json_file] File(%s) does not exist." % json_file_path)
        open(json_file_path, "a").close()
    else:
        print("[write_json_file] Removing file(%s)." % json_file_path)
        os.remove(json_file_path)

    with open(json_file_path, "w") as f:
        json.dump(vectors, f)
    print("[write_json_file] Write json file:{0} done.".format(json_file_path))
    return json_file_path

def run_search(b_binary_path: str, milvus_uri: str, user: str, password: str, collection_type: str, 
                   search_parameters: dict, index_type: str, search_timeoutseconds: int, embedding_file_path: str, 
                   concurrent_searches, seconds_search_duration, seconds_logging_interval, log_file_path: str, output_format="json", 
                   partition_names=[], use_authentication=False):
    """
    Function to perform a search using a benchmark binary.

    Parameters:
    - benchmark_binary_path: Path to the benchmark binary file
    - milvus_uri: URI of the Milvus server
    - user: User name for authentication
    - password: Password for authentication
    - collection_name: Name of the collection to search
    - search_params: Parameters for search
    - index_type: Type of index to use
    - search_timeout: Timeout for search
    - vector_file_path: Path to the vector file for search
    - concurrent_num: Number of concurrent searches
    - duration: Duration of search
    - interval: Interval for logging
    - log_file_path: Path to the log file
    - output_format: Output format for search results
    - partition_names: Names of partitions to search
    - use_authentication: Whether to use authentication for connecting to Milvus
    """

    # Construct the search query JSON
    search_query_parameters = {
        "collection_name": collection_type,
        "partition_names": partition_names,
        "fieldName": search_parameters["anns_field"],
        "index_type": INDEX_MAPPING[index_type],
        "metric_type": search_parameters["metric_type"],
        "params": search_parameters["param"],
        "limit": search_parameters["limit"],
        "expr": search_parameters["expression"],
        "output_fields": [],
        "timeout": search_timeoutseconds
    }

    # Construct parameters for the benchmark binary
    benchmark_parameters = [b_binary_path,
                        'locust',
                        '-u', milvus_uri,
                        '-q', embedding_file_path,
                        '-s', json.dumps(search_query_parameters, indent=2),
                        '-p', str(concurrent_searches),
                        '-f', output_format,
                        '-t', str(seconds_search_duration),
                        '-i', str(seconds_logging_interval),
                        '-l', str(log_file_path),
                        ]
    
    # Add authentication parameters if needed
    if use_authentication:
        benchmark_parameters.extend(['-n', user, '-w', password])
        benchmark_parameters.append('-v=true')

    print("Benchmark Parameters: {}".format(benchmark_parameters))

    # Run the benchmark binary
    sub_process_results = run_sub_process(params=benchmark_parameters)
    try:
        result = json.loads(sub_process_results)
    except ValueError:
        msg = "The response from the benchmark binary is not JSON: {}".format(sub_process_results)
        raise ValueError(msg)

    if isinstance(result, dict) and "response" in result and result["response"] is True:
        print("Result of benchmark: {}".format(result))
        return result
    else:
        raise Exception(result)

if __name__ == "__main__":
    # Parameters
    INDEX_TYPE = "IVF_SQ8"
    host_address = sys.argv[1]                                          
    benchmark_executable_path = sys.argv[2]                                  

    uri = f"{host_address}:19530"                                       
    user_name = ""                                                   
    password = ""
    use_authentication = False                                         

    num_queries = 1
    embedding_dimension = 128                                              
    topk_results = 1
    ef_param = 64                                                     
    collection_type = f"random_1m_{INDEX_TYPE}"                                  
    embedding_field_name = "embedding"                                    
    metric_type = "L2"
    expression = None

    search_timeout_seconds = 600                                        
    embedding_file_path = "search_vectors.json"                        
    concurrent_searches = 10                                          
    seconds_search_duration = 60                                       
    seconds_logging_interval = 20                                       
    log_file_path = f"./logs/{INDEX_TYPE}_2/benchmark_log_{INDEX_TYPE}.log"                                      

    # Generate search vectors
    search_vectors = generate_search_embeddings(num_queries, embedding_dimension)
    write_to_json_file(vectors=search_vectors, json_file_path=embedding_file_path)

    # Prepare search parameters
    search_parameters = {
        "anns_field": embedding_field_name,
        "metric_type": metric_type,
        "param": {
            "sp_value": ef_param,
            "dim": embedding_dimension,
        },
        "limit": topk_results,
        "expression": expression,
    }

    # Execute the benchmark
    run_search(b_binary_path=benchmark_executable_path, milvus_uri=uri, user=user_name, 
                   password=password, collection_type=collection_type, search_parameters=search_parameters, 
                   index_type=INDEX_TYPE, search_timeoutseconds=search_timeout_seconds, 
                   embedding_file_path=embedding_file_path, concurrent_searches=concurrent_searches, 
                   seconds_search_duration=seconds_search_duration, seconds_logging_interval=seconds_logging_interval, 
                   log_file_path=log_file_path, use_authentication=use_authentication)
