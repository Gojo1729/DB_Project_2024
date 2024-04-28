## For running Milvus Benchmarking
- `cd milvus_benchmark`
- Run the docker compose file with the command
    docker-compose up -d
- Install pymilvus using
     `pip install pymilvus==2.1.1`
- Create the collections using the following command
    `python3 create_collection.py 127.0.0.1`
- Run the benchmark using
    `python3 benchmark.py 127.0.0.1 ./benchmark`
- To delete the collection 
    `python3 delete_collection.py`

Note:
For any new collection and index type you use in the create_collection.py, you need to update benchmark.py and delete_collection.py accordingly.


## For similar image retrieval

- `cd Code_Modified_SQLITE3`

To execute:
	`python3 image_retrieval_code.py`

the code takes no arguments.

- image_data_embedding     --> is the database for train images (8000 count)
- image_data_embedding_new --> is the database for test images (2000 count)

The python file typically uses cifar10 image datasets of 32x32 size images.

Algorithm: (Unmodified SQLITE3)
	These images are read from the cifar10 dataset
	for every image (8000)
		convert the images into embedding
		store the label, image, and embedding in the train dataset

	for every image (2000)
		convert the images into embedding
		store the label, image, and embedding in the test dataset

	Start the Timer

	for every image in test dataset
		retrieve the image
		compare the image with images in train datasets for best match using
			cosine similarity
			Structural Similarity Index


	Store the best match index for the above methods
	Store the predicted and actual class of the comparison for the above methods

	End the Timer

	Compare the accuracy.
	Check the running time of the operations.




Repeat the same algorithm with the modified(remove triggers, procedures, assertions, views) SQLITE3.dll file replaced in the locally compiled SQLITE3 database built in C:\Users\<username>\AppData\Local\Programs\Python\Python311\DLLs

The results are mentioned in the execution report.


For compilation details:
github link for sqlite3 source code : https://github.com/sqlite/sqlite/blob/master/doc/compile-for-windows.md
