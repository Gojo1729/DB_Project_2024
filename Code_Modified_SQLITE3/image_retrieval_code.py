'''

Version  Date           Name               Description

  1.0    04/15/2024     Ayushi S      Created and added file
  1.1    04/19/2024     Ayushi S      Added Embedded conversion using python
  1.2    04/23/2024     Ashwin Sai C  Added Embedded comparision retreival using cosine similarity
  1.3    04/27/2024     Ashwin Sai C  Added Image SSIM comparison retreival
  1.4    04/27/2024     Ashwin Sai C  Accuracy metric added to measure performance
  1.5    04/27/2024     Ayushi S      Added Timing measurement

python3 image_retrieval.py

'''

import time
import sqlite3
from   PIL import Image
import numpy as np
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from   sklearn.metrics.pairwise import cosine_similarity
import pickle
import skimage

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calculate_embedding(image_array):
    # Convert image_array to PIL Image
    image = Image.fromarray((image_array.reshape(3, 32, 32).transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)
    
    # Calculate embedding using ResNet model
    with torch.no_grad():
        embedding = model(image_tensor).squeeze().numpy()
    
    return embedding

def insert_image(image_array, label):
    # Calculate embedding
    embedding = calculate_embedding(image_array)
    
    # Convert image_array to PIL Image
    image = Image.fromarray((image_array.reshape(3, 32, 32).transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Convert PIL Image to binary data
    with io.BytesIO() as f:
        image.save(f, format='PNG', quality=95)
        image_binary = f.getvalue()
        
    # Convert embedding to binary data
    embedding_binary = sqlite3.Binary(embedding.tobytes())
    
    # Insert data into database
    cursor.execute('''INSERT INTO images (label, image, embedding) VALUES (?, ?, ?)''', (label, sqlite3.Binary(image_binary), embedding_binary))
    conn.commit()

def insert_image_Test(image_array, label):
    # Calculate embedding
    embedding = calculate_embedding(image_array)
    
    # Convert image_array to PIL Image
    image = Image.fromarray((image_array.reshape(3, 32, 32).transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Convert PIL Image to binary data
    with io.BytesIO() as f:
        image.save(f, format='PNG', quality=95)
        image_binary = f.getvalue()
        
    # Convert embedding to binary data
    embedding_binary = sqlite3.Binary(embedding.tobytes())
    
    # Insert data into database
    cursor.execute('''INSERT INTO images_test (label, image, embedding) VALUES (?, ?, ?)''', (label, sqlite3.Binary(image_binary), embedding_binary))
    conn.commit()

def get_test_image_embedding(image_id):
    cursor.execute('''SELECT label, embedding FROM images_test WHERE id = ?''', (image_id,))
    label, embedding_binary = cursor.fetchone()
    

    # image = Image.open(io.BytesIO(image_binary))
    
    embedding = np.frombuffer(embedding_binary, dtype=np.float32).reshape(-1)
    print("Test : ", np.array(embedding).shape)
    
    return label, embedding

def get_test_image_vector(image_id):
    cursor.execute('''SELECT label, image FROM images_test WHERE id = ?''', (image_id,))
    label, image_binary = cursor.fetchone()
    
    # image = Image.open(io.BytesIO(image_binary))
    
    # embedding = np.frombuffer(embedding_binary, dtype=np.float32).reshape(-1)
    
    return label, image_binary

def retrieve_image_with_embedding(image_id, test_embedding, test_class):
    cursor.execute('''SELECT label, embedding FROM images''')
    embedding_list  = cursor.fetchall()
    similarity_list = []
    class_list      = []
    output_list     = []

    # print("len(embedding_binary) : ",len(embedding_list[0][1]))

    for index in range(0,8000):
        element    = np.frombuffer(embedding_list[index][1], dtype=np.float32).reshape(-1)        
        similarity = cosine_similarity(test_embedding.reshape(1,-1),element.reshape(1,-1))
        # print(similarity)
        similarity_list.append(similarity)        
        class_list.append(embedding_list[index][0])

    # print(class_list)
    # print("Len(X) : ",len(similarity_list))
    # print("Len(Y) : ",len(class_list))

    max_similarity = max(similarity_list)
    index_max      = similarity_list.index(max_similarity)
    print("---------Image ID        : ",image_id,"---------")
    print("Max similarity  : ",max_similarity)
    print("Index           : ",index_max)
    print("Predicted Class : ",class_list[index_max])
    print("Actual Class    : ",test_class)

    return class_list[index_max]

def retrieve_image_with_image(image_id, test_image, test_class):
    cursor.execute('''SELECT label, image FROM images''')
    image_list  = cursor.fetchall()
    similarity_list = []
    SSIM_list       = []
    class_list      = []

    # print("len(embedding_binary) : ",len(image_list))
    # print(type(image_list[0][1]))

    for index in range(0,8000):
        # Convert bytes to NumPy arrays
        image1 = skimage.io.imread(test_image, plugin='imageio')
        image2 = skimage.io.imread(image_list[index][1], plugin='imageio')

        # Calculate SSIM and MSE
        ssim_score = skimage.metrics.structural_similarity(image1, image2,win_size=3)
        mse_score  = skimage.metrics.mean_squared_error(image1, image2)
        
        # print("SSIM Score:", ssim_score)
        # print("MSE Score :", mse_score)
        SSIM_list.append(ssim_score)
        class_list.append(image_list[index][0])
    
    print("---------Image ID  : ",image_id,"---------")
    best_SSIM  = max(SSIM_list)
    best_index = SSIM_list.index(best_SSIM)
    print("Predicted Class : ",class_list[best_index])
    print("Actual Class    : ",test_class)

    return class_list[best_index]        

def initialize_database():
    cifar10_data = unpickle('data_batch_1')
    labels       = unpickle('batches.meta')

    # Print dictionary keys to understand the structure
    print("cifar10_data keys:", cifar10_data.keys())
    print("-----------------------------------------")
    print("labels keys:", labels.keys())


    # Connect to SQLite database
    conn = sqlite3.connect('image_data_embedding_new.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY,
                        label TEXT,
                        image BLOB,
                        embedding BLOB
                    )''')

    # Create tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS images_test (
                        id INTEGER PRIMARY KEY,
                        label TEXT,
                        image BLOB,
                        embedding BLOB
                    )''')

    return cifar10_data, cursor

def initialize_train_database(cifar10_data):
    count_ = 8000
    # Example usage: Inserting images and embeddings from unpickled CIFAR-10 data
    for i, (image_array, label) in enumerate(zip(cifar10_data[b'data'][:count_], cifar10_data[b'labels'][:count_])):
        print("Inserting Image : ",i)
        insert_image(image_array, labels[b'label_names'][label].decode())

def initialize_test_database(cifar10_data):
    count_ = 8000
    #Example usage: Inserting images and embeddings from unpickled CIFAR-10 data
    for i, (image_array, label) in enumerate(zip(cifar10_data[b'data'][count_:], cifar10_data[b'labels'][count_:])):
        print("Inserting Test Image : ",i)
        insert_image_Test(image_array, labels[b'label_names'][label].decode())

def validate_embedding_dataset(cifar10_data):
    # # Example usage: Retrieving an image by embedding
    count_      = 100
    output_list = []
    for image_id in range(1,count_):
        test_class, test_embedding = get_test_image_embedding(image_id)
        predicted_class = retrieve_image_with_embedding(image_id, test_embedding, test_class)
        output_list.append(test_class == predicted_class)

    print("------------")
    print(output_list)
    print("Accuracy : ",round(np.mean(output_list),2)*100,"%")

def validate_image_dataset(cifar10_data):
    # #Example usage: Retrieving an image by image
    count_      = 20
    output_list = []
    for image_id in range(1,count_):
        test_class, test_image = get_test_image_vector(image_id)
        predicted_class = retrieve_image_with_image(image_id, test_image, test_class)
        output_list.append(test_class == predicted_class)

    print("------------")
    print(output_list)
    print("Accuracy : ",round(np.mean(output_list),2)*100,"%")



if __name__ == "__main__":

    start_time = time.time()
    cifar10_data, cursor = initialize_database()

    print("Number of Rows   : ",len(cifar10_data[b'data']))
    print("Number of Labels : ",len(cifar10_data[b'labels']))
    
    # initialize_train_database(cifar10_data)
    # initialize_test_database(cifar10_data)
    validate_embedding_dataset(cifar10_data)   
    # validate_image_dataset(cifar10_data)
    end_time  = time.time()

    print("Operations duration : ",end_time - start_time, "s")


