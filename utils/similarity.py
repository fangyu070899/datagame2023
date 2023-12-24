import numpy as np
import hashlib
import json
import os

from multiprocessing import Pool

class Similarity:
    def __init__(self) -> None:
        self.cpus = os.cpu_count() 

    def jaccard_similarity(self, arr1, arr2):
        set1 = set(arr1)
        set2 = set(arr2)
        intersection_size = len(set1.intersection(set2))
        union_size = len(set1.union(set2))
        similarity = intersection_size / union_size if union_size != 0 else 0
        return similarity
    
    def hash_to_float(self, hash_str):
        hash_object = hashlib.sha256(hash_str.encode())
        hash_hex = hash_object.hexdigest()
        hash_int = int(hash_hex, 16)
        hash_float = float(hash_int)
        return hash_float
    
    def cosine_similarity(self, args):
        test_source, training_source = args
        
        arr1 = test_source[1]
        arr2 = training_source[1]

        unique_elements_combined = np.unique(np.concatenate((arr1, arr2)))

        vector_arr1 = np.array([1 if element in arr1 else 0 for element in unique_elements_combined])
        vector_arr2 = np.array([1 if element in arr2 else 0 for element in unique_elements_combined])

        cosine_sim = np.dot(vector_arr1, vector_arr2) / (np.linalg.norm(vector_arr1) * np.linalg.norm(vector_arr2))

        return (cosine_sim,training_source[0])
    
    def parallel_similarity_calculation(self, test_source, training_source):    
        test_key = int(test_source[0])
        valid_keys = set()

        valid_keys = set(map(str, range(test_key - 10000, test_key + 10000))) & set(training_source.keys())

        args_list = [(test_source, item) for item in training_source.items() if item[0] in valid_keys]

        # args_list = [(test_source, item) for item in training_source.items()]

        with Pool(8) as pool:
            similarities  = pool.map(self.cosine_similarity, args_list)
        
        max_sim = max(similarities, key=lambda x: x[0])
        if max_sim[0] < 0.6:
            return False
        
        result = {'max_sim':max_sim[0], 'max_key':max_sim[1]}
        return result
    
    def get_similarity(self, test_source, training_source):
        result = {}
        count =0 
        for item in test_source.items():
            if count >= 50000: break
            print(item[0])
            sim =  self.parallel_similarity_calculation(item, training_source)
            if sim != False:            
                with open('data/json/similarity.json', 'r', encoding='utf-8') as file:
                    result = json.load(file)
                result[item[0]] = sim
                with open('data/json/similarity.json', 'w', encoding='utf-8') as file:
                    json.dump(result, file, indent=2)
            count+=1

        