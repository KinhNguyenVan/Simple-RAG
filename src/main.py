from model import MyModel
import os
import time

def main():
    with open('contents.txt', 'r', encoding='utf-8') as file:  
        contents = file.readlines()  

    model = MyModel(contents=contents)
    start = time.time()
    query = "what is policy?"

    print(f"Answer: {model.make_response(query)}")
    end = time.time()
    print(f"Runining Time: {end - start} seconds")

    start = time.time()
    
    query = "when is the latest update?"

    print(f"Answer: {model.make_response(query)}")
    end = time.time()
    print(f"Runining Time: {end - start} seconds")