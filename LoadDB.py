"""
Acest fisier are scopul de a citi datele din setul de date Endomondo si executa o prima parte de preprocesare,
anume eliminarea utilizatorilor care au un numar prea mic de antrenamente.
"""

import json
import pickle
from collections import Counter

path = "D:\\OneDrive\\Education\\College\\Licenta\\RecSys - Data Sets\\"
file_path = str(path + "old_endomondoHR_proper.json")
open_file = open(file_path,'r')

dataset = []

def readlines(file = open_file,size = 10000):
    i=0
    for lines in range(size):
        current_line = open_file.readline()
        if current_line == '': break
        current_line = current_line.replace("'", '"')
        current_line = json.loads(current_line)
        dataset.append(current_line)
        i += 1

    return dataset

def trim(dataset, minWorkouts = 20, maxWorkouts = 50):
    a = []
    for x in dataset:
        a.append(x['userId'])
    count = Counter(a)
    dataset2 = [item for item in dataset if count[item['userId']] >= minWorkouts]
    dataset = dataset2

    # pastram doar primele maxWorkouts antrenamente
    dataset2 = []
    for key in count:
        keyCount = 0
        for element in dataset:
            if key == element['userId']:
                keyCount += 1
                if keyCount <= maxWorkouts:
                    dataset2.append(element)

    a = []
    for x in dataset2:
        a.append(x['userId'])
    count = Counter(a)

    return dataset2



def main():
    dataset = readlines(open_file)
    dataset = trim(dataset,20,50)
    with open('dataset_mic.pkl','wb') as pickle_file:
        pickle.dump(dataset,pickle_file)

if __name__=="__main__": 
    main() 


