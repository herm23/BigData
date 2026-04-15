import math
import sys
import os
import matplotlib.pyplot as plt
from enum import Enum
from pyspark import SparkContext, SparkConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from G23HW1 import FairFFT, MRFairFFT  

class Algorithm(Enum):
    SEQUENTIAL = 1
    MAPREDUCE = 2

def load_dataset(filepath):
    U = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            point = tuple(float(x) for x in parts[:-1])
            group = parts[-1]
            U.append((point, group))
    return U

def objective(U, S):
    def dist(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    return max(min(dist(p, c) for (c, _) in S) for (p, _) in U)


CHOICE = Algorithm.MAPREDUCE
filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'uber_small.csv')
kA, kB = 5, 2

conf = SparkConf().setAppName("FairKCenterTest").setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)

U_list = load_dataset(filepath)
print(f"Dataset caricato: N={len(U_list)}")

if CHOICE == Algorithm.SEQUENTIAL:
    print("Esecuzione Algoritmo Sequenziale...")
    S = FairFFT(U_list, kA, kB)
    
elif CHOICE == Algorithm.MAPREDUCE:
    print("Esecuzione Algoritmo MapReduce (Spark)...")
    U_rdd = sc.parallelize(U_list)
    S = MRFairFFT(U_rdd, kA, kB)

print(f"\nCentri selezionati:")
for point, group in S:
    print(f"  {group}: {point}")

print(f"\nObjective function: {objective(U_list, S):.4f}")

sc.stop()