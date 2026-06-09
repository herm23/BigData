import math
import random
import sys
import time
from pyspark import SparkConf, SparkContext

#NOTE: Fixed Seed
random.seed(42)

# Function to calculate the Euclidean distance between two points represented as tuples (x, y)
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Function to calculate the distance from a point to a set of centers
def dist_to_set(point, centers):
    if not centers:
        return float('inf')
    return min(euclidean_distance(point, c[0]) for c in centers)

# FairFFT Function
def FairFFT(U, kA, kB):

    """
    U: Lists of tuple ((x1, x2, ...), group) where group is 'A' or 'B'
    kA: Number of centers required for group 'A'
    kB: Number of centers required for group 'B'
    """

    S = []           # selected centers
    countA = 0       # selection count for group A
    countB = 0       # selection count for group B 

    # First center selection
    first_idx = random.randint(0, len(U) - 1)
    first = U[first_idx]
    
    S.append(first)
    if first[1] == 'A':
        countA += 1
    else:
        countB += 1

    for _ in range(1, kA + kB):
        
        # Determine which group to consider based on the current counts and budgets
        if countA >= kA:
            # Budget A full: consider only points from group B
            candidates = [(p, g) for (p, g) in U if g == 'B']
        elif countB >= kB:
            # Budget B full: consider only points from group A
            candidates = [(p, g) for (p, g) in U if g == 'A']
        else:
            # Both budgets available: consider all points
            candidates = U

        # Select the point farthest from the current set of centers
        farthest = max(candidates, key=lambda item: dist_to_set(item[0], S))
        S.append(farthest)

        if farthest[1] == 'A':
            countA += 1
        else:
            countB += 1

    return S


def run_local_fair_fft(partition, kA, kB):

    points = list(partition)
    if not points:
        return []

    local_kA = min(kA, sum(1 for (_, g) in points if g == 'A'))
    local_kB = min(kB, sum(1 for (_, g) in points if g == 'B'))
    if local_kA == 0 and local_kB == 0:
        return []
    return FairFFT(points, local_kA, local_kB)


def MRFairFFT(U, kA, kB):
    """
    U: RDD di tuple ((x1, x2, ...), group)
    kA: Budget group A
    kB: Budget group B
    t: Parameter for coreset size
    """
    
    #Round 1
    coreset_rdd = U.mapPartitions(lambda part: run_local_fair_fft(part, kA, kB))

    #Round 2
    coreset = coreset_rdd.collect()
    return FairFFT(coreset, kA, kB)

# Reads the set 𝑈 of input points into an RDD -called inputPoints-, subdivided into 𝐿 partitions.
def parse_line(line):
    parts = line.strip().split(',')
    point = tuple(float(x) for x in parts[:-1])
    group = parts[-1]
    return (point, group)

def main():
    # Command line arguments
    if len(sys.argv) != 5:
        print("Usage: python G23HW1.py <file_path> <kA> <kB> <L>")
        sys.exit(1)

    file_path = sys.argv[1]
    kA = int(sys.argv[2])
    kB = int(sys.argv[3])
    L = int(sys.argv[4])
    
    #Prints the command-line arguments.
    print(f"File path = {file_path}, KA = {kA}, KB = {kB}, L = {L}")

    conf = SparkConf().setAppName("G23HW1")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    #save in cache since in this way in the calculation time it does not include the time to load the dataset
    inputPoints = sc.textFile(file_path, minPartitions=L).map(parse_line).repartition(L).cache()
    inputPoints.count() 

    # Prints: 𝑁 =|𝑈|,𝑁𝐴 =|𝑈𝐴|  and 𝑁𝐵 =|𝑈𝐵|.
    counts = inputPoints.aggregate(
        (0, 0, 0),
        lambda acc, x: (acc[0]+1, acc[1]+(1 if x[1]=='A' else 0), acc[2]+(1 if x[1]=='B' else 0)),
        lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    )
    N, NA, NB = counts
    print(f"N = {N}, NA = {NA}, NB = {NB}")

    # Runs MRFairFFT to computes a solution 𝑆 to the fair k-center problem for the instance (𝑈,𝑘𝐴,𝑘𝐵).
    start = time.time()
    S = MRFairFFT(inputPoints, kA, kB)
    elapsed = int((time.time() - start) * 1000)

    # Prints the centers of 𝑆 together with their group label (on center per line) and the value of the objective function max𝑥∈𝑈⁡d⁢i⁢s⁢t⁢(𝑥,𝑆)
    for (point, group) in S:
        coords = ','.join(str(c) for c in point)
        print(f"Center = [{coords}] Label = {group}")


    def min_dist_to_S(point_group):
        p, _ = point_group
        return min(euclidean_distance(p, c) for (c, _) in S)

    objective = inputPoints.map(min_dist_to_S).max()
    print(f"Objective function = {objective}")

    #Print the time required by the execution of MRFairFFT in ms
    print(f"Running time of MRFairFFT = {elapsed} ms")

    sc.stop()

if __name__ == "__main__":
    main()