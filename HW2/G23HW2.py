import sys
import math
import random
import threading
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.streaming import StreamingContext

# NOTE: Fixed seed for reproducibility
random.seed(42)

# Prime used in the 2-universal hash family h(x) = ((a*x + b) mod P) mod C
P = 8191


# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    global streamLength, histogram, sticky_sample, CM_table

    # If we already have enough points (>= n), skip this batch.
    if streamLength[0] >= n:
        return

    batch_items = batch.collect()

    # Ignore items beyond the n-th one
    remaining = n - streamLength[0]
    if len(batch_items) > remaining:
        batch_items = batch_items[:remaining]

    streamLength[0] += len(batch_items)

    for s in batch_items:
        x = int(s)

        # True frequency (histogram of the first n items)
        histogram[x] = histogram.get(x, 0) + 1

        # Sticky Sampling
        if x in sticky_sample:
            sticky_sample[x] += 1
        elif random.random() <= p_sample:
            sticky_sample[x] = 1

        # Count-Min sketch
        for i in range(d):
            a, b = hash_params[i]
            j = ((a * x + b) % P) % w
            CM_table[i][j] += 1

    if streamLength[0] >= n:
        stopping_condition.set()


if __name__ == '__main__':
    assert len(sys.argv) == 8, "USAGE: n phi epsilon delta d w portExp"

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    d = int(sys.argv[5])
    w = int(sys.argv[6])
    portExp = int(sys.argv[7])

    print("INPUT PARAMETERS")
    print(f"n = {n}")
    print(f"phi = {phi}")
    print(f"epsilon = {epsilon}")
    print(f"delta = {delta}")
    print(f"d = {d}")
    print(f"w = {w}")
    print(f"port = {portExp}")

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    streamLength = [0]
    histogram = {}       # true frequencies of the first n items
    sticky_sample = {}   # Sticky Sampling reservoir: item -> counter

    # Sticky Sampling sampling rate p = r/n, with r = ln(1/(delta*phi)) / epsilon
    r = math.log(1 / (delta * phi)) / epsilon
    p_sample = r / n

    # Count-Min sketch: d x w table and d hash functions h(x) = ((a*x+b) mod P) mod w
    CM_table = [[0] * w for _ in range(d)]
    hash_params = [(random.randint(1, P - 1), random.randint(0, P - 1)) for _ in range(d)]

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # SPARK STREAMING SETUP
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    conf = SparkConf().setMaster("local[*]").setAppName("G23HW2")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.1)  # Batch duration of 0.1 sec = 100 ms
    ssc.sparkContext.setLogLevel("ERROR")

    stopping_condition = threading.Event()

    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # COMPUTING THE FINAL RESULTS
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    true_freq_threshold = phi * n

    # True frequent items
    true_frequent_items = sorted(x for x in histogram if histogram[x] >= true_freq_threshold)

    # F_SS: items in the Sticky Sampling reservoir whose estimated frequency
    # is at least (phi - epsilon) * n
    ss_threshold = (phi - epsilon) * n
    F_SS = sorted(x for x in sticky_sample if sticky_sample[x] >= ss_threshold)

    # F_CM: items whose Count-Min estimated frequency is at least phi * n
    F_CM = []
    for x in histogram:
        estimate = min(CM_table[i][((hash_params[i][0] * x + hash_params[i][1]) % P) % w] for i in range(d))
        if estimate >= true_freq_threshold:
            F_CM.append(x)
    F_CM = sorted(F_CM)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # PRINTING THE RESULTS
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    # Kept commented out: re-enable it for the final code submission.
    print("TRUE FREQUENT ITEMS")
    for x in true_frequent_items:
        print(f"Item = {x} True Freq = {histogram[x]}")
    
    print("STICKY SAMPLING")
    print(f"Size of dictionary = {len(sticky_sample)}")
    for x in F_SS:
        print(f"Item = {x} True Freq = {histogram[x]}")
    
    print("COUNT-MIN SKETCH")
    print(f"Size of F_CM = {len(F_CM)}")
    for x in F_CM:
        print(f"Item = {x} True Freq = {histogram[x]}")