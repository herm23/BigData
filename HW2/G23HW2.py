import sys
import math
import random
import threading
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.streaming import StreamingContext

# Prime value
P = 8191

# function to process each batch of the stream
def process_batch(time, batch):
    global streamLength, frequencies, sticky_sample, CM_table, cm_set

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

        # True frequency
        frequencies[x] = frequencies.get(x, 0) + 1

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

        # F_CM: add x the first time its Count-Min estimated frequency reaches phi * n
        if x not in cm_set:
            estimate = min(CM_table[i][((hash_params[i][0] * x + hash_params[i][1]) % P) % w] for i in range(d))
            if estimate >= true_freq_threshold:
                cm_set.add(x)

    if streamLength[0] >= n:
        stopping_condition.set()


if __name__ == '__main__':
    assert len(sys.argv) == 8, "Warning: n phi epsilon delta d w portExp"

    # Input parsing
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

    # Data structure initialization
    streamLength = [0]
    frequencies = {}
    sticky_sample = {}

    # Sticky Sampling sampling rate p = r/n, with r = ln(1/(delta*phi)) / epsilon
    r = math.log(1 / (delta * phi)) / epsilon
    p_sample = r / n

    # True-frequency threshold phi*n, also used to build F_CM during the stream
    true_freq_threshold = phi * n

    # Count-Min sketch: d x w table and d hash functions h(x) = ((a*x+b) mod P) mod w
    CM_table = [[0] * w for _ in range(d)]
    hash_params = [(random.randint(1, P - 1), random.randint(0, P - 1)) for _ in range(d)]
    cm_set = set()

    # Stram processing with Spark Streaming
    conf = SparkConf().setMaster("local[*]").setAppName("G23HW2")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.1)  #  0.1 sec = 100 ms
    ssc.sparkContext.setLogLevel("ERROR")

    stopping_condition = threading.Event()

    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)


    # Final results computation
    true_frequent_items = sorted(x for x in frequencies if frequencies[x] >= true_freq_threshold)

    # F_SS: items in the Sticky Sampling reservoir whose estimated frequency is at least (phi - epsilon) * n
    ss_threshold = (phi - epsilon) * n
    F_SS = sorted(x for x in sticky_sample if sticky_sample[x] >= ss_threshold)

    # F_CM: items added during the stream, the first time their estimate reached phi * n
    F_CM = sorted(cm_set)

    # Final printing of results
    print("TRUE FREQUENT ITEMS")
    for x in true_frequent_items:
        print(f"Item = {x} True Freq = {frequencies[x]}")

    print("STICKY SAMPLING")
    print(f"Size of dictionary = {len(sticky_sample)}")
    for x in F_SS:
        print(f"Item = {x} True Freq = {frequencies[x]}")

    print("COUNT-MIN SKETCH")
    print(f"Size of F_CM = {len(F_CM)}")
    for x in F_CM:
        print(f"Item = {x} True Freq = {frequencies[x]}")
