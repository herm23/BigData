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

        # F_CM: add x the FIRST TIME its current CM estimate becomes >= phi*n,
        # checked at insertion time (as the stream is processed). Computing this
        # at the end of the stream would inflate F_CM with false positives, since
        # the CM counters are monotonically increasing and collisions accumulate.
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

    # Number of runs over which we average the results to fill in the form tables
    NUM_RUNS = 3

    # Sticky Sampling sampling rate p = r/n, with r = ln(1/(delta*phi)) / epsilon
    r = math.log(1 / (delta * phi)) / epsilon
    p_sample = r / n

    # Frequency-class thresholds, used to classify the items returned by each
    # algorithm: frequent if true freq >= phi*n, almost frequent if
    # (phi-epsilon)*n <= true freq < phi*n, rare otherwise.
    true_freq_threshold = phi * n
    almost_threshold = (phi - epsilon) * n

    # Single Spark context reused across the runs
    conf = SparkConf().setMaster("local[*]").setAppName("G23HW2")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    def classify(items):
        """Split the returned items into (frequent, almost frequent, rare) counts."""
        freq = almost = rare = 0
        for x in items:
            tf = frequencies[x]
            if tf >= true_freq_threshold:
                freq += 1
            elif tf >= almost_threshold:
                almost += 1
            else:
                rare += 1
        return freq, almost, rare

    # Accumulators (one value per run) for the 8 form columns
    ss_freq, ss_almost, ss_rare, ss_dict = [], [], [], []
    cm_freq, cm_almost, cm_rare, cm_total = [], [], [], []

    for run in range(NUM_RUNS):
        # Per-run (re)initialization of the data structures
        streamLength = [0]
        frequencies = {}
        sticky_sample = {}
        CM_table = [[0] * w for _ in range(d)]
        cm_set = set()
        hash_params = [(random.randint(1, P - 1), random.randint(0, P - 1)) for _ in range(d)]

        # Stream processing with Spark Streaming
        ssc = StreamingContext(sc, 0.1)  #  0.1 sec = 100 ms
        stopping_condition = threading.Event()

        stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
        stream.foreachRDD(lambda time, batch: process_batch(time, batch))

        ssc.start()
        stopping_condition.wait()
        ssc.stop(False, False)

        # F_SS: items in the Sticky Sampling reservoir whose estimated frequency is at least (phi - epsilon) * n
        ss_threshold = (phi - epsilon) * n
        F_SS = sorted(x for x in sticky_sample if sticky_sample[x] >= ss_threshold)

        # F_CM: items added during stream processing, the first time their
        # Count-Min estimate reached phi * n (built incrementally in process_batch).
        F_CM = sorted(cm_set)

        # Per-run counts for the 8 form columns
        f_ss, a_ss, r_ss = classify(F_SS)
        f_cm, a_cm, r_cm = classify(F_CM)
        ss_freq.append(f_ss)
        ss_almost.append(a_ss)
        ss_rare.append(r_ss)
        ss_dict.append(len(sticky_sample))
        cm_freq.append(f_cm)
        cm_almost.append(a_cm)
        cm_rare.append(r_cm)
        cm_total.append(len(F_CM))

        # Print the 8 columns for this run (4 Sticky Sampling + 4 Count-Min)
        print(f"RUN {run + 1}/{NUM_RUNS}")
        print(f"Number of frequent items returned by SS = {f_ss}")
        print(f"Number of almost frequent items returned by SS = {a_ss}")
        print(f"Number of rare items returned by SS = {r_ss}")
        print(f"Number of elements stored in the dictionary used by SS = {len(sticky_sample)}")
        print(f"Number of frequent items returned by CM = {f_cm}")
        print(f"Number of almost frequent items returned by CM = {a_cm}")
        print(f"Number of rare items returned by CM = {r_cm}")
        print(f"Total number of items returned by CM = {len(F_CM)}")

    sc.stop()

    def avg(values):
        return sum(values) / len(values)

    # Averaged values over the NUM_RUNS runs (to be copied into the form tables)
    print(f"AVERAGE OVER {NUM_RUNS} RUNS")
    print(f"Number of frequent items returned by SS = {avg(ss_freq)}")
    print(f"Number of almost frequent items returned by SS = {avg(ss_almost)}")
    print(f"Number of rare items returned by SS = {avg(ss_rare)}")
    print(f"Number of elements stored in the dictionary used by SS = {avg(ss_dict)}")
    print(f"Number of frequent items returned by CM = {avg(cm_freq)}")
    print(f"Number of almost frequent items returned by CM = {avg(cm_almost)}")
    print(f"Number of rare items returned by CM = {avg(cm_rare)}")
    print(f"Total number of items returned by CM = {avg(cm_total)}")