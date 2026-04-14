import math
import sys
import os
import matplotlib.pyplot as plt

#SOLO PER FARE I TEST DI CORRETTO FUNZIONAMENTO, PALESEMENTE COPIATA DA CLAUDE ;)

# Root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from G23HW1 import FairFFT  

def load_dataset(filepath):
    U = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            point = tuple(float(x) for x in parts[:-1])
            group = parts[-1]
            U.append((point, group))
    return U

def objective(U, S):
    def dist(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    return max(min(dist(p, c) for (c, _) in S) for (p, _) in U)

def plot_centers(U, S):
    # Separiamo i punti di U per gruppo per colorarli diversamente
    points_A = [p for p, g in U if g == 'A']
    points_B = [p for p, g in U if g == 'B']
    
    # Separiamo i centri di S per gruppo
    centers_A = [p for p, g in S if g == 'A']
    centers_B = [p for p, g in S if g == 'B']

    plt.figure(figsize=(10, 7))

    # Disegniamo tutti i punti del dataset (piccoli e semitrasparenti)
    if points_A:
        plt.scatter([p[0] for p in points_A], [p[1] for p in points_A], 
                    c='lightblue', label='Dataset A', alpha=0.5, s=20)
    if points_B:
        plt.scatter([p[0] for p in points_B], [p[1] for p in points_B], 
                    c='lightcoral', label='Dataset B', alpha=0.5, s=20)

    # Disegniamo i centri (più grandi e con bordo)
    if centers_A:
        plt.scatter([p[0] for p in centers_A], [p[1] for p in centers_A], 
                    c='blue', label='Centri A', edgecolors='black', s=100, marker='o')
    if centers_B:
        plt.scatter([p[0] for p in centers_B], [p[1] for p in centers_B], 
                    c='red', label='Centri B', edgecolors='black', s=100, marker='o')

    plt.title(f"Fair k-center Clustering (kA={len(centers_A)}, kB={len(centers_B)})")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'uber_small.csv')
U = load_dataset(filepath)

# Dataset info
NA = sum(1 for (_, g) in U if g == 'A')
NB = sum(1 for (_, g) in U if g == 'B')
print(f"N={len(U)}, NA={NA}, NB={NB}")

# Test con kA=3, kB=2
kA, kB = 5, 2
S = FairFFT(U, kA, kB)

print(f"\nSelected centers (kA={kA}, kB={kB}):")
for point, group in S:
    print(f"  {point} -> group {group}")

countA = sum(1 for (_, g) in S if g == 'A')
countB = sum(1 for (_, g) in S if g == 'B')
plot_centers(U, S)

print(f"\Centers A: {countA}/{kA}, Centers B: {countB}/{kB}")
print(f"Objective (max min-dist): {objective(U, S):.4f}")