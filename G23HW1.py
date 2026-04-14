import math

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

    # Fisr center selection
    first = U[0]
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