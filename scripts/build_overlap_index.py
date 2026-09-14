'''
To recover the Rule 0, we need to compare all the algorithm pairs in the population.
With 14236 algorithms from DeepMind, there are 101,324,730 algorithm pairs. This is 
highly impractical for a notebook even when not done naively. The fix is 
build_overlap_index.py which is a pre-requisuite for the rule0_analsis.ipynb. 

This script does the following:
- Inspects the shapes of the algorithms
- Canonicalizes the algorithms
    - U_r, V_r, W_r = (-U_r), (-V_r), (W_r) = (-U_r), (V_r), (-W_r) = (U_r), (-V_r), (-W_r)
- Counts the terms pairs have in common

The data is saved in ../data_processed for use in rule0_analysis.ipynb.
'''

# 1. imports
import numpy as np
from itertools import combinations
import csv
from pathlib import Path

# 2. load data
output_dir = Path('../data_processed')
data = np.load('../alphatensor_14236_factorizations.npz')
factorizations = data['factorizations']

# 3. inspect shapes
print(factorizations.shape)
algorithm = factorizations[0]
print(algorithm.shape)
term = algorithm[0]
print(term.shape)
print(term)

# 4. canonicalize one term
def canon_term(term):
    u = term[0]
    v = term[1]
    w = term[2]

    su = None
    sv = None

    for coefficient in u:
        if coefficient != 0:
            if coefficient < 0:
                su = -1
            else:
                su = 1
            break

    for coefficient in v:
        if coefficient != 0:
            if coefficient < 0:
                sv = -1
            else:
                sv = 1
            break

    if su is None or sv is None:
        raise ValueError('Invalid rank-one term: U or V is zero.')

    sw = su * sv

    u_canonical = su * u
    v_canonical = sv * v
    w_canonical = sw * w

    return u_canonical, v_canonical, w_canonical

canonicalized_term = canon_term(term)
print(canonicalized_term)

original_tensor = np.einsum(
    'i,j,k->ijk',
    term[0],
    term[1],
    term[2]
)

canonical_tensor = np.einsum(
    'i,j,k->ijk',
    canonicalized_term[0],
    canonicalized_term[1],
    canonicalized_term[2]
)

# sanity check
print(np.array_equal(original_tensor, canonical_tensor))

# 5. canonicalize one algorithm
def canon_algorithm(algorithm):
    canonical_algorithm = []

    for term in algorithm:
        canonicalized_term = canon_term(term)
        canonical_algorithm.append(canonicalized_term)

    return np.array(canonical_algorithm)

canonical_algorithm = canon_algorithm(algorithm)
print(canonical_algorithm.shape)

def algorithm_tensor(algorithm):
    total = np.zeros((16, 16, 16), dtype=int)

    for term in algorithm:
        u = term[0]
        v = term[1]
        w = term[2]

        total += np.einsum('i,j,k->ijk', u, v, w)

    return total

original_algorithm_tensor = algorithm_tensor(algorithm)
canonical_algorithm_tensor = algorithm_tensor(canonical_algorithm)

# sanity check
print(np.array_equal(
    original_algorithm_tensor,
    canonical_algorithm_tensor
))

# 6. canonicalize whole population
canonical_population = []

for algorithm in factorizations:
    canonical_algorithm = canon_algorithm(algorithm)
    canonical_population.append(canonical_algorithm)

canonical_population = np.array(canonical_population)

print(canonical_population.shape)

# sanity check
for i in range(3):
    original = algorithm_tensor(factorizations[i])
    canonical = algorithm_tensor(canonical_population[i])

    print(i, np.array_equal(original, canonical))

# 7. count pair overlaps
# build inverted index
term_to_algorithms = {}

for algorithm_id, algorithm in enumerate(canonical_population):

    for term in algorithm:

        term_key = tuple(term.flatten())

        if term_key not in term_to_algorithms:
            term_to_algorithms[term_key] = []

        term_to_algorithms[term_key].append(algorithm_id)

print(f'Distinct canonical terms: {len(term_to_algorithms)}')


# count how many algorithm pairs exist
n_algorithms = len(canonical_population)
n_pairs = n_algorithms * (n_algorithms - 1) // 2

print(f'Number of algorithms: {n_algorithms}')
print(f'Number of possible pairs: {n_pairs}')


# count how many pair updates the inverted index requires
total_pair_updates = 0

for algorithm_ids in term_to_algorithms.values():
    k = len(algorithm_ids)
    total_pair_updates += k * (k - 1) // 2

print(f'Total pair updates: {total_pair_updates}')

pair_counts = np.zeros(n_pairs, dtype=np.uint8)


# map an algorithm pair (i, j), where i < j
def pair_index(i, j, n):
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


# sanity check 
last_pair_row_0 = pair_index(0, n_algorithms - 1, n_algorithms)
first_pair_row_1 = pair_index(1, 2, n_algorithms)

print(f'Last index in row 0: {last_pair_row_0}')
print(f'First index in row 1: {first_pair_row_1}')
print(f'Difference: {first_pair_row_1 - last_pair_row_0}')


# count shared terms
for algorithm_ids in term_to_algorithms.values():

    ids = np.array(algorithm_ids, dtype=np.int64)

    for position in range(len(ids) - 1):

        i = ids[position]
        js = ids[position + 1:]

        indices = (
            i * (2 * n_algorithms - i - 1) // 2
            + (js - i - 1)
        )

        pair_counts[indices] += 1


# inspect the largest overlap found
print(f'Maximum overlap: {pair_counts.max()}')

# 8. save results
def index_to_pair(index, n):
    i = 0
    row_length = n - 1

    while index >= row_length:
        index -= row_length
        i += 1
        row_length -= 1

    j = i + 1 + index

    return i, j

# find the overlap levels
overlap_levels = np.unique(pair_counts)
overlap_levels = overlap_levels[overlap_levels > 0]
overlap_levels = np.sort(overlap_levels)[::-1]

print('Top overlap levels:')
print(overlap_levels[:10])

# recover pairs from the highest few overlap levels
top_pairs = []

for overlap in overlap_levels[:5]:

    indices = np.where(pair_counts == overlap)[0]

    for index in indices:
        i, j = index_to_pair(int(index), n_algorithms)

        top_pairs.append(
            (i, j, int(overlap))
        )

print('\nTop pairs:')
for i, j, overlap in top_pairs:
    print(f'Algorithms {i} and {j}: {overlap} shared terms')

# save canonical population
np.savez_compressed(
    output_dir / 'canonical_population.npz',
    canonical_population=canonical_population
)

print('Saved canonical_population.npz')

# save top overlap pairs
top_pairs_path = output_dir / 'top_pairs.csv'

with open(top_pairs_path, 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow([
        'algorithm_i',
        'algorithm_j',
        'shared_terms'
    ])

    for i, j, overlap in top_pairs:
        writer.writerow([
            i,
            j,
            overlap
        ])

print('Saved top_pairs.csv')

# save all pairs
all_pairs_path = output_dir / 'all_nonzero_pair_overlaps.csv'

with open(all_pairs_path, 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow([
        'algorithm_i',
        'algorithm_j',
        'shared_terms'
    ])

    index = 0

    for i in range(n_algorithms - 1):

        for j in range(i + 1, n_algorithms):

            overlap = int(pair_counts[index])

            if overlap > 0:
                writer.writerow([
                    i,
                    j,
                    overlap
                ])

            index += 1

print('Saved all_nonzero_pair_overlaps.csv')

# save the pair count array
np.savez_compressed(
    output_dir / 'pair_overlap_counts.npz',
    pair_counts=pair_counts,
    n_algorithms=n_algorithms
)

print('Saved pair_overlap_counts.npz')