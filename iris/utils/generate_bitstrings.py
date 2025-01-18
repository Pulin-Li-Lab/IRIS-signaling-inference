import time
import random

def hamming_distance(
        string1: str, 
        string2: str
    ) -> int:
    '''
    Helper function that finds Hamming distance between two strings (sums up 
    number of characters that differ). Both strings must be same length.

    Args:
        string1: first string 
        string2: second string

    Returns:
        dist: integer Hamming distance
    '''
    dist = 0
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            dist += 1
    return dist

def generate_bitflip(
        n: int, 
        references: list[str]
    ) -> list:
    '''
    Helper function that flips a string of 0's and 1's one-by-one and returns list of 
    generated sequences. For example, given '101', this function will return 
    ['001', '111', '100']. Supports multiple reference strings. 

    Args:
        n: integer number of sequences to generate
        references: list of strings of reference sequences

    Returns:
        sequences: list of strings of generated sequences
    '''
    sequences = []
    l = len(references)
    count = 0

    for i in range(l):
        combos = []
        reference = references[i]
        m = len(reference)
        for j in range(m):
            flipped = reference[:j] + str(1-int(reference[j])) + reference[j+1:]
            combos.append(flipped)
            count += 1
            # generated enough from bitflip
            if count == n:
                sequences += combos
                return sequences
            
        sequences += combos
    return sequences

def generate_random(
        n: int, 
        reference: str, 
        min_distance: int = None, 
        sampling: bool = False
    ) -> set:
    '''
    Generates n bitstrings that are each min_distance away from all other 
    generated bitstrings, using reference as the starting point. If min_distance 
    is not provided, samples increasing distances to find minimum Hamming 
    distance able to generate n bitstrings. Uses timeout of 30s while sampling.

    Args:
        n: integer number of bitstrings to be generated
        reference: string, reference bit sequence (ex. '0110')
        min_distance: integer minimum distance apart each bitstring should be 
        sampling: boolean of whether to sample minimum distance or not 

    Returns:
        combos: set of generated bitstrings, including reference 
    '''
    length = len(reference)

    if not min_distance:
        min_distance = 1
        sampling = True

    valid = False
    i = 0
    combos = {reference}
    num_to_permute = min_distance

    start = time.time()
    while i < n:
        indices = random.sample(range(0,length), num_to_permute)
        perturbed = ''.join([str(1-int(reference[i])) if i in indices else reference[i] for i in range(length)])
        if perturbed not in combos:
            for elem in combos:
                if hamming_distance(elem, perturbed) < min_distance:
                    valid = False
                    break
                valid = True
            if valid:
                combos.add(perturbed)
                i += 1
        curr = time.time()

        while curr - start > 30:
            if num_to_permute < length:
                num_to_permute += 1
                start = time.time()
                break
            if sampling:
                # this distance timed out, so use results from previous one
                return generate_random(n, reference, min_distance - 1, False)
            else:
                # process timed out with a defined minimum distance
                return combos
        
    if sampling and min_distance < length:
        # keep looking only if distance away isn't maxed out 
        min_distance += 1
        return generate_random(n, reference, min_distance, True)

    return combos