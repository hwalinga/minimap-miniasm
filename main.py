#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimap and miniasm.
"""

import heapq
import sys
from collections import defaultdict, deque
from functools import partial
from itertools import groupby, islice, repeat, tee
from operator import itemgetter
from typing import IO, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

###########
# MINIMAP #
###########

complement = str.maketrans("ACGT", "TGCA")


# Type aliases minimap

# A set with tuples representing the hash value, the position, and the strand.
Minimizers = Set[Tuple[int, int, bool]]

# A function that takes a sequence and returns the minimizers.
Minimizer_Sketch = Callable[[str], Minimizers]

# A function that takes a sequence and returns a hash integer.
Seq_Hash = Callable[[str], int]

# A dictionary that maps the hash to a list of tuples with
# target sequence id, position, and strand.
Target_Index = Dict[int, List[Tuple[int, int, bool]]]

# A list with a tuple with the name, and the lenght of the sequences.
Seq_Info = List[Tuple[str, int]]


def reverse_complement(s: str) -> str:
    """
    Create the reverse complement of sequence
    """
    return s.translate(complement)[::-1]


def sliding_windowing(s: str, k: int) -> Iterator[str]:
    """
    Creates a sliding window of k-mers over a sequence (s).

    Parameters
    ----------
    s : str
        The sequence.
    k : int
        The size of the k-mer.

    Returns
    -------
    it : Iterator[str]
        The k-mers yield from this iterator.
    """
    if len(s) < k:
        return iter(())
    # To create this k-mer iterator, we create k letter iterators which
    # start all one letter further than the previous ones.
    # Then, get one letter from all these letter iterators and concatenate them.
    texts = []
    for ind, t in enumerate(tee(s, k)):
        for _ in repeat(None, ind):
            next(t)
        texts.append(t)
    return map("".join, zip(*texts))


def hash_seq(s: str) -> int:
    """
    Hash a sequence (or k-mer).

    Using Phi(A)=0, Phi(C)=1, Phi(G)=2, and Phi(T)=3

    Calculate with s = a_1 ... a_k:
    Phi(s) = Phi(a_1) x 4 ^ (k-1) + Phi(a_2) x 4 ^ (k-2) ... + Phi(a_k)

    Parameters
    ----------
    s : str
        s is the sequence (or k-mer) that needs to be hashed

    Returns
    -------
    x : int
        Returns the integer hashed
    """
    base_values = {"A": 0, "C": 1, "G": 2, "T": 3}
    k = len(s)
    return sum(base_values[b] * 4 ** (k - i) for i, b in enumerate(s))


def invertable_hash(x: int, p: int) -> int:
    """
    Invertible integer hash function.

    Invert a hash integer.

    Algorithm 2.

    A problem with :func:`hash_seq` is that the uninformative poly-A sequence
    will get the hash value of 0, which is the minimum. To prevent this,
    the hash function is altered using this function. Invertibility of the
    hash function is not a requirement, but prevent hash collisions.

    Parameters
    ----------
    x : int
        The hash integer.
    p : int
        The amount of bits of the hash.

    Returns
    -------
    x : int
        Return the new hash integer.
    """
    m = 2 ** p - 1
    x = (~x + (x << 21)) & m
    x = x ^ (x >> 24)
    x = (x + (x << 3) + (x << 8)) & m
    x = x ^ (x >> 14)
    x = (x + (x << 2) + (x << 4)) & m
    x = x ^ (x >> 28)
    x = (x + (x << 31)) & m
    return x


def compute_minimizers(s: str, w: int, k: int, hash_func: Seq_Hash) -> Minimizers:
    """
    Compute the (w, k)-minimizers.

    A minimizer is the smallest k-mer in a window of w k-mer.

    Algorithm 1.

    Parameters
    ----------
    s : str
        The input sequence. For the size of the sequence it should be
        |s| => w + k - 1
    w : int
        The amount of k-mers to use in a window in which to find the smallest k-mer.
    k : int
        The size of the k-mer.

    Returns
    -------
    M : Minimizers
        The minimizers are a set with a tuple with the numbers m, i, r
        Where m is the hash integer,
        i is the position
        r is the strand
    """
    kmers = sliding_windowing(s, k)
    w_window = deque(islice(kmers, w))
    M: Minimizers = set()

    i = 0  # We use 0-indexing.
    while True:

        # Find minimum k-mer
        m = sys.maxsize  # 2 ** 63 - 1 on 64 bit platforms.
        for j, kmer in enumerate(w_window):
            u, v = hash_func(kmer), hash_func(reverse_complement(kmer))
            if u != v:  # If reverse complement the same sequence, do not use.
                m = min(m, u, v)

        # Add minimizer
        for j, kmer in enumerate(w_window):
            u, v = hash_func(kmer), hash_func(reverse_complement(kmer))
            if u == m or v == m:
                M.add((m, i + j, v < u))

        # Move w window one k-mer further. (If not more k-mers, we are done.)
        new_kmer = next(kmers, None)
        if not new_kmer:
            break
        w_window.popleft()
        w_window.append(new_kmer)
        i += 1

    return M


def get_func_minimizer_sketch(w: int, k: int) -> Minimizer_Sketch:
    """
    A function to set all values to the :func:`compute_minimizers` function.

    Paramaters
    ----------
    w : int
        Amount of k-mers in window.
    k : int
        Length of the k-mer.

    Returns
    -------
    minimizer_sketch : Minimizer_Sketch
        A function that takes a sequence and returns the minimizers
    """
    return partial(
        compute_minimizers,
        w=w,
        k=k,
        hash_func=lambda s: invertable_hash(hash_seq(s), 2 * k),
    )


def index_targets(
    target_seqs: Iterable[Tuple[str, str]], minimizer_sketch: Minimizer_Sketch
) -> Tuple[Target_Index, Seq_Info]:
    """
    Algorithm 3.
    """
    H_list = []
    seq_info = []
    for t, (name, seq) in enumerate(target_seqs):

        seq_info.append((name, len(seq)))

        M = minimizer_sketch(seq)
        for h, i, r in M:
            H_list.append((h, (t, i, r)))

    H_list.sort(key=itemgetter(0))

    H_dict = {
        h: list(map(itemgetter(1), vals))
        for h, vals in groupby(H_list, key=itemgetter(0))
    }

    return H_dict, seq_info


def map_query(
    query_seq: str,
    query_name: str,
    target_seq_info: Seq_Info,
    H: Target_Index,
    minimizer_sketch: Minimizer_Sketch,
    epsilon: int,
    min_subset: int,
    min_overlap: int,
):
    """
    Algorithm 4.
    """
    M = minimizer_sketch(query_seq)
    A = []

    # Collect minimizers hit
    for h, i, r in M:
        for t, i_t, r_t in H[h]:
            if r == r_t:
                A.append((t, 0, i - i_t, i_t))
            else:
                A.append((t, 1, i + i_t, i_t))

    # Python automatically sorts tuples with radix sort, which we want here.
    A.sort()

    b = 0
    for e in range(len(A)):
        if (
            e + 1 == len(A)
            or A[e + 1][0] != A[e][0]
            or A[e + 1][1] != A[e][1]
            or A[e + 1][2] - A[e][2] >= e
        ):
            potential_overlap = A[b:e]
            b = e + 1

            begin, end = maximal_colinear_subset(map(itemgetter(3), potential_overlap))

            if end - begin < min_subset:
                continue

            res = overlap_hit(potential_overlap[begin:end])
            if res[10] > min_overlap:
                res[0] = query_name
                res[1] = len(query_seq)
                yield res


def overlap_hit(minimizer_hits: List[Tuple[int, int, int, int]]):
    """
    Return PAF tuple
    1	string	Query sequence name
    2	int	Query sequence length
    3	int	Query start coordinate (0-based)
    4	int	Query end coordinate (0-based)
    5	char	‘+’ if query/target on the same strand; ‘-’ if opposite
    6	string	Target sequence name
    7	int	Target sequence length
    9	int	Target end coordinate on the original strand
    10	int	Number of matching bases in the mapping
    11	int	Number bases, including gaps, in the mapping
    12	int	Mapping quality (0-255 with 255 for missing)
    """
    pass


def maximal_colinear_subset(li: Iterable[int]) -> Tuple[int, int]:
    """
    Find the longest increasing subsequence.

    Returns
    -------
    begin, end : Tuple[int, int]
        Return the first index and the last index of the longest increasing subsequence.
    """
    pass


def read_fastx(file: str, format: str) -> Iterator[Tuple[str, str]]:
    """
    Currently, this function just assumes for fasta that the sequences and the
    fasta header are on alternating lines.

    Parameters
    file : str
        Name of the input file.
    format : str
        The format ("fasta" or "fastq")

    Returns
    -------
    it : Iterator[Tuple[str, str]]
        Returns an iterator that yields tuples with the sequence name and the sequence.
    """
    if format == "fastq":
        group_size = 4
    elif format == "fasta":
        group_size = 2

    with open(file) as f:
        ind_line = enumerate(f)
        for ind, line in ind_line:
            if ind % group_size == 0:
                seq_name = line.strip()
                _, seq = next(ind_line)[1].strip()
                yield seq_name, seq


def minimap(target_seq_file_name, query_seq_file_name, output_file_name):
    """
    This is the minimap function
    """
    target_seqs = read_fastx(target_seq_file_name, format="fastq")
    query_seqs = read_fastx(query_seq_file_name, format="fastq")

    k = 6
    w = 6
    epsilon = 500
    min_subset = 4
    min_overlap = 100

    minimizer_sketch = get_func_minimizer_sketch(w, k)

    H, target_seq_info = index_targets(target_seqs, minimizer_sketch)

    with open(output_file_name, "w") if output_file_name else sys.stdout as out:
        for query_name, query_seq in query_seqs:
            for t in map_query(
                query_seq,
                query_name,
                target_seq_info,
                H,
                minimizer_sketch,
                epsilon,
                min_subset,
                min_overlap,
            ):
                print(*t, seq="\t", file=out)


###########
# MINIASM #
###########


# For PAF we will use fields 1-11. So not using the quality one, and optional additional
PAF = Tuple[str, int, int, int, str, str, int, int, int, int, int]

# A Mapping is described by two strings: The names of the query and the target,
# and one more string to describe if they match on the same strand or opposite
# (Just like in PAF.)
All_Mappings = Dict[Tuple[str, str, str], List[Tuple[int, int, int, int]]]
Mappings = Dict[Tuple[str, str, str], Tuple[int, int, int, int]]

# In the genome graph each strand is described with the name and a bool
# to indicate if True, it is the same strand, or False, it is the opposite.
# (This is equivalent to {True: '+', '-': False} compared with PAF.)
# The other two integers indicate the From start coord and the To end coord
# of the mapping.
Genome_Graph = Dict[Tuple[str, bool], Tuple[str, bool, int, int]]


def read_paf_file(paf_file: IO) -> Iterator[PAF]:
    """
    Just read the file and yields each line as a paf tuple.

    The PAF file is tab delimited file with the fields:

    1	string	Query sequence name
    2	int	Query sequence length
    3	int	Query start coordinate (0-based)
    4	int	Query end coordinate (0-based)
    5	char	‘+’ if query/target on the same strand; ‘-’ if opposite
    6	string	Target sequence name
    7	int	Target sequence length
    8	int	Target start coordinate on the original strand
    9	int	Target end coordinate on the original strand
    10	int	Number of matching bases in the mapping
    11	int	Number bases, including gaps, in the mapping
    12	int	Mapping quality (0-255 with 255 for missing)

    Parameters
    ----------
    paf_file : IO
        IO readable object with the paf file.

    Returns
    -------
    paf : tuple
        Returns the PAF tuple, but without quality and additional fields.
    """
    for paf_line in paf_file:
        p = paf_line.strip().split("\t")
        # Convert the rigth fields to integer, and
        # discard mapping quality, and additional fields after that.
        paf_tuple = (
            p[0],
            int(p[1]),
            int(p[2]),
            int(p[3]),  # Query
            p[4],  # Orientation
            p[5],
            int(p[6]),
            int(p[7]),
            int(p[8]),  # Target
            int(p[9]),
            int(p[10]),  # Mapping info
        )
        yield paf_tuple


def clean_small_overlaps(pafs, min_overlap_size, min_matching_bp):
    """
    Step 2.1

    Remove all pafs that have an overlap too small and/or a matching region
    too small.

    Returns
    -------
    pafs
    """
    return filter(lambda p: p[9] > min_matching_bp and p[10] > min_overlap_size, pafs)


def filter_overlaps_and_create_seq_lens(
    pafs: Iterable[PAF], min_coverage: int
) -> Tuple[Mappings, Dict[str, int]]:
    """
    Step 2.1

    This function gathers all pafs and only returns those overlaps that
    have the minimum coverage. It will also trim the mapping outside the
    region with not enough coverage.

    This function will also make the sequence lengths dictionary.

    Returns
    ------
    overhangs : Tuple[dict, dict]
        The first value in the tuple is the overhangs, which is a dictionary
        with as a key the query->target mapping with strand char
        and as a value a list with tuples
        with qstart, qend, tstart, tend.

        The second tuple is the sequence dictionary which maps the sequence name to
        its length.
    """
    mappings: All_Mappings = defaultdict(list)
    seq_lens: Dict[str, int] = dict()
    for p in pafs:
        mappings[p[0], p[5], p[4]].append((p[2], p[3], p[7], p[8]))
        seq_lens[p[0]] = p[1]

    # trim_mapping function will trim the mapping based on the list of mappings
    # and the min_coverage.
    trimmed_mappings_iter = (
        (key, trim_mapping(mappings_list, min_coverage))
        for key, mappings_list in mappings.items()
    )
    # The trim_mapping function will return None if there is no mapping with
    # the min_coverage. This is filtered out here.
    trimmed_mappings: Mappings = {
        key: trimmed for key, trimmed in trimmed_mappings_iter if trimmed
    }
    return trimmed_mappings, seq_lens


def trim_mapping(mapping_list, min_coverage) -> Optional[Tuple[int, int, int, int]]:
    """
    Trim the mappings where the min_coverage is not achieved.

    This function will calculate the mapping for each strand (query and target)
    separately. (They can differ slightly.)

    Returns
    -------
    mapping : Tuple[int, int, int, int]
        The mapping with query start and end, and target start and end.

    If there will be no mapping left after trimming return None.
    """
    q_map_strand = map_on_strand(((m[0], m[1]) for m in mapping_list), min_coverage)
    t_map_strand = map_on_strand(((m[2], m[3]) for m in mapping_list), min_coverage)

    if q_map_strand and t_map_strand:
        return (*q_map_strand, *t_map_strand)
    else:
        return None


def map_on_strand(coords, min_coverage) -> Optional[Tuple[int, int]]:
    """
    Trim the coordinates for minimum coverage.

    We do this by looping over the collection in sorted order of the first
    coordinate. We keep a priority queue with the end coordinates of the
    ranges. Each time we process a new range, we remove all the end coordinates
    that are before the beginning of the current range. Now the size of the
    priority queue is equal to the coverage.
    """
    # Use heapq as a min heap to use as a priority qeueu
    end_coords: List[int] = []
    trimmed_range_start: List[int] = []
    trimmed_range_end: List[int] = []
    in_range = False
    for b, e in sorted(coords, key=itemgetter(0)):

        # Remove all end_coords that are behind the begin of the current range.
        while end_coords and end_coords[0] < b:
            last_end = heapq.heappop(end_coords)

            # If this removing, reduces the coverage below min_coverage and we were
            # in_range, add the last removed end as the end of the trimmed_range.
            if len(end_coords) < min_coverage:
                if in_range:
                    trimmed_range_end.append(last_end)
                    in_range = False

        # Now add the new end coord to the end_coords
        heapq.heappush(end_coords, e)

        # If this new range bring the coverage on or over the min_coverage, and we
        # were not in range, start a new range with the beginning of the current range.
        if len(end_coords) >= min_coverage:
            if not in_range:
                trimmed_range_start.append(b)
                in_range = True

    # Now return the maximum range with min_coverage
    if trimmed_range_start:
        return max(
            zip(trimmed_range_start, trimmed_range_end), key=lambda c: c[1] - c[0]
        )
    else:
        return None


def create_genome_graph(
    mappings: Mappings,
    seq_lens: Dict[str, int],
    max_overhang: int,
    max_overhang_ratio: float,
) -> Genome_Graph:
    """
    Step 2.2; Algorithm 5

    This function finds the overlaps with the correct characteristics.

    With all the overlaps with the correct characteristics a new dictionary
    is created which represents the genome graph.

    Returns
    -------
    genome_graph : Genome_Graph
        Returns a dictionary that represents the genome graph.
    """
    genome_graph: Genome_Graph = dict()
    for (query, target, ori), (qstart, qend, tstart, tend) in mappings.items():
        # We only add one edge per mapping as we assume the PAF file is already
        # symmetric. (I.e. it contains all-vs-all mappings where all reads can be
        # found in both the target role as the query role.)

        # First we change the situation to the 'normal' one, where we look
        # to the same site of both strands.
        if ori == "-":
            tlen = seq_lens[target]
            tstart, tend = tlen - tend, tlen - tstart




# Graph cleaning functions.


def remove_transitive_edges(genome_graph, fuzz):
    """
    Myers 2005
    """
    pass


def remove_small_tips(genome_graph, min_size_tip):
    """
    ...
    """
    pass


def popping_bubbles(genome_graph, min_size_tip):
    """
    ...
    """
    pass


def create_unitigs(genome_graph):
    """
    ...
    """
    pass


# Print to file functions.


def print_gfa_file(genome_graph, read_file, output_gfa_file):
    """
    ...
    """
    pass


def miniasm(paf_file_name, reads_file_name, output_gfa):
    """
    This is the miniasm function
    """
    # Overlap classification parameters
    min_overlap_size = 2000
    min_matching_bp = 100
    max_overhang = 1000
    max_overhang_ratio = 0.8
    min_coverage = 3

    # Graph cleaning paramaters
    min_size_tip = 4
    fuzz = 10
    d = 50000  # Something with the bubbles

    # Create graph from pafs (2.1)
    with open(paf_file_name) as paf_file:
        pafs = read_paf_file(paf_file)

        pafs = clean_small_overlaps(pafs, min_overlap_size, min_matching_bp)
        mappings, seq_lens = filter_overlaps_and_create_seq_lens(pafs, min_coverage)

    # 2.2
    genome_graph = create_genome_graph(
        mappings, seq_lens, max_overhang, max_overhang_ratio
    )

    # Graph cleaning (2.3)
    genome_graph = remove_transitive_edges(genome_graph, fuzz)
    genome_graph = remove_small_tips(genome_graph, min_size_tip)
    genome_graph = popping_bubbles(genome_graph, d)
    unitig_genome_graph = create_unitigs(genome_graph)

    # Convert to gfa format
    with open(reads_file_name) as reads_file:
        read_seqs = read_fastx(reads_file, "fastq")
        with open(output_gfa, "w") if output_gfa else sys.stdout as out:
            print_gfa_file(unitig_genome_graph, read_seqs, out)


if __name__ == "__main__":
    print("hi")
