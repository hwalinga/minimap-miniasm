#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimap and miniasm in Python.

Requirements
------------
For this program the minimum required is Python version 3.6.

No additional requirements needed.

README
------
This program consists of two subprograms, minimap, and miniasm.

You can run one or the other using the first argument, followed by the other
arguments.

For example:
python3 hielkewalinga_4374561_minimap-miniasm.py minimap query.fq target.fq > output.paf

For more details, you can use the --help flag to one of the subprograms:
python3 hielkewalinga_4374561_minimap-miniasm.py miniasm --help

The main program (without supplying a subprogram) accepts a --help as well.

NB.

There is also a third subprogram, which is the 'test' subprogram. This will
run the unittest included in this program. These unittest reflect the tests
given on the project page of this project.
https://abeellab.github.io/cs4255/minimap-doc.html

There is also a fourth subprogram 'mytest', used for debugging purposes.
"""

import argparse
import heapq
import sys
import unittest
from argparse import Namespace
from collections import defaultdict, deque
from functools import partial
from itertools import chain, groupby, islice, repeat, tee
from operator import itemgetter
from typing import (
    IO,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

###########
# MINIMAP #
###########

# Type aliases minimap

# A set with tuples representing the hash value, the position, and the strand.
Minimizers = Set[Tuple[int, int, bool]]

# A function that takes a sequence and returns the minimizers.
Minimizer_Sketch = Callable[[str], Minimizers]

# A function that takes a sequence and returns a hash integer.
Seq_Hash = Callable[[str], int]

# A dictionary that maps the hash to a list of tuples with
# target sequence id, position, and strand.
Target_Index = Dict[int, List[Tuple[str, int, bool]]]

# A dictionary that maps the sequence name to the length.
Seq_Lens = Dict[str, int]

# For PAF we will use fields 1-11.
# So not using the quality one, and optional additional fields.
Strand_Range = Tuple[str, int, int, int]
PAF = Tuple[Strand_Range, str, Strand_Range, Tuple[int, int]]


def reverse_complement(
    s: str, complement: Dict[int, Optional[int]] = str.maketrans("ACGT", "TGCA")
) -> str:
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


def hash_seq(
    s: str, base_values: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
) -> int:
    """
    Hash a sequence (or k-mer).

    Using Phi(A)=0, Phi(C)=1, Phi(G)=2, and Phi(T)=3

    Calculate with s = a_1 ... a_k:
    Phi(s) = Phi(a_1) x 4 ^ (k-1) + Phi(a_2) x 4 ^ (k-2) ... + Phi(a_k)

    Parameters
    ----------
    s : str
        s is the sequence (or k-mer) that needs to be hashed
    base_values : Dict[str, int]
        This is a dict that maps the base pair to the value used in the hash.
        Default is {"A": 0, "C": 1, "G": 2, "T": 3}

    Returns
    -------
    x : int
        Returns the integer hashed
    """
    k = len(s)
    return sum(base_values[b] * 4 ** (k - i - 1) for i, b in enumerate(s))


def invertable_hash(x: int, p: int = 64) -> int:
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

    Notes
    -----
    The paper mentions that the hash can be a 2 * k bit hash. However
    the tests included of in the project assume p=64, so I went with that instead.
    However, p=63 would make a bit more sense, as that is the maximum size of
    Python integer (So, one bit smaller than a C integer size). Luckily this
    isn't a big problem as Python will increase the memory size of an integer
    automatically if it will overflow.
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
        i is the position where the k-mer starts. (Also start position on reverse)
        r is the strand

    Notes
    -----
    NB. In the test set the starting coords of the minimizers is
    reversed if the minimizer is found on the reverse strand.
    This is however different from the given pseudocode which does not perform
    this reversal. The reason is explained here.

    Algo 4 will also not work if you perform the reversal.
    The reason for this is that the clustering is performed
    on i + i_t instead of i - i_t for opposite strands.

    Different strands use a different invariant to find minimizers
    with the same maplen, as shown below:
    The invariants:
    Same strand: i - i_t = qlen - maplen
    Opposite strand: i + i_t = qlen + tlen - maplen

    Of course this wasn't needed when we would use reversed coords for different
    strands, but the convenience comes in the final end of the implementation.

    Here we made a distinction between a qeury-target mapping on the same strands
    and a query-target mapping on different strands. What is convenient is that
    a query-target mapping is the same as a query(reversed)-target(reversed) mapping,
    and that a query(reversed)-target mapping is the same as a query-target(reversed)
    mapping. However, since we did not reverse any coords we can just output a
    query-target mapping for a query(reversed)-target(reversed) mapping (same strand),
    and a query-target(reversed) mapping for a query(reversed)-target mapping.
    Conveniently we only reverse coords for the target strand, when the mapping
    is not on the same strand.
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
                # NB. We do not reverse the coord here, but use different invariants
                # for the opposite strands. See Algo 4: compute_minimizers
                M.add((m, i + j, v < u))

        # Move w window one k-mer further. (If not more k-mers, we are done.)
        new_kmer = next(kmers, None)
        if not new_kmer:
            break
        w_window.popleft()
        w_window.append(new_kmer)
        i += 1

    return M


def get_func_minimizer_sketch(
    w: int, k: int, hash_func: Seq_Hash = lambda s: invertable_hash(hash_seq(s))
) -> Minimizer_Sketch:
    """
    A function to set all values to the :func:`compute_minimizers` function.

    Paramaters
    ----------
    w : int
        Amount of k-mers in window.
    k : int
        Length of the k-mer.
    hash_func : Seq_Hash
        A function to map a sequence (str) to a hash integer.
        Defaults to the invertible hash from the minimap/miniasm paper.

    Returns
    -------
    minimizer_sketch : Minimizer_Sketch
        A function that takes a sequence and returns the minimizers

    Raises
    ------
    Exception
        If k * 2 is larger than the bit of the hash, do not proceed.

        As integers can be arbitrarily large in Python, changing the hardcoded
        hash size (p) can be used to run with a 'k' larger than 32.

        You can probably run with a larger k anyway, as hash collisions aren't
        that bad. During clustering of minimizer hits this is dealt with as
        minimizer hits aren't guaranteed to be correct.
    """
    if k >= 32:
        raise Exception("k is too big")
    return partial(compute_minimizers, w=w, k=k, hash_func=hash_func,)


def index_targets(
    target_seqs: Iterable[Tuple[str, str]], minimizer_sketch: Minimizer_Sketch
) -> Tuple[Target_Index, Seq_Lens]:
    """
    Algorithm 3.
    """
    H_list = []
    seq_lens: Seq_Lens = dict()
    for name, seq in target_seqs:

        seq_lens[name] = len(seq)

        M = minimizer_sketch(seq)
        for h, i, r in M:
            H_list.append((h, (name, i, r)))

    H_list.sort(key=itemgetter(0))

    H_dict = {
        h: list(map(itemgetter(1), vals))
        for h, vals in groupby(H_list, key=itemgetter(0))
    }

    return H_dict, seq_lens


def map_query(
    qseq: str, qname: str, seq_lens: Seq_Lens, H: Target_Index, args: Namespace,
) -> Iterable[PAF]:
    """
    Takes a query and maps it to the target indexers.

    Algorithm 4.

    Returns
    -------
    paf_it : Iterable[PAF]
        PAF tuple, this is a structured PAF line.

    Notes
    -----
    PAF file : A PAF file is used to describe mapings. It is a tab seperated file
    with te following fields:

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
    """
    M = args.minimizer_sketch(qseq)
    A = []

    # Collect minimizers hit
    for h, i, r in M:
        for t, i_t, r_t in H[h]:
            if t == qname:
                continue  # Do not map the same read to the same read.
            # NB. For opposite strand the coord is not reversed,
            # therefore use the invariants for the third element as such:
            # Same strand: i - i_t = qlen - maplen
            # Opposite strand: i + i_t = qlen + tlen - maplen
            # The third element can now be used to group minimizers that create
            # the same kind of mapping (i.e. that have the same maplen).
            if r == r_t:
                A.append((t, 0, i - i_t, i_t))
            else:
                A.append((t, 1, i + i_t, i_t))

    # Python automatically sorts tuples with radix sort if no key provided.
    # This is what we want here.
    A.sort()

    b = 0
    for e in range(len(A)):
        if (
            e + 1 == len(A)
            or A[e + 1][0] != A[e][0]
            or A[e + 1][1] != A[e][1]
            # Here we perform single linkage if the link < args.epsilon
            or A[e + 1][2] - A[e][2] >= args.epsilon
        ):
            # lh3/minimap also merges minimizers
            # if there is a -m=50% overlap between the two, I do not.

            potential_overlap = A[b:e]
            b = e + 1

            orientation = "-" if potential_overlap[0][1] == 1 else "+"

            # Find the maximum colinear subset by finding the longest increasing
            # subset of the target positions.
            # Notice that for the invariants above, the only factor that can
            # change slightly is the map length.
            # Find the longest increasing subset will make sure that the small
            # variations in the map length does not cause a target coord to
            # be shifted that far that it starts mixing with another match.

            if orientation == "-":
                # Since we use coords that have not been reversed for opposite
                # mappings we have to find the longest decreasing subset here.
                potential_overlap.reverse()
                indices = longest_decreasing_subset(
                    map(itemgetter(3), potential_overlap)
                )
            else:
                indices = longest_increasing_subset(
                    map(itemgetter(3), potential_overlap)
                )

            # We have now a (potential) cluster that represents a mapping.
            # Now use this cluster to create paf or reject the cluster.

            # NB. lh3/minimap breaks this chain if
            # there is a gap longer than -g=10000, I do not.

            if len(indices) <= args.min_subset:
                # Not enough minimizers in this mapping, we do not use this cluster.
                continue

            bh = potential_overlap[indices[0]]  # First minimizer for query
            eh = potential_overlap[indices[-1]]  # Last minimizer for query

            # Does not check for overlapping minimizers
            mapped_bp = len(indices) * args.k

            qlen = len(qseq)
            tname = potential_overlap[0][0]
            tlen = seq_lens[tname]

            # Undo the transformation, and find the exact coords of the ranges.
            if orientation == "+":  # (+) same strand
                tstart, tend = bh[3], eh[3] + args.k
                qstart, qend = bh[2] + bh[3], eh[2] + eh[3] + args.k
            else:  # Opposite strand (-)
                # Note: here we finally reverse the coordinates of the target strand.
                # The only place where we do that is here!
                # It could be that in fact query was reversed here, but still
                # that doesn't matter as query(reversed)-target is equivalent
                # to query-target(reversed), so we pick the latter always,
                # and do not have to bother with reversing query, as we never
                # reversed before.
                tstart, tend = tlen - bh[3] - args.k, tlen - eh[3]
                qstart, qend = bh[2] - bh[3], eh[2] - eh[3] + args.k

            maplen = max(qend - qstart, tend - tstart)
            if maplen < args.min_overlap:
                # Mapping not large enough. Do not use.
                continue

            query = qname, qlen, qstart, qend
            target = tname, tlen, tstart, tend
            mapping_info = mapped_bp, maplen

            paf = query, orientation, target, mapping_info

            yield paf


def longest_decreasing_subset(seq: Iterable[int]) -> List[int]:
    seq = list(seq)
    seq.reverse()
    indices = longest_increasing_subset(seq)
    indices_reversed = [len(seq) - ind - 1 for ind in indices]
    indices_reversed.reverse()
    return indices_reversed


def longest_increasing_subset(seq: Iterable[int]) -> List[int]:
    """
    Find the longest increasing subsequence.

    Source: https://stackoverflow.com/questions/3992697/longest-increasing-subsequence

    Returns
    -------
    indices : List[int]
        Returns the indices of the longest increasing subsequence.
    """
    seq = list(seq)
    if not seq:
        return seq

    P: List[Optional[int]] = [None]
    M = [0]

    for i in range(1, len(seq)):

        # Binary search
        lo = 0
        up = len(M)

        if seq[M[up - 1]] < seq[i]:
            j = up
            M.append(i)
        else:
            while up - lo > 1:
                mid = (lo + up) // 2
                if seq[M[mid - 1]] < seq[i]:
                    lo = mid
                else:
                    up = mid
            j = lo

            if seq[i] < seq[M[j]]:
                M[j] = i

        # Update P
        P.append(None if j == 0 else M[j - 1])

    # Trace back using the predecessor array (P).
    def trace(i):
        if i is not None:
            yield from trace(P[i])
            yield i

    indices = list(trace(M[-1]))
    return indices


def read_fastx(file: IO) -> Iterator[Tuple[str, str]]:
    """
    Currently, this function just assumes for fasta that the sequences and the
    fasta header are on alternating lines.

    Parameters
    file : IO
        Name of the input file.
    format : str
        The format ("fasta" or "fastq")

    Returns
    -------
    it : Iterator[Tuple[str, str]]
        Returns an iterator that yields tuples with the sequence name and the sequence.
    """
    first_line = next(file)
    if first_line[0] == ">":
        group_size = 2
    elif first_line[0] == "@":
        group_size = 4
    else:
        raise Exception(f"Something is wrong with the file {file.name}")

    ind_line = enumerate(chain([first_line], file))
    for ind, line in ind_line:
        if ind % group_size == 0:
            seq_name = line.strip()[1:]
            _, seq = next(ind_line)
            # We will always use upper case sequences
            yield seq_name, seq.strip().upper()


def minimap(target_file: IO, query_file: IO, output_file: IO, args: Namespace):
    """
    This is the minimap function
    """
    if not args.w:
        args.w = round(2 * args.k / 3)  # Default is 2 / 3 of k.

    target_seqs = read_fastx(target_file)
    query_seqs = read_fastx(query_file)

    args.minimizer_sketch = get_func_minimizer_sketch(args.w, args.k)

    H, seq_lens = index_targets(target_seqs, args.minimizer_sketch)

    for qname, qseq in query_seqs:
        for t in map_query(qseq, qname, seq_lens, H, args):
            # We don't calculate quality, so we just print 255 there.
            print(*chain(*t), 255, sep="\t", file=output_file)


###########
# MINIASM #
###########


# A Mapping is described by two strings: The names of the query and the target,
# and one more string to describe if they match on the same strand or opposite
# (Just like in PAF.)
Mappings = Dict[Tuple[str, str, str], Tuple[int, int, int, int]]

# The all mappings dict is used to trim ranges that don't have enough mapped coverage.
All_Mappings = Dict[str, List[Tuple[int, int, int, int, str, str]]]

# In the genome graph each strand is described with the name and a bool
# to indicate strand side. If True, it is the same strand,
# if False, it is the opposite.
# (This is equivalent to {True: '+', '-': False} compared with PAF.)
# The other integer indicates the lenght of the mapping.
# The key value pairs indicate from-to mapping.
Vertex = Tuple[str, bool]
Genome_Graph = DefaultDict[Vertex, List[Tuple[Vertex, int]]]


def read_paf_file(paf_file: IO) -> Iterator[PAF]:
    """
    Just read the file and yields each line as a paf tuple.

    Notes
    -----
    PAF file : A PAF file is used to describe mapings. It is a tab seperated file
    with te following fields:

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
        query = p[0], int(p[1]), int(p[2]), int(p[3])
        orientation = str(p[4])
        target = p[5], int(p[6]), int(p[7]), int(p[8])
        mapping_info = int(p[9]), int(p[10])
        paf = query, orientation, target, mapping_info
        yield paf


def clean_small_overlaps(
    pafs: Iterable[PAF], min_overlap_size: int, min_matching_bp: int
) -> Iterator[PAF]:
    """
    Step 2.1

    Remove all pafs that have an overlap too small and/or a matching region too small.

    Returns
    -------
    pafs : Iterator[PAF]
        An iterable with pafs.
    """
    return filter(
        lambda p: p[3][0] > min_matching_bp and p[3][1] > min_overlap_size, pafs
    )


def filter_overlaps_and_create_seq_lens(
    pafs: Iterable[PAF], min_coverage: int
) -> Tuple[Mappings, Seq_Lens]:
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
    seq_lens: Seq_Lens = dict()

    for query, orientation, target, mapping_info in pafs:
        mappings[query[0]].append(
            (query[2], query[3], target[2], target[3], target[0], orientation)
        )
        seq_lens[query[0]] = query[1]

    # trim_mapping function will trim the mapping based on the list of mappings
    # and the min_coverage.

    trimmed_mappings: Mappings = {
        (qname, trimmed_mapping[4], trimmed_mapping[5]): trimmed_mapping[:4]
        for qname, mappings_list in mappings.items()
        for trimmed_mapping in trim_mappings(mappings_list, min_coverage)
    }
    return trimmed_mappings, seq_lens


def trim_mappings(
    mappings_list: List[Tuple[int, int, int, int, str, str]], min_coverage: int
) -> Iterator[Tuple[int, int, int, int, str, str]]:
    """
    Trim the mappings where the min_coverage is not achieved.

    This function will calculate the mapping for each strand (query and target)
    separately. (They can differ slightly.)

    Returns
    -------
    trimmed_mapping_list : List[Tuple[int, int, int, int, str, str]]
        The mapping with query start and end, and target start and end.

    If there will be no mapping left it is not included in the list again.

    Notes
    -----
    For this algorithm there is still an unresolved edge case. That is, when a
    mapping can be found in two different trimmed ranges, it is unclear if
    this mapping should be cut in two. Here we just keep the first half.
    """
    trimmed_ranges_it = find_minimum_coverage_range(
        ((m[0], m[1]) for m in mappings_list), min_coverage
    )
    return trim_mappings_with_ranges(mappings_list, trimmed_ranges_it)


def trim_mappings_with_ranges(
    mappings_list: List[Tuple[int, int, int, int, str, str]],
    trimmed_ranges_it: Iterator[Tuple[int, int]],
) -> Iterator[Tuple[int, int, int, int, str, str]]:

    trim_start, trim_end = next(trimmed_ranges_it)
    for mapping in sorted(mappings_list, key=itemgetter(0)):
        qstart, qend, tstart, tend, tname, ori = mapping

        # Move to a new trimmed range if needed
        while qstart > trim_end:
            trim_range = next(trimmed_ranges_it, None)
            if not trim_range:
                return
            trim_start, trim_end = trim_range

        qtrim, etrim = 0, 0

        if trim_start > qstart:  # Not inside trimmed range -> trim
            strim = trim_start - qstart
            qstart = trim_start

        if trim_end < qend:  # Not inside trimmed_range
            etrim = qend - trim_end
            qend = trim_end

        if not qtrim and not etrim:  # No trimming, prevent tuple reconstruction.
            yield mapping
        else:
            # We trim the target range as well with the same parameters.
            # There can however be an edge case where the target range is
            # slightly different and give an invalid range, then we just
            # skip it completely
            tstart = tstart + strim
            tend = tend - etrim
            if tstart < tend and tstart > 0 and tend > 0:
                yield qstart, qend, tstart, tend, tname, ori


def find_minimum_coverage_range(
    coords: Iterable[Tuple[int, int]], min_coverage: int
) -> Iterator[Tuple[int, int]]:
    """
    Trim the coordinates for minimum coverage.

    We do this by looping over the collection in sorted order of the first
    coordinate. We keep a priority queue with the end coordinates of the
    ranges. Each time we process a new range, we remove all the end coordinates
    that are before the beginning of the current range. Now the size of the
    priority queue is equal to the coverage.

    Returns
    -------
    trimmed_mappings_it : Iterator[Tuple[int, int]]
        Returns the trimmed ranges in sorted order as an iterator.
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

    # Now return all the trimmed ranges with the minimum coverage
    return zip(trimmed_range_start, trimmed_range_end)


def create_genome_graph(
    mappings: Mappings,
    seq_lens: Seq_Lens,
    max_overhang: int,
    max_overhang_ratio: float,
) -> Genome_Graph:
    """
    Step 2.2; Algorithm 5

    This function finds the overlaps with the correct characteristics.

    With all the overlaps with the correct characteristics a new dictionary
    is created which represents the genome graph.

    Parameters
    ----------
    mappings : Mappings
        A dictionary with all the mappings.
    seq_lens : Seq_Lens
        A dictionary that maps the sequence name to the lenght of the sequence.
    max_overhang : int
        The maximum size the overhangs of the mappings can be to still be a
        viable mapping for the genome graph.
    max_overhang_ratio : float
        The maximum ratio between the overhang and the mapping length for
        the mapping to still be a viable candidate for the genome graph.

    Returns
    -------
    genome_graph : Genome_Graph
        Returns a dictionary that represents the genome graph.
    """
    genome_graph: Genome_Graph = defaultdict(list)
    for (query, target, ori), (qstart, qend, tstart, tend) in mappings.items():
        # We only add one edge per mapping
        # as we assume the PAF file is already
        # symmetric. (I.e. it contains all-vs-all mappings where all reads can be
        # found in both the target role as well as the query role.)

        qlen = seq_lens[query]
        tlen = seq_lens[target]

        # First we change the situation to the 'normal' one, where we look
        # to the same site of both strands.
        if ori == "-":
            tstart, tend = tlen - tend, tlen - tstart

        overhang = min(qstart, tstart) + min(qlen - qend, tlen - tend)
        maplen = max(qend - qstart, tend - tstart)

        if overhang > min(max_overhang, max_overhang_ratio * maplen):
            # Internal match
            continue
        elif qstart <= tstart and qlen - qend <= tlen - tend:
            # Query contained
            continue
        elif qstart >= tstart and qlen - qend >= tlen - tend:
            # Target contained
            continue
        elif qstart > tstart:
            # Query to target
            target_ori = ori != "-"  # False if on opposite strands.
            genome_graph[query, True].append(((target, target_ori), maplen))
        else:
            # Target to query
            # For the symmetry we only add query to target mappings.
            # For this target to query mapping, we just swap the strand side
            # and add query complement strand to target complement strand.
            # (NB. The target to query mapping will be added with another
            # mapping that swaps this roles. Provided the PAF is symmetrical.)

            # To change target-to-query mapping, to query-to-target mapping,
            # we swap strands. If mapping is on opposite strand, target is
            # on the same strand again.
            target_ori = ori == "-"
            genome_graph[query, False].append(((target, target_ori), maplen))

    return genome_graph


# Graph cleaning functions.


def remove_transitive_edges(
    genome_graph: Genome_Graph, vertices: List[Vertex], seq_lens: Seq_Lens, fuzz: int
) -> None:
    """
    Myers 2005

    This function goes through the genome graph and removes transitive edges.
    These are edges that have a mapping that is not really needed as other
    edges already show that the nodes relate.
    """

    def edge_len(edge_info: Tuple[Vertex, int]) -> int:
        vertex, maplen = edge_info
        return seq_lens[vertex[0]] - maplen

    for val in genome_graph.values():
        # Sort all edges so that the longest
        val.sort(key=edge_len)

    # This dict has the following convention: {0: vacant, 1: inplay, 2: eliminated}
    vertices_dict: Dict[Vertex, int] = dict.fromkeys(vertices, 0)

    for v in vertices:
        for w, _ in genome_graph[v]:
            vertices_dict[w] = 1

        longest = edge_len(genome_graph[v][-1]) + fuzz

        for w, maplen_w in genome_graph[v]:
            if vertices_dict[w] != 1:
                continue
            v2w_len = edge_len((w, maplen_w))
            edge_it = iter(genome_graph[w])

            # Check the smallest edge of w
            x, maplen = next(edge_it)
            if vertices_dict[x] == 1:
                vertices_dict[x] == 2  # eliminated

            for x, maplen_x in edge_it:

                # Check all edges smaller than fuzz.
                if edge_len((x, maplen_x)) < fuzz and vertices_dict[x] == 1:
                    vertices_dict[x] == 2  # eliminated
                # With fuzz=10, this might seem very small,
                # but the coverage trimming might
                # reduce a mapping below what you could get from minimap

                # When edge too long, we don't have to check anymore
                if v2w_len + edge_len((x, maplen_x)) > longest:
                    break

                if vertices_dict[x] == 1:
                    vertices_dict[x] = 2  # eliminated

        new_edges: List[Tuple[Vertex, int]] = []
        for w, maplen in genome_graph[v]:
            if vertices_dict[w] != 2:
                new_edges.append((w, maplen))
            vertices_dict[w] = 0

        if new_edges:
            genome_graph[v] = new_edges
        else:
            del genome_graph[v]


def remove_small_tip(genome_graph: Genome_Graph, min_size_tip: int, v: Vertex) -> None:
    """
    Checks if v is the end of a small tip (a small tip is defined if it connects
    to something biggers within min_size_tip nodes).
    """
    # We find the tip of edges,
    # by inspecting if the symmetry vertex does not contain edges.
    if genome_graph.get((v[0], not v[1])):
        return

    # This is a tip, see if it is small.

    # Follow incoming edges and see if we hit a junction within min_size_tip.
    tip_consists: List[Vertex] = []
    for _ in repeat(None, min_size_tip):
        # We check if there are more than one edges in the vertex and its complement.
        # If there is more than one in one of them, this is a junction,
        # not part of the tip.
        if len(genome_graph[v]) > 1 or len(genome_graph[v[0], not v[1]]) > 1:
            # Junction: remove tip and return
            for w in tip_consists:
                del genome_graph[w]
                del genome_graph[w[0], not v[1]]
            return
        tip_consists.append(v)
        v = genome_graph[v][0][0]

    # Moved over edges and no junction, so this is a long tip.
    return


def pop_bubble(
    genome_graph: Genome_Graph, seq_lens: Seq_Lens, probe_distance: int, start: Vertex,
) -> None:
    """
    Popping small bubbles from the genome_graph.

    Algorithm 6.

    A bubble is defined as the isolated part of the genome from node A to node B
    where there are multiple paths from A to B. The probe_distance here defines
    how big a bubble can be.

    We remove all of the bubble accept a chosen path in the bubble. This path
    is chosen by greedily pick the largest edge when we traverse the bubble.
    """
    if len(genome_graph[start]) < 2:  # Cannot be the source of a bubble.
        return

    # Keep track of a path through the bubble, and all other vertices
    bubble_path = {start}
    path_tip = start
    other_vertices = set()

    # Keeping track of things to find the bubble.
    unvisited_incoming = dict()
    distances = dict.fromkeys(genome_graph.keys(), sys.maxsize)
    distances[start] = 0
    S = [start]
    p = 0  # Number of visited vertices not yet in S.

    while S:
        v = S.pop()
        first = True
        for w, maplen in sorted(genome_graph[v], key=itemgetter(1), reverse=True):
            if path_tip == v and first:
                # We keep track of a path by checking if the current vertex
                # is in the last in the path and appending the first next vertex.
                # The sorting puts the longest map length as the preferred path.
                bubble_path.add(w)
                path_tip = w
                first = False
            elif w not in bubble_path:
                other_vertices.add(w)

            if w == start:  # A circle is not a bubble.
                return

            new_dist = distances[v] + seq_lens[v[0]] - maplen
            if new_dist > probe_distance:  # Too far away
                return

            if distances[w] == sys.maxsize:  # Not visited
                # We make use of the symmetry to find the incoming edges.
                unvisited_incoming[w] = len(genome_graph[w[0], not [1]])
                p += 1
            elif new_dist < distances[w]:
                distances[w] = new_dist

            unvisited_incoming[w] -= 1
            if unvisited_incoming[w] == 0:  # We have visisted all incoming edges.
                if genome_graph[w]:  # Not a tip
                    S.append(w)
                p -= 1

        if len(S) == 1 and p == 0:  # This is the sink.
            # Now remove all vertices that are part of the bubble, but not the path.
            # As a bonus we remove the other side as well.
            for v in other_vertices:
                del genome_graph[v]
                del genome_graph[v[0], not v[1]]
            return


def create_unitigs(
    genome_graph: Genome_Graph, vertices: List[Vertex]
) -> Dict[str, List[Tuple[Vertex, int]]]:
    """
    Goes over the graph and when a path does not branches it merges the nodes of
    that path. These mergings are saved in a dict and returned. This dict can
    later be used to create the sequences that form the unitigs.
    """

    def utg_to_name(utg_nr: int) -> str:
        return f"{utg_nr:06}"

    unitig_to_reads: Dict[str, List[Tuple[Vertex, int]]] = dict()
    utg_nr = 1
    converted_start: Dict[str, str] = dict()  # dict with start tip to converted utg

    for start in vertices:
        # First see if the start of the unitig goes one way.
        if len(genome_graph[start]) != 1:
            continue

        # The start of a unitig must be a junction or a tip.
        other_start = start[0], not start[1]  # Using symmetry vertex again
        in_edges = len(genome_graph[other_start])
        if in_edges == 1:
            continue

        # Continue till junction or tip
        utg_path = [(start, 0)]
        v = start
        while len(genome_graph[v]) == 1:
            del genome_graph[v]
            v_info = genome_graph[v][0]
            v = v_info[0]
            utg_path.append(v_info)

        end = v
        utg_name = converted_start[end[0]]
        if utg_name:
            ori = False
        else:
            utg_name = utg_to_name(utg_nr)
            utg_nr += 1
            ori = True

        unitig_to_reads[utg_name] = utg_path

        incoming_edges = [(w[0], not w[1]) for w, _ in genome_graph[other_start]]
        for w in incoming_edges:
            genome_graph[w] = [
                ((utg_name, ori) if x[0] == start else x, maplen)
                for x, maplen in genome_graph[w]
            ]

        genome_graph[utg_name, ori] = genome_graph[end]
        del genome_graph[end]

    return unitig_to_reads


# Print to file functions.


def gfa(
    genome_graph: Genome_Graph,
    unitig_to_reads: Dict[str, List[Tuple[Vertex, int]]],
    read_seqs: Dict[str, str],
) -> Iterator[Tuple[str, ...]]:
    """
    This function takes the genome_graph that is completely cleaned and
    with the reads, a dict to outlines how the unitigs are formed it will
    output the genome assembly lines of the GFA file.
    """

    def get_ori(ori: bool) -> str:
        return "+" if ori else "-"

    for name, ori in genome_graph.keys():
        seq = read_seqs[name]
        if seq:
            yield "S", name, seq
            continue

        # It's a utg
        utg_name = name
        utg_path = iter(unitig_to_reads[utg_name])
        name, ori = next(utg_path)[0]
        new_seq = read_seqs[name] if ori else reverse_complement(read_seqs[name])
        for (name, ori), maplen in utg_path:
            add_seq = read_seqs[name] if ori else reverse_complement(read_seqs[name])
            new_seq += add_seq[maplen:]
        yield "S", utg_name, new_seq

    for v, edges in genome_graph.items():
        for (w, maplen) in edges:
            yield "L", v[0], get_ori(v[1]), w[0], get_ori(w[1]), f"{maplen}M"


def print_gfa_file(
    genome_graph: Genome_Graph,
    unitig_to_reads: Dict[str, List[Tuple[Vertex, int]]],
    read_seqs: Dict[str, str],
    out: IO,
):
    """
    Prints the GFA lines to the output file.
    """
    for line in gfa(genome_graph, unitig_to_reads, read_seqs):
        print(*line, sep="\t", file=out)


def miniasm(paf_file: IO, reads_file: IO, out: IO, args: Namespace):
    """
    This is the miniasm function.
    """
    # First read the PAF file and filter the pafs (2.1).
    pafs = read_paf_file(paf_file)

    pafs = clean_small_overlaps(pafs, args.min_overlap_size, args.min_matching_bp)
    mappings, seq_lens = filter_overlaps_and_create_seq_lens(pafs, args.min_coverage)

    # Create graph from pafs (2.2).
    genome_graph = create_genome_graph(
        mappings, seq_lens, args.max_overhang, args.max_overhang_ratio
    )

    # Graph cleaning (2.3).

    # Looping the keys-view, while modifying dict leads to bizarre behavior.
    # Thus we make a copy of the vertices for every cleaning operation.

    vertices = list(genome_graph.keys())
    remove_transitive_edges(genome_graph, vertices, seq_lens, args.fuzz)

    vertices = list(genome_graph.keys())
    for v in vertices:
        remove_small_tip(genome_graph, args.min_size_tip, v)

    vertices = list(genome_graph.keys())
    for start in vertices:
        pop_bubble(genome_graph, seq_lens, args.probe_distance, start)

    vertices = list(genome_graph.keys())
    unitig_to_reads = create_unitigs(genome_graph, vertices)

    # Convert to gfa format

    read_seqs: Dict[str, str] = dict(read_fastx(reads_file))

    print_gfa_file(genome_graph, unitig_to_reads, read_seqs, out)


#########
# Tests #
#########


def mytest():
    pass


class TestSuite(unittest.TestCase):
    def test_hash(self):
        seq = "TTA"
        h = hash_seq(seq)
        self.assertEqual(h, 60)

    def test_invertible_hash(self):
        new_h = invertable_hash(60)
        self.assertEqual(new_h, 9473421487448830983)

    def test_find_minimizers(self):
        """
        Note: As can be seen, here the coords are not the same as given in the
        project page. The reason is that we only apply reversal of the coords
        in the very end at the paf outputting. This is explained more in-depth
        in the notes of the :func:`compute_minimizers` function and all the
        details of reversal or dealt with in the :func:`map_query` function.
        """
        seq = "GATTACAACT"
        minimizer_sketch = get_func_minimizer_sketch(w=4, k=3)
        self.assertEqual(
            minimizer_sketch(seq),
            {(1396078460937419741, 1, True), (2859083004982788208, 4, True)},
        )

    def test_longest_increasing(self):
        alist = [3, 1, 2, 5, 4, 8, 6]
        indices = [1, 2, 4, 6]
        self.assertEqual(longest_increasing_subset(alist), indices)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="MINIMAP-MINIASM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This program contains a minimap and a miniasm implementation."
        " Run one of them by providing the first word the same as that program."
        " Those subprograms contain a --help as well.",
    )

    # The '-' for files types indicates stdin or stdout.
    parser.add_argument("--out", type=argparse.FileType("w"), default="-")

    subparsers = parser.add_subparsers(help="MINIMAP-MINIASM")

    # minimap argument parsing
    minimap_argparser = subparsers.add_parser(
        "minimap",
        help="Map sequences.",
        description="This is the minimap program."
        " Arguments have the same flags as the minimap program,"
        " but also have a longer name option. Defaults are equal as well."
        " Not all options are included."
        " The -L flag can only be found in the original minimap program,"
        " but not in the new minimap2 program.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    minimap_argparser.add_argument("target", type=argparse.FileType("r"))
    minimap_argparser.add_argument("query", type=argparse.FileType("r"))

    indexing = minimap_argparser.add_argument_group("Indexing options")
    indexing.add_argument("-k", help="Minimizer length.", default=15, type=int)
    indexing.add_argument(
        "-w",
        help="Minimizer window size (default is 2/3 of k).",
        type=int,
        default=argparse.SUPPRESS,
    )

    mapping = minimap_argparser.add_argument_group("Mapping options")
    mapping.add_argument(
        "-n",
        "--min-subset",
        help="Minimum minimizers in mappings",
        default=4,
        type=int,
    )
    mapping.add_argument(
        "-r",
        "--epsilon",
        help="Maximum gap size between minimizers. The bandwidth used in the"
        " single linkage clustering.",
        default=500,
        type=int,
    )
    mapping.add_argument(
        "-L",
        "--min-overlap",
        help="Minimum overlap of a mapping",
        default=100,
        type=int,
    )

    # miniasm argument parsing
    miniasm_argparser = subparsers.add_parser(
        "miniasm",
        help="Assembly from PAF file.",
        description="This is the miniasm program."
        " Arguments have the same flags as the minimap program,"
        " but also have a longer name option. Defaults are equal as well."
        " Not all options are included."
        " The -h option for miniasm has been changed to the -H option"
        " as it conflicts with the -h, --help option.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    miniasm_argparser.add_argument("paf", type=argparse.FileType("r"))

    preselection = miniasm_argparser.add_argument_group("Preselection options")
    preselection.add_argument(
        "-m",
        "--min-matching-bp",
        default=100,
        type=int,
        help="Drop mappings having less mapped bp than min-matching-bp.",
    )
    preselection.add_argument(
        "-s",
        "--min-overlap-size",
        default=1000,
        type=int,
        help="Drop mappings that are shorter than min-overlap-size.",
    )

    overlapping = miniasm_argparser.add_argument_group("Overlapping options")
    overlapping.add_argument(
        "-H", "--max-overhang", help="Maximum overhang length", default=1000, type=int
    )
    overlapping.add_argument(
        "-I",
        "--max-overhang-ratio",
        help="Maximum overhang ratio.",
        default=0.8,
        type=float,
    )

    graph = miniasm_argparser.add_argument_group("Graph layout options")
    graph.add_argument(
        "-n", "--min-coverage", help="Minimum coverage of overlap", default=3, type=int
    )
    graph.add_argument(
        "-e", "--min-size-tip", help="The minimum size of a tip", default=4, type=int
    )
    graph.add_argument(
        "-f",
        "--reads",
        type=argparse.FileType("r"),
        required=True,
        help="This is the read file. It is for now required.",
    )
    graph.add_argument(
        "-d",
        "--probe-distance",
        help="Maximum probe distance for bubble popping.",
        default=50000,
        type=int,
    )

    subparsers.add_parser("test", help="Run the unittests.")
    subparsers.add_parser("mytest", help="Run some debugging function.")

    subprograms = {"minimap", "miniasm", "test", "mytest", "--help", "-h"}

    if len(sys.argv) < 1:
        print("Debug mode for Python REPL.")
    elif sys.argv[1] not in subprograms:
        print(
            "You did not select a correct sub program"
            f" from the subprograms: {subprograms}"
        )
        sys.exit(1)
    elif sys.argv[1] == "test":
        unittest.main(argv=["ignore-this"])
    elif sys.argv[1] == "mytest":
        mytest()
    else:
        args = parser.parse_args()
        if sys.argv[1] == "minimap":
            minimap(args.target, args.query, args.out, args)
        elif sys.argv[1] == "miniasm":
            args.fuzz = 10  # Used by the removal of transitive edges
            miniasm(args.paf, args.reads, args.out, args)
