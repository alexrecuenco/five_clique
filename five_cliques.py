""" LICENSE: See LICENSE"""
from string import ascii_lowercase
from tqdm import tqdm
import csv
from typing import Dict, Iterable, List, Set
from collections import defaultdict
from itertools import product


# This is a compact representation of a "char set" that fits in just 26 bits
# each bit represents a letter in the alphabet
# we can quickly check for an intersection between two words with the bitwise AND operator
# or union them with bitwise OR
def letter_set(word: str):
    ret = 0
    for char in word:
        alphaIndex = ord(char) - ord("a")
        ret |= 1 << alphaIndex
    return ret


def bit_count(num: int):
    count = 0
    while num:
        num &= num - 1
        count += 1
    return count


def read_words(fname: str = "words_alpha.txt") -> Dict[int, List[str]]:
    anagrams = defaultdict(list)
    # words_alpha.txt from https://github.com/dwyl/english-words
    with open(fname) as f:
        word: str
        for word in f:
            word = word.strip()
            if len(word) != 5:
                continue
            # compute set representation of the word
            char_set = letter_set(word)
            if bit_count(char_set) != 5:
                continue

            anagrams[char_set].append(word)
    return dict(anagrams)


# prepare a data structure for all five-letter words in string and set representation


def neighbor_graph(anagrams: Dict[int, List[str]]) -> Dict[int, Set[int]]:
    # compute the 'neighbors' for each word, i.e. other words which have entirely
    # distinct letters
    graph = defaultdict(set)
    anagram_sets = sorted(anagrams.keys())
    for i, char_set in enumerate(tqdm(anagram_sets)):
        neighbors = graph[char_set]
        for other_set in anagram_sets[i + 1 :]:
            if not (char_set & other_set):
                neighbors.add(other_set)
    return dict(graph)


ALL_LETTERS = letter_set(ascii_lowercase)
# Each word is obtained from the previous one plus the new neighbor, we untangle them substracting
def untangle_words(*words: List[int]):
    remain = ALL_LETTERS
    for word in reversed(words):
        word = word & remain
        yield word
        remain = remain - word


TOTAL_LENGTH = 5
TOTAL_LENGTH_MINUS_ONE = 4

# We maintain a set of tree branches that we have already visited and know
# don't contain any clique subsets
__prune: Set[int] = set()
__add_to_checked = __prune.add
__checked_already = __prune.__contains__


def merge(
    last_word: int,
    *words: List[int],
    neighbors: Set[int],
    graph: Dict[int, Set[int]],
    cliques: List[Iterable[int]],
    # We bind them to the local environment for cpython tiiiny speed up
    add_to_checked=__add_to_checked,
    checked_already=__checked_already,
) -> bool:
    num_words = len(words) + 1
    if len(neighbors) + num_words < TOTAL_LENGTH:
        return False
    if num_words == TOTAL_LENGTH_MINUS_ONE:
        for neighbor_word in neighbors:
            all_letters = neighbor_word | last_word
            cliques.append(tuple(untangle_words(all_letters, last_word, *words)))
        return True

    result = False
    for neighbor_word in neighbors:
        if last_word & neighbor_word:
            continue

        shared_letters = last_word | neighbor_word

        if checked_already(shared_letters):
            continue

        if merge(
            shared_letters,
            last_word,
            *words,
            neighbors=neighbors & graph[neighbor_word],
            cliques=cliques,
            graph=graph,
        ):
            result = True
        else:
            add_to_checked(shared_letters)

    return result


def expand(word_list: List[int], anagrams: Dict[int, List[str]]) -> Iterable[List[str]]:
    all_anagrams = [anagrams[word] for word in word_list]
    return product(*all_anagrams)


def main():
    print("--- reading words file ---")
    anagrams = read_words("words_alpha.txt")
    print("--- building neighborhoods ---")
    graph = neighbor_graph(anagrams)

    anagram_sets = sorted(anagrams.keys())
    cliques: List[Iterable[int]] = []
    print("--- finding cliques ---")

    for word in tqdm(anagram_sets):
        merge(word, neighbors=graph[word], graph=graph, cliques=cliques)

    print("completed! Found %d cliques" % len(cliques))
    print(f"pruned {len(__prune)}")

    print("--- write to output ---")

    answers = sorted(
        tuple(sorted(expanded_words))
        for cliq in cliques
        for expanded_words in expand(cliq, anagrams)
    )

    print(f"answers {len(answers)}")

    with open("cliques.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        for answer in answers:
            writer.writerow(answer)


if __name__ == "__main__":
    main()
