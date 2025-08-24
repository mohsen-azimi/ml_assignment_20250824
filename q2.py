PHONEME_DICT = {
    "ABACUS": [["AE", "B", "AH", "K", "AH", "S"]],
    "BOOK": [["B", "UH", "K"]],
    "THEIR": [["DH", "EH", "R"]],
    "THERE": [["DH", "EH", "R"]],
    "TOMATO": [["T", "AH", "M", "AA", "T", "OW"],
               ["T", "AH", "M", "EY", "T", "OW"]],
}


# ---------------------------
#  Prefix Search Tree
# (Ref: https://en.wikipedia.org/wiki/Trie;
# https://www.geeksforgeeks.org/dsa/trie-insert-and-search/)
# ---------------------------

class PhonemePrefixTree:
    def __init__(self):
        self.root = {"branches": {}, "leaves": []}  # branches: phoneme -> node; leaves: words ending

    @staticmethod
    def build_tree(phoneme_dict):
        tree = PhonemePrefixTree()
        for word, phonemes_list in phoneme_dict.items():
            for phonemes in phonemes_list:
                node = tree.root
                for p in phonemes:
                    if p not in node["branches"]:
                        node["branches"][p] = {"branches": {}, "leaves": []}
                    node = node["branches"][p]
                node["leaves"].append(word)
        return tree


# Build once
PPT = PhonemePrefixTree.build_tree(PHONEME_DICT)


# ---------------------------
# 2) DP/DFS with memoization
# idea = Recursive DAG Traversal
# ---------------------------
def solver(phonemes):

    n_ph = len(phonemes)
    memo = {}

    def solve_from(i): # all word sequences that spell the suffix phonemes[i:].
        # Base cases
        if i in memo:
            return memo[i]
        if i == n_ph:
            return [[]]

        out = []
        node = PPT.root
        j = i

        # Walk the tree
        while j < n_ph:
            p = phonemes[j]
            if p not in node["branches"]:
                break
            node = node["branches"][p]
            j += 1

            # If a word ends here, recurse on the suffix
            if node["leaves"]:
                suffixes = solve_from(j)
                for w in node["leaves"]:
                    for suffix in suffixes:
                        out.append([w] + suffix)

        # update memo and return
        memo[i] = out
        return out

    return solve_from(0)  # Start solving from the beginning


# ---------------------------
if __name__ == "__main__":
    inputs = ["DH", "EH", "R", "T", "AH", "M", "AA", "T", "OW"]
    solutions = solver(inputs)
    print("Inputs:", inputs)
    print("|Solutions:")
    for s in solutions:
        print("|__", s)

    inputs = ["DH", "EH", "R", "T", "AH", "M", "AA", "T", "AH"]
    solutions = solver(inputs)
    print("\n-------------------\nInputs:", inputs)
    print("|Solutions:")
    for s in solutions:
        print("|__", s)
