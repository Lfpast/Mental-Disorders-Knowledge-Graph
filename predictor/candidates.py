from itertools import product
from typing import Iterable, Iterator, List, Sequence, Tuple

Pair = Tuple[str, str]


class GlobalCandidateGenerator:
    def __init__(self, disease_nodes: Sequence[str], drug_nodes: Sequence[str]) -> None:
        self.diseases = list(disease_nodes)
        self.drugs = list(drug_nodes)

    def iter_pairs(self) -> Iterator[Pair]:
        for disease, drug in product(self.diseases, self.drugs):
            yield (disease, drug)

    def generate_all_pairs(self) -> List[Pair]:
        return list(self.iter_pairs())

    def label_pairs(self, pairs: Iterable[Pair], treat_edges: set[Pair]) -> Iterator[int]:
        for pair in pairs:
            yield 1 if pair in treat_edges else 0

    def iter_labeled_pairs(self, treat_edges: set[Pair]) -> Iterator[Tuple[Pair, int]]:
        for pair in self.iter_pairs():
            yield (pair, 1 if pair in treat_edges else 0)
