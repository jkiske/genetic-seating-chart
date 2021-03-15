import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import itertools
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Relationship:
    pair: (str, str)
    pair_is: (int, int)
    score: float


@dataclass(frozen=True)
class Table:
    relationships: list[Relationship]
    people: list[str]
    indices: list[int]

    @cached_property
    def score(self):
        return np.sum([r.score for r in self.relationships])


def clean_relationships(r: ArrayLike, names: list[str]) -> ArrayLike:
    couples = []
    for n in names:
        if "&" in n:
            n1, n2 = n.split("&")
            n1, n2 = n1.strip(), n2.strip()
            if " " not in n1:
                n1 = f'{n1} {n2.split(" ")[1]}'
            couples.append((n1, n2))
        else:
            couples.append((n.strip(),))

    i = 0
    for c in couples:
        if len(c) == 2:
            # Copy the relationship from the first member of the couple (person A -> person i)
            relationship = r[i, :]
            r = np.insert(r, i + 1, relationship, axis=0)
            # Copy the relationship from the first member of the couple (person i -> person A)
            relationship = r[:, i]
            r = np.insert(r, i + 1, relationship, axis=1)
            # Make sure that the first member of the couple has a high relationship score with the second member
            r[i, i + 1] = 500
            i += 2
        else:
            i += 1

    # Set relationship to self as 0
    np.fill_diagonal(r, 0)
    # Set relationship to unknown as 0
    r[np.isnan(r)] = 0.0
    # Make relationships bi-directional
    r += r.T

    new_names = flatten(couples)
    return r, new_names


def flatten(l):
    return list([item for sublist in l for item in sublist])


def main():
    csv = "seating-chart.csv"
    df: pd.DataFrame = pd.read_csv(csv, sep=",", header=None)
    names = list(df.loc[1:, 0])
    relationships = df.loc[1:, 1:].astype(float).to_numpy()

    relationships, names = clean_relationships(relationships, names)

    n_tables = 10
    table_size = 8
    init_tables = generate_tables(relationships, names, n_tables, table_size)

    n_generations = 1000
    n_children = 1000
    n_half_lives = 5
    for generation in range(n_generations):
        samples = [init_tables]
        for child in range(n_children):
            step = generation * n_children + child
            num_swaps = int(
                np.round(
                    (n_tables * table_size / 2)
                    * np.e
                    ** (-np.log(2) / (n_children * n_generations / n_half_lives) * step)
                )
            )
            indices = permute_people(init_tables, num_swaps=num_swaps)
            new_tables = generate_tables(
                relationships, names, n_tables, table_size, all_table_indices=indices
            )
            samples.append(new_tables)
        init_tables = sorted(samples, key=lambda x: sum(t.score for t in x))[-1]
        score = sum(t.score for t in init_tables)
        print(score)

    for tables in init_tables:
        print(tables.people)


def permute_people(tables: list[Table], num_swaps):
    n_people = sum([len(t.people) for t in tables])
    swaps = np.random.choice(np.arange(n_people), num_swaps * 2, replace=False)
    swaps = np.array(list(zip(swaps[0::2], swaps[1::2])))
    indices = np.array(flatten([t.indices for t in tables]))
    indices[swaps[:, 0]], indices[swaps[:, 1]] = (
        indices[swaps[:, 1]],
        indices[swaps[:, 0]],
    )
    return indices


def generate_tables(
    relationships, names, n_tables, table_size, all_table_indices: ArrayLike = None
) -> list[Table]:
    if all_table_indices is None:
        all_table_indices = np.arange(n_tables * table_size)
        np.random.shuffle(all_table_indices)

    tables = []
    for i in range(n_tables):
        t = all_table_indices[(i * table_size) : ((i + 1) * table_size)]

        i_pairs = list(itertools.combinations(t, 2))
        name_pairs = [(names[ri[0]], names[ri[1]]) for ri in i_pairs]
        score_pairs = [relationships[ri[0], ri[1]] for ri in i_pairs]

        table_relationships = [
            Relationship(pair=pair, pair_is=pair_is, score=score)
            for pair, pair_is, score in zip(name_pairs, i_pairs, score_pairs)
        ]
        table = Table(
            relationships=table_relationships, people=[names[ti] for ti in t], indices=t
        )
        tables.append(table)
    return tables


if __name__ == "__main__":
    main()
