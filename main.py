import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class Relationship:
    people: (str, str)
    indices: (int, int)
    score: int


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

    new_names = [item for sublist in couples for item in sublist]
    return r, new_names


if __name__ == "__main__":
    csv = "seating-chart.csv"
    df: pd.DataFrame = pd.read_csv(csv, sep=",", header=None)
    names = list(df.loc[1:, 0])
    r = df.loc[1:, 1:].astype(float).to_numpy()

    r, names = clean_relationships(r, names)

    table_size = 8
    n_tables = 10

    all_table_indices = np.arange(n_tables * table_size)
    # np.random.shuffle(indices)
    tables = []
    for i in range(n_tables):
        start = i * table_size
        end = (i + 1) * table_size
        table_indices = all_table_indices[start:end]
        tables.append(table_indices)

    for t in tables:
        all_relationship_indices = list(itertools.combinations(t, 2))
        scores = [r[ri[0], ri[1]] for ri in all_relationship_indices]
        n = [(names[ri[0]], names[ri[1]]) for ri in all_relationship_indices]
        table_relationships = [
            Relationship(name, idxs, s)
            for name, idxs, s in zip(n, all_relationship_indices, scores)
        ]
