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


def main():
    csv = "seating-chart.csv"
    df: pd.DataFrame = pd.read_csv(csv, sep=",", header=None)
    names = list(df.loc[1:, 0])
    relationships = df.loc[1:, 1:].astype(float).to_numpy()

    relationships, names = clean_relationships(relationships, names)

    best_score = 0
    best_table = None

    for i in range(100000):
        tables = generate_tables(relationships, names, n_tables=10, table_size=8)
        score = np.sum([np.sum([r.score for r in t]) for t in tables])
        if score > best_score:
            best_table = tables
            best_score = score

    print(best_score)
    for t in best_table:
        print(t)

def generate_tables(relationships, names, n_tables, table_size):
    all_table_indices = np.arange(n_tables * table_size)
    np.random.shuffle(all_table_indices)
    tables = []
    for i in range(n_tables):
        t = all_table_indices[(i * table_size):((i + 1) * table_size)]

        pairs = list(itertools.combinations(t, 2))
        score_pairs = [relationships[ri[0], ri[1]] for ri in pairs]
        name_pairs = [(names[ri[0]], names[ri[1]]) for ri in pairs]

        table_relationships = [
            Relationship(name, idxs, s)
            for name, idxs, s in zip(name_pairs, pairs, score_pairs)
        ]
        tables.append(table_relationships)
    return tables


if __name__ == "__main__":
    main()
