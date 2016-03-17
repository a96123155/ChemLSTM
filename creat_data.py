#!/usr/bin/env python2.7
# coding=utf-8

from __future__ import print_function
import pymongo
import sys
import json
import random

def reaction_exists(c1, c2, out=None):
    if out is None:
        return db.reactions.find({"reactants": {"$all": [c1, c2]}}, {"reactants": 1, "products": 1}).count() > 0
    else:
        return db.reactions.find({"reactants": {"$all": [c1, c2]}, "products": {"$all": [out]}}, {"reactants": 1, "products": 1}).count() > 0

def deref(chemical):
    found = db.chemicals.find({"_id": chemical})
    for chemical in found:
        names = chemical["name"]
        if names is not None:
            return names[0]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} [# chemicals] [# reactions]".format(sys.argv[0]))
        quit(1)

    N = int(sys.argv[1])
    M = int(sys.argv[2])

    client = pymongo.MongoClient("localhost", 27017)
    db = client.chemnet

    # Produce M reactions and chemical ids/names required for them
    chem_ids = {}

    reactions = []

    # Create positive reactions
    for reaction in db.reactions.find({"reactants": {"$size": 2}, "products": {"$size": 1}}, {"reactants": 1, "products": 1}, limit=M):
        if len(reactions) >= M / 2:
            break

        chem1, chem2 = reaction["reactants"]
        if chem1 is None or chem2 is None:
            continue

        chem1name = deref(chem1)
        chem2name = deref(chem2)
        if chem1name is None or chem2name is None:
            continue

        prod = reaction["products"][0]
        if prod is None:
            continue

        prod_name = deref(prod)
        if prod_name is None:
            continue

        if chem1 not in chem_ids:
            chem_ids[chem1] = chem1name

        if chem2 not in chem_ids:
            chem_ids[chem2] = chem2name

        if prod not in chem_ids:
            chem_ids[prod] = prod_name

        # don't count the product name towards chemical ids
        reactions.append([chem1name, chem2name, prod_name])

    print("{} positive examples ({} chemicals)".format(len(reactions), len(chem_ids)))

    negative_examples = 0
    while negative_examples < (M - (M/2)):
        c1, c2 = random.sample(chem_ids.keys(), 2)
        if not reaction_exists(c1, c2):
            c1name = chem_ids[c1]
            c2name = chem_ids[c2]
            reactions.append([c1name, c2name, None])
            negative_examples += 1

    # Dump out reactions
    with open("train_reactions_with_outputs_{}.json".format(M), "w") as f:
        json.dump(reactions, f)

    print(len(reactions), "reactions dumped")

    quit(0)

    # Pad out the chemical ids to N
    chemicals_cursor = db.chemicals.find(limit=2*N)
    i = 0
    while len(chem_ids) < N and i < chemicals_cursor.count():
        chemical = chemicals_cursor[i]
        i += 1

        chem1 = chemical["_id"]
        if chem1 in chem_ids:
            continue

        chem1name = deref(chem1)
        if chem1name is None:
            continue

        chem_ids[chem1] = chem1name

    # Dump out chemicals
    with open("train_chemicals_{}.txt".format(N), "w") as f:
        f.write("\n".join(chem_ids.values()))

    print(len(chem_ids), "chemicals dumped")
