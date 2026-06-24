import json
from typing import Dict

global WEATHER_TO_ID, TERRAIN_TO_ID, STATUS_TO_ID, POKE_ID, MOVES_ID, ABILITIES_ID, ITEMS_ID

WEATHER_TO_ID = {
    "NONE": 0,     # Ciel dégagé
    "SUN": 1,      # Zénith
    "RAIN": 2,     # Danse Pluie
    "SAND": 3,     # Tempête de sable
    "HAIL": 4,     # Grêle
    "SNOW": 4      # Neige (Gen 9 remplace la Grêle)
}

TERRAIN_TO_ID  = {
    "NONE": 0,
    "ELECTRIC": 1,
    "GRASSY": 2,
    "MISTY": 3,
    "PSYCHIC": 4
}

STATUS_TO_ID  = {
    "NONE": 0,
    "BRN": 1,  # Brûlure
    "PAR": 2,  # Paralysie
    "SLP": 3,  # Sommeil
    "FRZ": 4,  # Gel
    "PSN": 5,  # Poison
    "TOX": 6,  # Mauvais Poison (Toxic)
    "FNT": 7   # K.O. (Fainted)
}

TYPE_TO_ID: Dict[str, int] = {
    "normal": 0,   "fire": 1,     "water": 2,  "electric": 3,
    "grass": 4,    "ice": 5,      "fighting": 6, "poison": 7,
    "ground": 8,   "flying": 9,   "psychic": 10, "bug": 11,
    "rock": 12,    "ghost": 13,   "dragon": 14,  "dark": 15,
    "steel": 16,   "fairy": 17,   "stellar": 18
}

def getPoolOfInfos(dico) :
    """
    Dict : Dictionnary with the following format
    {'Abomasnow': {'level': 84,
    'abilities': ['Snow Warning'],
    'items': ['Light Clay'],
    'roles': {'Bulky Support': {'abilities': ['Snow Warning'],
        'items': ['Light Clay'],
        'teraTypes': ['Ghost', 'Water'],
        'moves': ['Aurora Veil',
        'Blizzard',
        'Earthquake',
        'Ice Shard',
        'Wood Hammer'],
        'evs': {'hp': 77}}},
    'evs': {'hp': 77}},

    The goal is to get the name and attacks that are present in the sets of random battle, for example larvitar is not available in this game mode and by filtering it out it 
    allows us to gain computational space
    """
    pokemons = [*dico]
    moves = set()
    items = set()
    abilities = set()
    for poke in dico.values() :
        items.update(poke.get("items", []))
        abilities.update(poke.get("abilities", []))
        for sets in poke["roles"].values() :
            moves.update(sets.get('moves',[]))

    return pokemons, list(moves), list(items), list(abilities)


def list_to_dict_ID(listofItem) :
    dico = {}
    dico["unknown"] = 0
    it = 1
    for item in listofItem :
        dico[item] = it
        it += 1
    return dico

with open('gen9randombattle.json') as f:
    dico = json.load(f)

pokemons, moves, items, abilities = getPoolOfInfos(dico)

items.append("No-item")

POKE_ID  = list_to_dict_ID(pokemons)
MOVES_ID  = list_to_dict_ID(moves)
ITEMS_ID  = list_to_dict_ID(items)
ABILITIES_ID  = list_to_dict_ID(abilities)
