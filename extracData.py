import json
from typing import Dict
import requests
import re



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
    "steel": 16,   "fairy": 17,   "stellar": 18, "unknown" : 19
}

#Certains noms des pokemons des sets ne sont pas ceux attendus pas pokeAPI -> enamorus (pour la forme de base) au lieu de enamorus-incarnate pour pokeAPI
POKE_API_MAPPING = {
    # --- Formes de base implicites (PokeAPI exige le nom complet de la forme) ---
    "Basculin": "basculin-red-striped",
    "Deoxys": "deoxys-normal",
    "Enamorus": "enamorus-incarnate",
    "Giratina": "giratina-altered",
    "Landorus": "landorus-incarnate",
    "Lycanroc": "lycanroc-midday",
    "Meloetta": "meloetta-aria",
    "Mimikyu": "mimikyu-disguised",
    "Shaymin": "shaymin-land",
    "Thundurus": "thundurus-incarnate",
    "Tornadus": "tornadus-incarnate",
    "Toxtricity": "toxtricity-amped",
    "Urshifu": "urshifu-single-strike",
    
    # --- Pokémon avec des différences de genre codées en dur ---
    "Basculegion": "basculegion-male",
    "Basculegion-F": "basculegion-female",
    "Indeedee": "indeedee-male",
    "Indeedee-F": "indeedee-female",
    "Meowstic": "meowstic-male",
    "Meowstic-F": "meowstic-female",
    "Oinkologne": "oinkologne-male",
    "Oinkologne-F": "oinkologne-female",
    
    # --- Formes spécifiques et esthétiques ---
    "Dudunsparce": "dudunsparce-two-segment",
    "Eiscue": "eiscue-ice",
    "Greninja-Bond": "greninja-battle-bond",
    "Maushold": "maushold-family-of-four",
    "Minior": "minior-red-meteor",
    "Morpeko": "morpeko-full-belly",
    "Necrozma-Dawn-Wings": "necrozma-dawn",
    "Necrozma-Dusk-Mane": "necrozma-dusk",
    "Oricorio": "oricorio-baile",
    "Oricorio-Pa'u": "oricorio-pau",
    "Palafin": "palafin-zero",
    "Squawkabilly": "squawkabilly-green-plumage",
    "Squawkabilly-Blue": "squawkabilly-blue-plumage",
    "Squawkabilly-White": "squawkabilly-white-plumage",
    "Squawkabilly-Yellow": "squawkabilly-yellow-plumage",
    "Tatsugiri": "tatsugiri-curly",
    
    # --- Les Tauros de Paldea ---
    "Tauros-Paldea-Aqua": "tauros-paldea-aqua-breed",
    "Tauros-Paldea-Blaze": "tauros-paldea-blaze-breed",
    "Tauros-Paldea-Combat": "tauros-paldea-combat-breed",
    
    # --- Les Arceus ---
    # PokeAPI n'a pas les stats par type, donc on pointe tout sur le modèle de base (qui a 120 partout).
    # RAPPEL : Dans ton code de téléchargement, n'oublie pas le hack pour écraser le type 
    # API ("Normal") par le type contenu dans le raw_name !
    "Arceus-Bug": "arceus",
    "Arceus-Dark": "arceus",
    "Arceus-Dragon": "arceus",
    "Arceus-Electric": "arceus",
    "Arceus-Fairy": "arceus",
    "Arceus-Fighting": "arceus",
    "Arceus-Fire": "arceus",
    "Arceus-Flying": "arceus",
    "Arceus-Ghost": "arceus",
    "Arceus-Grass": "arceus",
    "Arceus-Ground": "arceus",
    "Arceus-Ice": "arceus",
    "Arceus-Poison": "arceus",
    "Arceus-Psychic": "arceus",
    "Arceus-Rock": "arceus",
    "Arceus-Steel": "arceus",
    "Arceus-Water": "arceus",
}


def to_showdown_id(name: str) -> str:
    # Met en minuscules et garde uniquement les lettres et les chiffres
    return re.sub(r'[^a-z0-9]', '', name.lower())

    # Exemple : "Oricorio-Pa'u" -> "oricoriopau"
    # Exemple : "Iron Valiant" -> "ironvaliant"

def to_api_id(name: str) -> str:
    res = name.lower().replace("'","")
    res = res.replace("(", '')
    res = res.replace(")", '')
    return res.replace(' ', '-')

    # Exemple : "Oricorio-Pa'u" -> "oricorio-pau"
    # Exemple : "Iron Valiant" -> "iron-valiant"

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


def list_to_dict_ID(listofItem, POKE_API_MAPPING = None) :
    dico = {}

    if POKE_API_MAPPING :
        dico[0] = {
        "raw_name": "unknown",
        "showdown_id": "unknown",
        "api_name": "unknown"}
        it = 1
        for item in listofItem :
            dico[it] = {
                "raw_name": item,
                "showdown_id": to_showdown_id(item),
                "api_name": POKE_API_MAPPING.get(item, to_api_id(item))
            }
            it += 1
    else :
        dico[0] = {
                "raw_name": "unknown",
                "showdown_id": "unknown",
                "api_name": "unknown"}
        it = 1
        for item in listofItem :
            dico[it] = {
                "raw_name": item,
                "showdown_id": to_showdown_id(item),
                "api_name": to_api_id(item)
            }
            it += 1
    return dico


url = "https://pkmn.github.io/randbats/data/gen9randombattle.json"

response = requests.get(url)

data = response.json()

with open("current_sets.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Pokemon's set is up to date !")

with open('current_sets.json') as f:
    dico = json.load(f)

pokemons, moves, items, abilities = getPoolOfInfos(dico)

items.append("No-item")

POKE_ID  = list_to_dict_ID(pokemons)
MOVES_ID  = list_to_dict_ID(moves)
ITEMS_ID  = list_to_dict_ID(items)
ABILITIES_ID  = list_to_dict_ID(abilities)
