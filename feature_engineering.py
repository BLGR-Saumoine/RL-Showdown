from extracData import getPoolOfInfos, list_to_dict_ID, WEATHER_TO_ID, TERRAIN_TO_ID, STATUS_TO_ID, POKE_ID, MOVES_ID, ABILITIES_ID, ITEMS_ID, TYPE_TO_ID
import json
import requests

MOVE_CATEGORY = {
    "unknown" : [0.0, 0.0, 0.0],
    "physical" : [1.0, 0.0, 0.0],
    "special" : [0.0, 1.0, 0.0],
    "status" : [0.0, 0.0, 1.0],
    }

# [is_choice, is_consumable, passive_healing, self_harm_or_status, unremovable_or_form_change, is_unknown]

EXPLICIT_ITEM_FEATURES = {
    # --- Special case ---
    'unknown':           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Only one unknown (that's kinda the point)
    'No-item':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    # --- Choice items ---
    'Choice Specs':      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Choice Scarf':      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Choice Band':       [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    # --- One use item ---
    'Lum Berry':         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Leppa Berry':       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Custap Berry':      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Sitrus Berry':      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Chesto Berry':      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Focus Sash':        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'White Herb':        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Power Herb':        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Throat Spray':      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Air Balloon':       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], #Not sure if it fit here i will have to see
    'Weakness Policy':   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Booster Energy':    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],

    # --- Passive healing ---
    'Leftovers':         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],

    # --- Recoil or inflict status ---
    'Life Orb':          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Flame Orb':         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Toxic Orb':         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],

    # --- Not removable / Form Changing ---
    'Griseous Core':     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Rusted Shield':     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Rusted Sword':      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Adamant Crystal':   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Lustrous Globe':    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Hearthflame Mask':  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Wellspring Mask':   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Cornerstone Mask':  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # Arceus plates
    'Meadow Plate':      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Icicle Plate':      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Dread Plate':       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Spooky Plate':      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Toxic Plate':       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Fist Plate':        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Earth Plate':       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Iron Plate':        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Insect Plate':      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Mind Plate':        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Zap Plate':         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Stone Plate':       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Pixie Plate':       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Flame Plate':       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Splash Plate':      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Sky Plate':         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Draco Plate':       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],

    # --- Standards object -> infos will come from embeddings ---
    'Loaded Dice':       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Heavy-Duty Boots':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Wide Lens':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Silk Scarf':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Lustrous Orb':      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Expert Belt':       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Magnet':            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Light Ball':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Assault Vest':      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Silver Powder':     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Eviolite':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Rocky Helmet':      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Light Clay':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Soul Dew':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Scope Lens':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

CUSTOM_ITEM_DESCRIPTIONS = {
    # --- Objets Méta Récents (Génération 8 & 9) ---
    "Booster Energy": "Held: Activates Protosynthesis or Quark Drive abilities upon entry. Consumed after use.",
    "Loaded Dice": "Held: Multi-hit moves hit more consistently, guaranteeing at least 4 hits for standard 2-5 hit moves.",
    "Heavy-Duty Boots": "Held: The holder is completely immune to all entry hazard damage and effects (Stealth Rock, Spikes, Toxic Spikes, Sticky Web).",
    "Throat Spray": "Held: Raises the holder's Special Attack by 1 stage after it uses a sound-based move. Consumed after use.",

    # --- Objets de Forme Ogerpon (Génération 9) ---
    "Wellspring Mask": "Held by Ogerpon. Changes form to Wellspring Mask. Boosts Water-type moves by 20% and cannot be knocked off. Turns Ivy Cudgel into a Water-type move.",
    "Hearthflame Mask": "Held by Ogerpon. Changes form to Hearthflame Mask. Boosts Fire-type moves by 20% and cannot be knocked off. Turns Ivy Cudgel into a Fire-type move.",
    "Cornerstone Mask": "Held by Ogerpon. Changes form to Cornerstone Mask. Boosts Rock-type moves by 20% and cannot be knocked off. Turns Ivy Cudgel into a Rock-type move.",

    # --- Objets de Forme Légendaires (Génération 8 Origin/Crowned) ---
    "Rusted Sword": "Held by Zacian. Changes form to Crowned Sword. Cannot be knocked off or tricked. Changes Iron Head into Behemoth Blade.",
    "Rusted Shield": "Held by Zamazenta. Changes form to Crowned Shield. Cannot be knocked off or tricked. Changes Iron Head into Behemoth Bash.",
    "Adamant Crystal": "Held by Dialga. Changes form to Origin Form. Boosts Steel-type and Dragon-type moves by 20%.",
    "Lustrous Globe": "Held by Palkia. Changes form to Origin Form. Boosts Water-type and Dragon-type moves by 20%.",
    "Griseous Core": "Held by Giratina. Changes form to Origin Form. Boosts Ghost-type and Dragon-type moves by 20%."
}


def get_one_hot_type(poke_type : str) :
    type_vector = [0.0] * 19
    idx = TYPE_TO_ID.get(poke_type, None)
    if idx is not None :
        type_vector[idx] = 1.0
    return type_vector


def get_moves_json(path : str) :
    dict_moves = {}
    bad_moves = []

    for move,id in MOVES_ID.items() :

        description = "No description available."

        if move == "unknown" :
            dict_moves[id] = {
                "stats_vector" :[
                0.0, # Base power
                0.0, # Accuracy
                0.0, 0.0, 0.0, # Category 
                0.0, # Priority bracket
                *get_one_hot_type("unknown") # One hot encoding of the type (see TYPE_TO_ID) 
            ],
            "description": description
            }
            continue

        clean_move = move.replace(' ', '-')
        response = requests.get(f'https://pokeapi.co/api/v2/move/{clean_move}/')

        if response.status_code != 200 : 

            bad_moves.append((move,id))

        else :
            move_desc = []
            info_move = response.json()

            #------------------------- Getting the base power ---------------------------------------
            if info_move["power"] is None :
                move_desc.append(0.0)
            else :
                move_desc.append(info_move["power"] / 250.0)

            #------------------------- Getting the accuracy ---------------------------------------
            if info_move["accuracy"] is None :
                move_desc.append(1.0) # If it is a status move 
            else :
                move_desc.append(info_move["accuracy"] / 100.0)

            #------------------------- Getting the category ---------------------------------------
            categorie_one_hot = MOVE_CATEGORY.get(info_move["damage_class"]["name"], [0.0, 0.0, 0.0])
            move_desc.extend(categorie_one_hot)

            #------------------------- Getting the priority bracket ---------------------------------------
            move_desc.append(info_move["priority"]/10.0)

            #------------------------- Getting the type of the move ---------------------------------------
            move_desc.extend(get_one_hot_type(info_move["type"]["name"]))

            #------------------------- Getting the description of the move ---------------------------------------
            for entry in info_move.get("flavor_text_entries", []):
                if entry["language"]["name"] == "en" and entry["version_group"]["name"] == "scarlet-violet":
                    description = entry["flavor_text"].replace('\n', ' ')
                    break

            dict_moves[id] = {
                "stats_vector": move_desc,
                "description": description
            }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict_moves, f, indent=4, ensure_ascii=False)
    
    return bad_moves


def get_items_json(path : str) :
    dict_items = {}
    bad_items = []

    for item, id in ITEMS_ID.items() :
        description = "This Pokémon is holding an item, but its exact identity and effects are currently unknown."

        if item == "unknown" :
            dict_items[id] = {
                "stats_vector" : EXPLICIT_ITEM_FEATURES.get(item),
            "description": description
            }
            continue

        if item == "No-item" :
            dict_items[id] = {
                "stats_vector" : EXPLICIT_ITEM_FEATURES.get(item),
            "description": "This Pokémon is not holding any item. It has no passive item effects."
            }
            continue

        clean_item = item.replace(' ', '-')
        response = requests.get(f'https://pokeapi.co/api/v2/item/{clean_item}/')

        if response.status_code != 200 : 

            bad_items.append((clean_item,id))

        else :
            info_item = response.json()

            move_desc = EXPLICIT_ITEM_FEATURES.get(item, "PROBLEM")

            description = CUSTOM_ITEM_DESCRIPTIONS.get(item, "PROBLEM")

            for entry in info_item.get("effect_entries", []):
                if entry["language"]["name"] == "en" :
                    description = entry["short_effect"].replace('\n', ' ')
                    break
                
            if description == "This Pokémon is holding an item, but its exact identity and effects are currently unknown." :
                description = CUSTOM_ITEM_DESCRIPTIONS.get(item, "PROBLEM")

            dict_items[id] = {
                "stats_vector": move_desc,
                "description": description
            }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict_items, f, indent=4, ensure_ascii=False)
    
    return bad_items
        



    




        




