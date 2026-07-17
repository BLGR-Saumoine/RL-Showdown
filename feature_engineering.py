from extracData import getPoolOfInfos, list_to_dict_ID, WEATHER_TO_ID, TERRAIN_TO_ID, STATUS_TO_ID, POKE_ID, MOVES_ID, ABILITIES_ID, ITEMS_ID, TYPE_TO_ID
import json
import requests
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

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

# [is_immunity, is_stat_booster, is_weather, is_priority, is_dmg_mod, is_contact_punish, is_unknown]
EXPLICIT_ABILITIES_FEATURES = {
    "unknown" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    'Slush Rush': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    'Psychic Surge': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Tinted Lens': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Defiant': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Gale Wings': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Unseen Fist': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Serene Grace': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Tablets of Ruin': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Mycelium Might': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Protean': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Drought': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Full Metal Body': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Cheek Pouch': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Huge Power': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Imposter': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Hydration': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Sap Sipper': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Insomnia': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Water Bubble': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Rattled': [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Magnet Pull': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Grim Neigh': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Ice Scales': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Early Bird': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Fluffy': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Beads of Ruin': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Shadow Tag': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Magic Guard': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Ice Face': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Competitive': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Natural Cure': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Dauntless Shield': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Dry Skin': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Weak Armor': [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Queenly Majesty': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Wind Rider': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Grassy Surge': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Inner Focus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Skill Link': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Chlorophyll': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    'Heavy Metal': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Drizzle': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Slow Start': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Bulletproof': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Guts': [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Toxic Debris': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Battle Bond': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Dragon's Maw": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Snow Warning': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Unnerve': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Zero to Hero': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Hadron Engine': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    'Adaptability': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Multitype': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Oblivious': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Overgrow': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Sturdy': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Damp': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Air Lock': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Leaf Guard': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Liquid Ooze': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Blaze': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Filter': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Dancer': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Water Veil': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Protosynthesis': [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Shadow Shield': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Water Compaction': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Toxic Chain': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Poison Touch': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Cute Charm': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Quick Feet': [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Moxie': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Unaware': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Punk Rock': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Heatproof': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Prankster': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Analytic': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'As One (Glastrier)': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Flame Body': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Levitate': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Chilling Neigh': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Sword of Ruin': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Aftermath': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Sand Stream': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Multiscale': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Electric Surge': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Poison Heal': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Shed Skin': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Strong Jaw': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Ice Body': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Tough Claws': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Scrappy': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Thick Fat': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Unburden': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Storm Drain': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Motor Drive': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Swarm': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Flower Veil': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Soul-Heart': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Illusion': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Berserk': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Seed Sower': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Aroma Veil': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Sniper': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Harvest': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Reckless': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Pickpocket': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Vessel of Ruin': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Galvanize': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Contrary': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Static': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Intimidate': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Pure Power': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Mirror Armor': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Intrepid Sword': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Hunger Switch': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Rocky Payload': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Toxic Boost': [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Sharpness': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Tangling Hair': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Frisk': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'As One (Spectrier)': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Sand Force': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    'Liquid Voice': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Iron Fist': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Anger Shell': [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "Mind's Eye": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Rock Head': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Arena Trap': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Rough Skin': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Cloud Nine': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Triage': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Light Metal': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Libero': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Regenerator': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Torrent': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Justified': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Orichalcum Pulse': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    'Lightning Rod': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Cud Chew': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Sand Rush': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    'Corrosion': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Prism Armor': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Hustle': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Magician': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Supreme Overlord': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Comatose': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Pixilate': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Surge Surfer': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Sheer Force': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Shell Armor': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Clear Body': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Bad Dreams': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Mold Breaker': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Shields Down': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Electromorphosis': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Good as Gold': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Well-Baked Body': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Flash Fire': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Technician': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Vital Spirit': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Quark Drive': [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Fur Coat': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Sticky Hold': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Compound Eyes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Power Spot': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Gooey': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Tera Shift': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Water Absorb': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Soundproof': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Infiltrator': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Limber': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Truant': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Turboblaze': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Own Tempo': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Magic Bounce': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Volt Absorb': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Stamina': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Big Pecks': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Synchronize': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'No Guard': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Download': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Swift Swim': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    'Thermal Exchange': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Earth Eater': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Keen Eye': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Cursed Body': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Overcoat': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'Pressure': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Solid Rock': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Purifying Salt': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Teravolt': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Speed Boost': [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'Gulp Missile': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Poison Puppeteer': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Transistor': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Effect Spore': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'Trace': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Stakeout': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'Disguise': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Shield Dust': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Mega Launcher': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}


def get_one_hot_type(poke_type) :
    
    typeList = list(poke_type)
    type_vector = [0.0] * 19
    for typePoke in typeList : 
        idx = TYPE_TO_ID.get(typePoke, None)
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
            move_vec = []
            info_move = response.json()

            #------------------------- Getting the base power ---------------------------------------
            if info_move["power"] is None :
                move_vec.append(0.0)
            else :
                move_vec.append(info_move["power"] / 250.0)

            #------------------------- Getting the accuracy ---------------------------------------
            if info_move["accuracy"] is None :
                move_vec.append(1.0) # If it is a status move 
            else :
                move_vec.append(info_move["accuracy"] / 100.0)

            #------------------------- Getting the category ---------------------------------------
            categorie_one_hot = MOVE_CATEGORY.get(info_move["damage_class"]["name"], [0.0, 0.0, 0.0])
            move_vec.extend(categorie_one_hot)

            #------------------------- Getting the priority bracket ---------------------------------------
            move_vec.append(info_move["priority"]/10.0)

            #------------------------- Getting the type of the move ---------------------------------------
            move_vec.extend(get_one_hot_type(info_move["type"]["name"]))

            #------------------------- Getting the description of the move ---------------------------------------
            for entry in info_move.get("flavor_text_entries", []):
                if entry["language"]["name"] == "en" and entry["version_group"]["name"] == "scarlet-violet":
                    description = entry["flavor_text"].replace('\n', ' ')
                    break

            dict_moves[id] = {
                "stats_vector": move_vec,
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

            item_vec = EXPLICIT_ITEM_FEATURES.get(item, "PROBLEM")

            description = CUSTOM_ITEM_DESCRIPTIONS.get(item, "PROBLEM")

            for entry in info_item.get("effect_entries", []):
                if entry["language"]["name"] == "en" :
                    description = entry["short_effect"].replace('\n', ' ')
                    break
                
            if description == "This Pokémon is holding an item, but its exact identity and effects are currently unknown." :
                description = CUSTOM_ITEM_DESCRIPTIONS.get(item, "PROBLEM")

            dict_items[id] = {
                "stats_vector": item_vec,
                "description": description
            }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict_items, f, indent=4, ensure_ascii=False)
    
    return bad_items


def get_abilities_json(path : str) :
    dict_abilities = {}
    bad_abilities = []

    for ability, id in ABILITIES_ID.items() :
        description = "This Pokémon's ability is not known"

        if ability == "unknown" :
            dict_abilities[id] = {
            "stats_vector" : EXPLICIT_ABILITIES_FEATURES.get(ability),
            "description": description
            }
            continue

        clean_ability = ability.replace(' ', '-')
        clean_ability = clean_ability.replace("'", '')
        clean_ability = clean_ability.replace("(", '')
        clean_ability = clean_ability.replace(")", '')
        response = requests.get(f'https://pokeapi.co/api/v2/ability/{clean_ability}/')

        if response.status_code != 200 : 
            print(clean_ability)
            bad_abilities.append((clean_ability,id))

        else :
            info_ability = response.json()

            ability_vec = EXPLICIT_ABILITIES_FEATURES.get(ability, "PROBLEM")
            
            for entry in info_ability.get("effect_entries", []):
                if entry["language"]["name"] == "en" :
                    description = entry["short_effect"].replace('\n', ' ')
                    break

            dict_abilities[id] = {
                "stats_vector": ability_vec,
                "description": description
            }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict_abilities, f, indent=4, ensure_ascii=False)
    
    return bad_abilities
  

def calc_stat(lvl : int, stat : int) :
    eff_stat = (((2 * stat + 31 + (85/4)) * lvl)/100) + 5
    return eff_stat
    
def calc_effective_stat(lvl : int, bstats : list) :
    hp, *stats = bstats
    eff_HP = ((2 * hp + 31 + (85/4)) * lvl)/100 + lvl + 10
    eff_stat = [calc_stat(lvl,x) for x in stats]
    eff_stat.insert(0, eff_HP)
    return [int(x) for x in eff_stat]


def get_poke_json(path : str) :
    dict_poke = {}
    bad_poke = []

    for poke, id in POKE_ID.items() :

        if poke == "unknown" :
            dict_poke[id] = {
            # One hot encoding type (19), weight, Base stats
            "stats_vector" : []
            }
            continue

        response = requests.get(f'https://pokeapi.co/api/v2/pokemon/{poke}/')

        if response.status_code != 200 : 
            print(poke)
            bad_poke.append((poke,id))

        else :
            poke_vec = []

            info_poke = response.json()

            for ele in info_poke['types'] : 
                temp = ['type']['name']


            poke_vec.append(info_poke["weight"])

            bstat = []
            for ele in info_poke["stats"] :
                bstat.append(ele["base_stat"])

            with open('sets.json') as f:
                temp = json.load(f)
            level = temp["abomasnow"]["level"]
            
            dict_poke[id] = {
                "stats_vector": poke_vec,
            }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict_poke, f, indent=4, ensure_ascii=False)
    
    return bad_poke



def get_pickle_FE(path_json : str, path_to_save : str) :
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(path_json) as f:
        dico_json = json.load(f)
    pickle_dict = {}
    for key, items in dico_json.items() :
        desc = items["description"]
        desc_emb = model.encode(desc).astype(np.float32)
        stat_vect = np.array(items["stats_vector"], dtype=np.float32)
        pickle_dict[key] = {
            "stats_vector": stat_vect,
            "description_embedding": desc_emb
        }
    with open(path_to_save, 'wb') as f:
            pickle.dump(pickle_dict, f)








    




        




