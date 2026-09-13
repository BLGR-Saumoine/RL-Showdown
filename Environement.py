import numpy as np
import gymnasium as gym
from gymnasium import spaces

from poke_env.battle import Battle
from poke_env.player import Gen9EnvSinglePlayer, LocalhostServerConfiguration

from extracData import POKE_ID, MOVES_ID, ABILITIES_ID, ITEMS_ID, TYPE_TO_ID, WEATHER_TO_ID, TERRAIN_TO_ID, STATUS_TO_ID

class PokemonRandbatsEnv(Gen9EnvSinglePlayer):
    
    def __init__(self, **kwargs):
        #Connecting to local pkmn server
        if "server_configuration" not in kwargs:
            kwargs["server_configuration"] = LocalhostServerConfiguration

        super().__init__(**kwargs)
        
        self.action_space = spaces.Discrete(14) # 4 Attaques + 4 Attaques mais en terracristalisant + 5 pokemons à switcher + 1 cas autre au cas où

        self.observation_space = spaces.Box(low=-6.0, high=1000.0, shape=(138,), dtype=np.float32)

        """
        # =========================================================================
        # OBSERVATION SPACE : gym.spaces.Box(shape=(164,))
        # Vecteur 1D représentant l'état du jeu à un instant T pour l'agent.
        # Les valeurs "inconnues" (surtout pour l'adversaire) sont initialisées à 0.
        # =========================================================================

        # --- BLOC A : Le Pokémon Actif du Joueur (18 dimensions) ---
        # 1.  ID du Pokémon (Entier)
        # 2.  PV restants (Flottant, idéalement normalisé de 0.0 à 1.0)
        # 3.  ID du Statut (0=None, 1=BRN, 2=PAR, etc.)
        # 4.  Modificateur Attaque (-6 à +6)
        # 5.  Modificateur Défense (-6 à +6)
        # 6.  Modificateur Attaque Spéciale (-6 à +6)
        # 7.  Modificateur Défense Spéciale (-6 à +6)
        # 8.  Modificateur Vitesse (-6 à +6)
        # 9.  Modificateur Précision (-6 à +6)
        # 10. Modificateur Esquive (-6 à +6)
        # 11. ID Attaque 1
        # 12. ID Attaque 2
        # 13. ID Attaque 3
        # 14. ID Attaque 4
        # 15. ID Objet (Item)
        # 16. ID Talent (Ability)
        # 17. Est Téra-cristallisé (0 = Non, 1 = Oui)
        # 18. ID du Type Téra (0 si non cristallisé)

        # --- BLOC B : L'Équipe en Réserve du Joueur (5 * 11 = 55 dimensions) ---
        # Pour chacun des 5 Pokémon en attente (Slots 1 à 5) :
        # 1.  ID du Pokémon
        # 2.  PV restants (0.0 à 1.0, 0.0 signifiant K.O.)
        # 3.  ID du Statut
        # 4.  ID Attaque 1
        # 5.  ID Attaque 2
        # 6.  ID Attaque 3
        # 7.  ID Attaque 4
        # 8.  ID Objet
        # 9.  ID Talent
        # 10. Est Téra-cristallisé (0 = Non, 1 = Oui)
        # 11. ID du Type Téra (0 si non cristallisé)

        # --- BLOC C : Le Pokémon Actif Adverse (18 dimensions) ---
        # (Structure identique au Bloc A. L'information cachée vaut 0)
        # 1.  ID du Pokémon adverse
        # 2.  Pourcentage de PV visibles (0.0 à 1.0)
        # 3.  ID du Statut
        # 4-10. Modificateurs de stats visibles (-6 à +6)
        # 11-14. IDs des attaques (0 si non révélée)
        # 15. ID Objet (0 si non révélé/déclenché)
        # 16. ID Talent (0 si non révélé/déclenché)
        # 17. Est Téra-cristallisé (0 = Non, 1 = Oui)
        # 18. ID du Type Téra (0 si non cristallisé)

        # --- BLOC D : Le Terrain & Les Effets Globaux (18 dimensions) ---
        # 1.  ID de la Météo (0=None, 1=Sun, 2=Rain, etc.)
        # 2.  ID du Terrain (0=None, 1=Electric, 2=Grassy, etc.)
        # Effets du côté du JOUEUR :
        # 3.  Picots / Spikes (0, 1, 2, ou 3 rangées)
        # 4.  Pics Toxik / Toxic Spikes (0, 1, ou 2 rangées)
        # 5.  Piège de Roc / Stealth Rock (0 ou 1)
        # 6.  Toile Gluante / Sticky Web (0 ou 1)
        # 7.  Voile Aurore / Aurora Veil (0 ou 1)
        # 8.  Mur Lumière / Light Screen (0 ou 1)
        # 9.  Protection / Reflect (0 ou 1)
        # Effets du côté de l'ADVERSAIRE :
        # 10. Picots / Spikes (0 à 3)
        # 11. Pics Toxik / Toxic Spikes (0 à 2)
        # 12. Piège de Roc / Stealth Rock (0 ou 1)
        # 13. Toile Gluante / Sticky Web (0 ou 1)
        # 14. Voile Aurore / Aurora Veil (0 ou 1)
        # 15. Mur Lumière / Light Screen (0 ou 1)
        # 16. Protection / Reflect (0 ou 1)
        # Droits Téra restants :
        # 17. Droit au Téra Joueur (0 = Utilisé, 1 = Disponible)
        # 18. Droit au Téra Adversaire (0 = Utilisé, 1 = Disponible)

        # --- BLOC E : L'Équipe en Réserve Adverse [MÉMOIRE] (5 * 11 = 55 dimensions) ---
        # (Structure identique au Bloc B. Initialisé à 0. Se remplit 
        # progressivement lorsque l'adversaire révèle son équipe).
        # Pour chaque emplacement adverse en attente (Slots 1 à 5) :
        # 1.  ID du Pokémon (0 si inconnu)
        # 2.  PV restants vus lors de son dernier switch (0.0 à 1.0)
        # 3.  ID du Statut (0 si aucun/inconnu)
        # 4-7. IDs des attaques (0 si non révélée)
        # 8.  ID Objet (0 si inconnu)
        # 9.  ID Talent (0 si inconnu)
        # 10. Est Téra-cristallisé (0 = Non, 1 = Oui)
        # 11. ID du Type Téra (0 si non cristallisé)

        # TOTAL = 18 + 55 + 18 + 18 + 55 = 164 dimensions.
        """

    def embed_battle(self, battle: Battle):
        """
        L'EXTRACTEUR D'ÉTAT.
        Organisé suivant les 5 Blocs décris dans le __init__
        """
        state_vec = []


    



        
        # Test rapide :
        # print("Pokémon actif adverse :", battle.opponent_active_pokemon)
        # print("Équipe adverse révélée :", battle.opponent_team)
        
        # ... Ta logique ici ...
        
        # Pour éviter que le code ne plante en attendant que tu finisses, 
        # on retourne des zéros de la bonne taille.
        return np.zeros(138, dtype=np.float32)


    def step(self, action):
        # 3. Exécution d'un tour de jeu
        ...
        return observation, reward, terminated, truncated, info



    def bloc_A_C(self, battle: Battle, state_vec: list):
        active = battle.active_pokemon

        # Sécurité absolue : si pas de Pokémon sur le terrain (suite à un K.O) je regarde apres
        if active is None:
            state_vec.extend([0.0] * 18)
            return

        # 1. ID du Pokémon (Entier)
        state_vec.append(POKE_ID.get(active.species, 0))
        
        # 2. PV restants (Déjà un flottant 0.0 - 1.0)
        state_vec.append(active.current_hp_fraction)
        
        # 3. ID du Statut
        status_name = active.status.name if active.status is not None else "NONE"
        state_vec.append(STATUS_TO_ID.get(status_name, 0))

        # 4 à 10. Modificateurs de stats (-6 à +6)
        boosts = active.boosts
        state_vec.extend([
            boosts.get("atk", 0),
            boosts.get("def", 0),
            boosts.get("spa", 0),
            boosts.get("spd", 0),
            boosts.get("spe", 0),
            boosts.get("accuracy", 0),
            boosts.get("evasion", 0)
        ])

        # 11-14. Attaques 
        for move in active.moves.keys() :
            state_vec.append(MOVES_ID.get(move, 0))
        # 15. Objet 
        state_vec.append(ITEMS_ID.get(active.item, 0))
        # 16. Talent 
        state_vec.append(ABILITIES_ID.get(active.ability, 0))
        # 17 is Téra ?
        state_vec.append(active.is_terastallized())
        # 18 what Téra ?
        tera_type = active.tera_type if active.tera_type is not None else "unknown"
        state_vec.append(TYPE_TO_ID.get(tera_type, [0.0]*19)) # Le [0.0] * 19 correspond à la value de unknown


    