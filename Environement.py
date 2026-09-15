import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import torch

from poke_env.battle import Battle
from poke_env.player import Gen9EnvSinglePlayer, LocalhostServerConfiguration

from extracData import POKE_ID, MOVES_ID, ABILITIES_ID, ITEMS_ID, TYPE_TO_ID, WEATHER_TO_ID, TERRAIN_TO_ID, STATUS_TO_ID

class PokemonRandbatsEnv(Gen9EnvSinglePlayer):
    
    def __init__(self, **kwargs):
        #Connecting to local pkmn server
        if "server_configuration" not in kwargs:
            kwargs["server_configuration"] = LocalhostServerConfiguration

        super().__init__(**kwargs)

        self.training_mode = "random"
        
        self.action_space = spaces.Discrete(14) # 4 Attaques + 4 Attaques mais en terracristalisant + 5 pokemons à switcher + 1 cas autre au cas où

        self.observation_space = spaces.Box(low=-6.0, high=1000.0, shape=(380,), dtype=np.float32)

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

        #Info sur le pokemon actif du joueur 
        self.bloc_A_C(battle.active_pokemon, state_vec)

        #Info sur les pokemons en reserves du joueur
        player_reserve = [mon for mon in battle.team.values() if not mon.active]
        self.bloc_B_E(player_reserve, state_vec)

        #Info sur le pokemon actif adverse 
        self.bloc_A_C(battle.opponent_active_pokemon, state_vec)

        #Info de terrain 
        self.bloc_D(battle, state_vec)

        #Info sur les pokemons en reserves adverse
        oppo_reserve = [mon for mon in battle.opponent_team.values() if not mon.active]
        self.bloc_B_E(oppo_reserve, state_vec)

        if len(state_vec) != 380 :
            return np.zeros(380, dtype=np.float32)
        else : 
            return np.array(state_vec, dtype=np.float32)

    def action_to_order(self, action, battle : Battle):
        """
        Returns the BattleOrder relative to the given action.
        The action mapping is as follows: action = -2: default 
        action = -1: forfeit 
        0 <= action <= 5: switch 
        6 <= action <= 9: move 
        10 <= action <= 13: move and mega evolve Inutile ici
        14 <= action <= 17: move and z-move Inutile ici
        18 <= action <= 21: move and dynamax Inutile ici
        22 <= action <= 25: move and terastallize
        """
        poke_env_action = 0
        if 0 <= action <= 3 : # Attaquer simplement
            poke_env_action = action + 6
        elif 4 <= action <= 7 : # Attaquer en teracrystalisant
            poke_env_action = action + 18
        elif 8 <= action <= 13 : #Switcher
            poke_env_action = action - 8
        return self.action_to_order(poke_env_action, battle)

    def calc_reward(self, last_turn, current_turn,) -> float:
        # Plusieurs mode d'entrainement et reward calculées différemment en fonction du nombre d'adversaires
        reward = 0.0

        if self.training_mode == "random" :
            reward = self.rewardVSrandom(last_turn, current_turn)

        return reward

    def select_action(self, battle : Battle, vec_state):
        """
        Espace d'action :
        0-3 : Choisir un attaque parmis les 4 disponibles
        4-7 : Choisir une attaque et teracrystaliser
        8-13 : Switcher de pokemon parmis les 6 (masque d'attention par défaut sur le poke actif)

        Mon objectif est de conçevoir un mask d'attention
        """
        mask = np.ones(14, dtype=np.float32)

        # 0-3
        active_moves = list(battle.active_pokemon.moves.values())
        
        for i, move in enumerate(active_moves):
            # Si l'attaque est dans la liste des cliquables renvoyée par le serveur
            if move in battle.available_moves:
                mask[i] = 0.0

        #4-7 : pour le tera c'est très facile 
        if battle.can_tera :
            mask[4:8] = mask[0:4] 

        # 8-13 les switches

        for i, mon in enumerate(battle.team.values()):
            # Si le poke est switchable
            if mon in battle.available_switches:
                mask[i+8] = 0.0

        epsilon = self.epsEnd + (self.epsStart - self.epsEnd) * math.exp(-self.numberStep/self.epsDecay)

        if torch.rand(1).item() < epsilon :
            possible_action = np.where(mask==0.0)[0]
            action = np.random.choice(possible_action)

        else :
            with torch.no_grad():
                mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
                gpu_state = torch.tensor(vec_state).to(self.device, dtype=torch.float32)
                action = torch.argmax(self.onlineNetwork(gpu_state).masked_fill(mask_tensor, -1e9)).item()
        self.numberStep += 1
        return action

    def bloc_A_C(self, pokemon, state_vec: list):
        """
        Décris les pokemons actifs sur le terrain
        """

        # Sécurité absolue : si pas de Pokémon sur le terrain (suite à un K.O) je regarde apres
        if pokemon is None:
            state_vec.extend([0.0] * 36) # checker les dims c'est le bazar là
            return

        # 1. ID du Pokémon (Entier)
        state_vec.append(POKE_ID.get(pokemon.species, 0))
        
        # 2. PV restants
        state_vec.append(pokemon.current_hp_fraction)
        
        # 3. ID du Statut
        status_name = pokemon.status.name if pokemon.status is not None else "NONE"
        state_vec.append(STATUS_TO_ID.get(status_name, 0))

        # 4 à 10. Modificateurs de stats (-6 à +6)
        boosts = pokemon.boosts
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
        moves = list(pokemon.moves.keys())
        for i in range(4):
            if i < len(moves):
                state_vec.append(MOVES_ID.get(moves[i], 0))
            else:
                state_vec.append(0) # Padding si attaque inconnue/manquante

        # 15. Objet 
        state_vec.append(ITEMS_ID.get(pokemon.item, 0))
        # 16. Talent 
        state_vec.append(ABILITIES_ID.get(pokemon.ability, 0))
        # 17 is Téra ?
        state_vec.append(pokemon.is_terastallized)
        # 18 what Téra ?
        tera_type = pokemon.tera_type if pokemon.tera_type is not None else "unknown"
        state_vec.extend(TYPE_TO_ID.get(tera_type, [0.0]*19)) # Le [0.0] * 19 correspond à la value de unknown

    def bloc_B_E(self, reserve: list, state_vec: list):
        # On force une boucle stricte de 5 itérations pour les 5 emplacements du banc
        for i in range(5):
            if i < len(reserve):
                mon = reserve[i]
                
                # 1. ID du Pokémon
                state_vec.append(POKE_ID.get(mon.species, 0))
                
                # 2. PV restants
                state_vec.append(mon.current_hp_fraction)
                
                # 3. ID du Statut
                status_name = mon.status.name if mon.status is not None else "NONE"
                state_vec.append(STATUS_TO_ID.get(status_name, 0))

                # 4 à 7. Attaques
                moves = list(mon.moves.keys())
                for j in range(4):
                    if j < len(moves):
                        state_vec.append(MOVES_ID.get(moves[j], 0))
                    else:
                        state_vec.append(0) 

                # 8. Objet
                item_str = mon.item if mon.item is not None else "unknown"
                state_vec.append(ITEMS_ID.get(item_str, 0))

                # 9. Talent
                ability_str = mon.ability if mon.ability is not None else "unknown"
                state_vec.append(ABILITIES_ID.get(ability_str, 0))

                # 10. Est Téra-cristallisé ?
                state_vec.append(1.0 if mon.is_terastallized else 0.0)

                # 11 à 29. Téra-Type (One-Hot Encoding 19 dimensions)
                if mon.tera_type is not None:
                    tera_name = mon.tera_type.name.lower() 
                else:
                    tera_name = "unknown"
                state_vec.extend(TYPE_TO_ID.get(tera_name, [0.0] * 19))

            else:
                # PADDING GÉANT : Le Pokémon est inconnu ou manquant
                # 1 (ID) + 1 (PV) + 1 (Statut) + 4 (Attaques) + 1 (Objet) + 1 (Talent) + 1 (Téra bool) = 10 zéros
                # + 19 zéros pour le Téra-Type One-Hot
                state_vec.extend([0.0] * 29)

    def bloc_D(self, battle: Battle, state_vec: list):
        
        # 1. ID de la Météo
        weather_name = list(battle.weather.keys())[0].name if battle.weather else "NONE"
        state_vec.append(WEATHER_TO_ID.get(weather_name, 0))
        
        # 2. ID du Terrain 
        terrain_name = list(battle.fields.keys())[0].name if battle.fields else "NONE"
        state_vec.append(TERRAIN_TO_ID.get(terrain_name, 0))

        my_sides = {k.name: v for k, v in battle.side_conditions.items()}
        opp_sides = {k.name: v for k, v in battle.opponent_side_conditions.items()}

        # --- JOUEUR (7 dimensions) ---
        state_vec.append(my_sides.get("SPIKES", 0))          # 3. Picots (0 à 3)
        state_vec.append(my_sides.get("TOXIC_SPIKES", 0))    # 4. Pics Toxik (0 à 2)
        state_vec.append(1 if "STEALTH_ROCK" in my_sides else 0) # 5. Piège de Roc
        state_vec.append(1 if "STICKY_WEB" in my_sides else 0)   # 6. Toile Gluante
        state_vec.append(1 if "AURORA_VEIL" in my_sides else 0)  # 7. Voile Aurore
        state_vec.append(1 if "LIGHT_SCREEN" in my_sides else 0) # 8. Mur Lumière
        state_vec.append(1 if "REFLECT" in my_sides else 0)      # 9. Protection

        # --- ADVERSAIRE (7 dimensions) ---
        state_vec.append(opp_sides.get("SPIKES", 0))          # 10. Picots
        state_vec.append(opp_sides.get("TOXIC_SPIKES", 0))    # 11. Pics Toxik
        state_vec.append(1 if "STEALTH_ROCK" in opp_sides else 0) # 12. Piège de Roc
        state_vec.append(1 if "STICKY_WEB" in opp_sides else 0)   # 13. Toile Gluante
        state_vec.append(1 if "AURORA_VEIL" in opp_sides else 0)  # 14. Voile Aurore
        state_vec.append(1 if "LIGHT_SCREEN" in opp_sides else 0) # 15. Mur Lumière
        state_vec.append(1 if "REFLECT" in opp_sides else 0)      # 16. Protection

        # --- DROITS TÉRA-CRISTAL (2 dimensions) ---
        # 17. Le joueur peut-il encore Téra ?
        state_vec.append(1.0 if battle.can_tera else 0.0)
        
        # 18. L'adversaire peut-il encore Téra ?
        # Showdown ne le dit pas explicitement pour l'adversaire, on vérifie si l'un de ses Pokémon l'a déjà fait
        opp_has_tera = any(p.is_terastallized for p in battle.opponent_team.values())
        state_vec.append(0.0 if opp_has_tera else 1.0)

    def rewardVSrandom(self, last_turn, current_turn,) -> float:
        reward = 0.0
        """
        Bonne récompense pour les dégat bruts, l'objectif est que le modèle comprenne les bases 
        Évidemment la victoire et la défaite sont priorisées 
        """

        deltaPV_inflicted = 0 # Mesure la variation de PV de l'adversaire
        deltaPV_taken = 0 # Mesure la variation de PV du joueur

        PV_before = 0
        PV_now = 0

        for i, mon in enumerate(last_turn.opponent_team.values()):
            PV_before += mon.current_hp_fraction

        for i, mon in enumerate(current_turn.opponent_team.values()):
            PV_now += mon.current_hp_fraction

        deltaPV_inflicted = PV_before - PV_now

        PV_before = 0
        PV_now = 0

        for i, mon in enumerate(last_turn.team.values()):
            PV_before += mon.current_hp_fraction

        for i, mon in enumerate(current_turn.team.values()):
            PV_now += mon.current_hp_fraction

        deltaPV_taken = PV_before - PV_now

        reward += deltaPV_inflicted - deltaPV_taken

        if current_turn.won :
            reward += 100

        elif current_turn.lost :
            reward -= 100

        return reward


    