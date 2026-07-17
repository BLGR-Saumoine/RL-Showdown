import gymnasium as gym
from gymnasium import spaces

class PokemonRandbatsEnv(gym.Env):
    
    def __init__(self):
        self.action_space = spaces.Discrete(14) # 4 Attaques, 4 Attaques mais en terracristalisant, 5 pokemons à switcher, 1 cas autre au cas où

        self.observation_space = spaces.Discrete(138) 
        # =========================================================================
        # OBSERVATION SPACE : gym.spaces.Box(shape=(138,))
        # Vecteur 1D représentant l'état du jeu à un instant T pour l'agent.
        # Les valeurs "inconnues" (surtout pour l'adversaire) sont initialisées à 0.
        # =========================================================================

        # --- BLOC A : Le Pokémon Actif du Joueur (16 dimensions) ---
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

        # --- BLOC B : L'Équipe en Réserve du Joueur (5 * 9 = 45 dimensions) ---
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

        # --- BLOC C : Le Pokémon Actif Adverse (16 dimensions) ---
        # (Structure identique au Bloc A. L'information cachée vaut 0)
        # 1.  ID du Pokémon adverse
        # 2.  Pourcentage de PV visibles (0.0 à 1.0)
        # 3.  ID du Statut
        # 4-10. Modificateurs de stats visibles (-6 à +6)
        # 11-14. IDs des attaques (0 si non révélée)
        # 15. ID Objet (0 si non révélé/déclenché)
        # 16. ID Talent (0 si non révélé/déclenché)

        # --- BLOC D : Le Terrain & Les Effets Globaux (16 dimensions) ---
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

        # --- BLOC E : L'Équipe en Réserve Adverse [MÉMOIRE] (5 * 9 = 45 dimensions) ---
        # (Structure identique au Bloc B. Initialisé à 0. Se remplit 
        # progressivement lorsque l'adversaire révèle son équipe).
        # Pour chaque emplacement adverse en attente (Slots 1 à 5) :
        # 1.  ID du Pokémon (0 si inconnu)
        # 2.  PV restants vus lors de son dernier switch (0.0 à 1.0)
        # 3.  ID du Statut (0 si aucun/inconnu)
        # 4-7. IDs des attaques (0 si non révélée)
        # 8.  ID Objet (0 si inconnu)
        # 9.  ID Talent (0 si inconnu)

        # TOTAL = 16 + 45 + 16 + 16 + 45 = 138 dimensions.


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        

        return observation, info

    def step(self, action):
        # 3. Exécution d'un tour de jeu
        ...
        return observation, reward, terminated, truncated, info