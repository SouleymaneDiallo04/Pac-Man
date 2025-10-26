# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

# --------------------------------------------------------------------------
# Classe de base pour tous les extracteurs
# --------------------------------------------------------------------------
class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Retourne un dictionnaire {feature: valeur}.
        La valeur est souvent 1.0 pour les features indicatrices.
        """
        util.raiseNotDefined()


# --------------------------------------------------------------------------
# Extracteur "identité" : retourne simplement (state, action)
# --------------------------------------------------------------------------
class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


# --------------------------------------------------------------------------
# Extracteur basé sur les coordonnées et l'action
# --------------------------------------------------------------------------
class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats


# --------------------------------------------------------------------------
# Fonction utilitaire pour trouver la nourriture la plus proche
# --------------------------------------------------------------------------
def closestFood(pos, food, walls):
    """
    Retourne la distance de la nourriture la plus proche à partir de 'pos'.
    Utilise BFS pour se propager dans le labyrinthe.
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if food[pos_x][pos_y]:
            return dist
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    return None


# --------------------------------------------------------------------------
# Fonction utilitaire pour trouver la capsule la plus proche
# --------------------------------------------------------------------------
def closestCapsule(pos, capsules, walls):
    """
    Retourne la distance de la capsule la plus proche à partir de 'pos'.
    Utilise BFS pour se propager dans le labyrinthe.
    """
    if not capsules:
        return None
        
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x, pos_y) in capsules:
            return dist
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    return None


# --------------------------------------------------------------------------
# Fonction pour calculer la distance de Manhattan
# --------------------------------------------------------------------------
def manhattanDistance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# --------------------------------------------------------------------------
# Extracteur simple pour reflex Pacman
# --------------------------------------------------------------------------
class SimpleExtractor(FeatureExtractor):
    """
    Simple feature extractor:
    - Si Pacman mangera la nourriture après l'action
    - Distance à la nourriture la plus proche
    - Nombre de fantômes à 1 pas
    """
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        features = util.Counter()
        features["bias"] = 1.0

        # Position de Pacman après l'action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Fantômes à 1 pas
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # Manger la nourriture si pas de fantôme proche
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Distance à la nourriture la plus proche
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features


# --------------------------------------------------------------------------
# Extracteur avancé CORRIGÉ et AMÉLIORÉ
# --------------------------------------------------------------------------
class AdvancedExtractor(FeatureExtractor):
    """
    Advanced Feature Extractor CORRIGÉ :
    - PRIORITÉ À LA NOURRITURE : features principales pour encourager à manger
    - Gestion intelligente des fantômes : éviter seulement quand nécessaire
    - Utilisation des capsules stratégiquement
    - Features de densité de nourriture
    """

    def getFeatures(self, state, action):
        feats = util.Counter()
        
        # Position de Pacman après l'action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_pos = (next_x, next_y)

        # Éléments du jeu
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules()
        ghosts = state.getGhostPositions()
        ghost_states = state.getGhostStates()
        scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]

        # === FEATURES PRINCIPALES POUR MANGER ===
        
        # 1. MANGER DIRECTEMENT DE LA NOURRITURE (TRÈS IMPORTANT)
        if food[next_x][next_y]:
            feats["eats-food"] = 10.0  # Forte récompense pour manger
        
        # 2. DISTANCE À LA NOURRITURE LA PLUS PROCHE (PRIORITAIRE)
        dist_food = closestFood(next_pos, food, walls)
        if dist_food is not None:
            # Normaliser et inverser (plus c'est proche, mieux c'est)
            feats["closest-food"] = -0.5 * (float(dist_food) / (walls.width * walls.height))
        
        # 3. DENSITÉ DE NOURRITURE AUTOUR (nouveau)
        food_count = 0
        total_count = 0
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                nx, ny = next_x + dx, next_y + dy
                if 0 <= nx < food.width and 0 <= ny < food.height:
                    total_count += 1
                    if food[nx][ny]:
                        food_count += 1
        if total_count > 0:
            feats["food-density"] = 2.0 * (food_count / total_count)  # Encourager les zones denses en nourriture

        # === GESTION INTELLIGENTE DES FANTÔMES ===
        
        # 4. DISTANCE AU FANTÔME LE PLUS PROCHE (seulement s'il n'est pas effrayé)
        min_ghost_dist = float('inf')
        for ghost_pos, scared_time in zip(ghosts, scared_times):
            dist = manhattanDistance(next_pos, ghost_pos)
            if scared_time == 0:  # Fantôme normal - l'éviter
                min_ghost_dist = min(min_ghost_dist, dist)
            else:  # Fantôme effrayé - le poursuivre!
                if dist < 3:  # Si proche d'un fantôme effrayé
                    feats["chase-scared-ghost"] = 2.0 * (1.0 / (dist + 1))
        
        if min_ghost_dist < float('inf'):
            # Éviter les fantômes seulement s'ils sont proches
            if min_ghost_dist <= 2:
                feats["ghost-too-close"] = -5.0 * (1.0 / (min_ghost_dist + 0.1))
            elif min_ghost_dist <= 5:
                feats["ghost-nearby"] = -1.0 * (1.0 / min_ghost_dist)
        
        # 5. FANTÔMES À 1 CASE (DANGER IMMÉDIAT)
        ghosts_1_away = sum(next_pos in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        if ghosts_1_away > 0:
            feats["ghost-adjacent"] = -8.0 * ghosts_1_away  # Pénalité forte

        # === STRATÉGIE DES CAPSULES ===
        
        # 6. MANGER UNE CAPSULE DIRECTEMENT
        if (next_x, next_y) in capsules:
            feats["eats-capsule"] = 5.0  # Récompense pour manger capsule
        
        # 7. DISTANCE À LA CAPSULE LA PLUS PROCHE (seulement si utile)
        if capsules:
            dist_capsule = closestCapsule(next_pos, capsules, walls)
            if dist_capsule is not None:
                # Aller vers capsule seulement si des fantômes menacent
                if min_ghost_dist < 4:
                    feats["closest-capsule"] = -0.3 * (float(dist_capsule) / (walls.width * walls.height))

        # === FEATURES DE MOBILITÉ ===
        
        # 8. NOMBRE DE COUPS POSSIBLES (éviter les culs-de-sac)
        legal_actions = state.getLegalPacmanActions()
        if action in legal_actions:
            feats["legal-action"] = 0.1
        
        # 9. ÉVITER LES CULS-DE-SAC
        if len(legal_actions) <= 2:  # Peu d'options de mouvement
            feats["dead-end-risk"] = -0.5

        # === FEATURE DE BIAS ===
        feats["bias"] = 0.1  # Petit biais pour encourager l'exploration

        return feats