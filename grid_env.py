import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridEnv:
    def __init__(self, deterministic=False):
        self.grid_size = (4, 4)
        self.terminal_state = (3, 3)
        self.current_state = (0, 0)
        self.history = []
        self.deterministic = deterministic
        
        # Grille de rewards
        self.rewards = np.zeros(self.grid_size)
        self.rewards[self.terminal_state] = 1.0
        self.rewards[1, 1] = -0.1
        self.rewards[2, 2] = -0.1
        
        # Actions possibles
        self.actions = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
        self.action_space = list(self.actions.keys())
        
        # Initialisation de la visualisation
        self.fig, self.ax = None, None
        self.init_visualization()
    
    def init_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-0.5, 3.5)
        self.ax.set_ylim(-0.5, 3.5)
        self.ax.set_xticks(range(4))
        self.ax.set_yticks(range(4))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.set_title('Grid World - Reinforcement Learning', fontsize=16)
    
    def reset(self):
        self.current_state = (0, 0)
        self.history = [self.current_state]
        return self.current_state
    
    def step(self, action):
        if self.deterministic:
            move = self.actions[action]
        else:
            import random
            move = random.choice(list(self.actions.values()))
            
        new_state = (
            max(0, min(3, self.current_state[0] + move[0])),
            max(0, min(3, self.current_state[1] + move[1]))
        )
        
        reward = self.rewards[new_state]
        done = (new_state == self.terminal_state)
        
        self.current_state = new_state
        self.history.append(new_state)
        return new_state, reward, done, {}
    
    def get_all_states(self):
        return [(i, j) for i in range(4) for j in range(4)]
    
    def render(self, show_path=True, delay=0.5):
        self.ax.clear()
        
        # Dessiner la grille
        for i in range(4):
            for j in range(4):
                if (i, j) == self.terminal_state:
                    color = 'lightgreen'
                elif self.rewards[i, j] < 0:
                    color = 'lightcoral'
                else:
                    color = 'lightblue'
                
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor=color, alpha=0.7)
                self.ax.add_patch(rect)
                
                reward_text = f"{self.rewards[i, j]:.1f}"
                self.ax.text(j, i, reward_text, ha='center', va='center', 
                           fontsize=12, fontweight='bold')
        
        # Afficher le chemin parcouru
        if show_path and len(self.history) > 1:
            history_array = np.array(self.history)
            self.ax.plot(history_array[:, 1], history_array[:, 0], 
                        'ro-', linewidth=2, markersize=8, alpha=0.6, 
                        label='Chemin de l\'agent')
        
        # Position actuelle de l'agent
        current_y, current_x = self.current_state
        self.ax.plot(current_x, current_y, 's', markersize=20, 
                    color='red', label='Agent actuel')
        
        # Configuration du graphique
        self.ax.set_xlim(-0.5, 3.5)
        self.ax.set_ylim(-0.5, 3.5)
        self.ax.set_xticks(range(4))
        self.ax.set_yticks(range(4))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Position: {self.current_state} | Historique: {len(self.history)} steps', 
                         fontsize=14)
        self.ax.legend()
        
        plt.draw()
        plt.pause(delay)