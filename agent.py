import random
import math

import torch
import torch.nn as nn

from rules import rules
from lib import *

""" 
Idée : Chaque agent aura des caractéristiques fixes (physiques)
et des caractéristiques qui évoluent (cerveau)
"""



class AgentGene:
    def __init__(self):
        # Health
        self.max_energy: float = random.randint(50, 1000)/10.0
        self.aging: float = random.randint(10, 10000)/10.0
        self.max_life: float = random.randint(10, 10000)/10.0
        # Reproduction
        self.reproduction_method: float = random.randint(100, 400)/100.0 # La valeur entière de cette var détermine le nombre d'individus qu'il faut pour 
        # Eating
        self.eat_other_agent: float = random.randint(0, 200)/100.0 # Capacité à absorber les autres agents
        self.eat_vegetation: float = random.randint(0, 200)/100.0 # Capacité à absorber la végétation
        # Moving
        self.rot_acc: float = random.randint(0, 100)/10.0
        self.dir_acc: float = random.randint(0, 100)/10.0
        self.global_acc: float = random.randint(0, 100)/10.0
        # Brain
        # Brain - Structure
        self.braindepth: int = random.randint(1, 3)
        self.avglayersize: int = random.randint(1, 15)
        self.variance: float = random.randint(0, int(self.avglayersize*0.5))
        self.brain_layers: list = [
            max(1, self.avglayersize+random.randint(-self.variance, self.variance))
            for _ in range(self.braindepth)
        ]
        # Brain - Evolution
        self.freq_learning: float = random.randint(0, 100)/100.0 # 0 = n'apprend jamais, 1 = apprend tout le temps
        self.learning_rate: float = random.randint(1, 100000)/100000.0 # amplitude des changements lors d'un apprentissage
        # Body
        self.size: float = self.max_energy*random.randint(50, 150)/1000.0
        self.color: tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def new_gene_from(lst_parents: list) -> AgentGene:
    new_agent = AgentGene()
    #
    new_agent.max_energy = clamp(
        sum([x.max_energy for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10.0,
        5, 50000)
    new_agent.aging = clamp(
        sum([x.aging for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10.0,
        5, 50000)
    new_agent.max_life = clamp(
        sum([x.max_life for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10.0,
        5, 50000)
    #
    new_agent.reproduction_method = clamp(
        sum([x.reproduction_method for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        0, 4)
    #
    new_agent.eat_other_agent = clamp(
        sum([x.eat_other_agent for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        0, 2)
    new_agent.eat_vegetation = clamp(
        sum([x.eat_vegetation for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        0, 2)
    #
    new_agent.rot_acc = clamp(
        sum([x.rot_acc for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        0, 10)
    new_agent.dir_acc = clamp(
        sum([x.dir_acc for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        0, 10)
    new_agent.global_acc = clamp(
        sum([x.global_acc for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        0, 10)
    #
    new_agent.braindepth = clamp(
        int(sum([x.braindepth for x in lst_parents])/float(len(lst_parents))
        +random.randint(-100, 100)/100.0),
        1, 3
    )
    new_agent.avglayersize = clamp(
        int(sum([x.avglayersize for x in lst_parents])/float(len(lst_parents))
        +random.randint(-100, 100)/100.0),
        1, 15
    )
    new_agent.variance = random.randint(0, int(new_agent.avglayersize*0.5))
    new_agent.avglayersize = clamp(
        int(sum([x.avglayersize for x in lst_parents])/float(len(lst_parents))
        +random.randint(-100, 100)/100.0),
        1, 7
    )
    new_agent.brain_layers = [
            max(1, new_agent.avglayersize+random.randint(-new_agent.variance, new_agent.variance))
            for _ in range(new_agent.braindepth)
        ]
    #
    new_agent.freq_learning = clamp(
        sum([x.freq_learning for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10000.0,
        0, 1)
    new_agent.learning_rate = clamp(
        sum([x.learning_rate for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/100000.0,
        0.000001, 1)
    #
    new_agent.size = new_agent.max_energy*random.randint(50, 150)/1000.0
    new_agent.color = (
        int(clamp(sum([x.color[0] for x in lst_parents])/len(lst_parents)+random.randint(-2,2),0,255)),
        int(clamp(sum([x.color[1] for x in lst_parents])/len(lst_parents)+random.randint(-2,2),0,255)),
        int(clamp(sum([x.color[2] for x in lst_parents])/len(lst_parents)+random.randint(-2,2),0,255))
    )
    #
    return new_agent

def gene_dist(a: AgentGene , b: AgentGene) -> float:
        dist: float = 0.0
        dist += abs(a.aging-b.aging)/100.0 # 0 - 10
        dist += abs(a.reproduction_method-b.reproduction_method)*10 # 0 - 40
        dist += abs(a.eat_other_agent-b.eat_other_agent)*10 # 0 - 20
        dist += abs(a.eat_vegetation-b.eat_vegetation)*10 # 0 - 20
        dist += abs(a.rot_acc - b.rot_acc) # 0 - 10
        dist += abs(a.dir_acc - b.dir_acc) # 0 - 10
        dist += abs(a.global_acc - b.global_acc) # 0 - 10
        dist += abs(a.braindepth - b.braindepth)*5 # 0 - 50
        dist += abs(a.avglayersize - b.avglayersize)/20.0 # 0 - 50
        dist += abs(a.color[0] - b.color[0])/100.0 # 0 - 2.55
        dist += abs(a.color[1] - b.color[1])/100.0 # 0 - 2.55
        dist += abs(a.color[2] - b.color[2])/100.0 # 0 - 2.55
        # tot : 0 - 227.65
        return dist


class AgentBrain(nn.Module):
    def __init__(self, dims):
        super(AgentBrain, self).__init__()
        self.vue: nn.Conv2d = nn.Conv2d(3, dims[0], int(rules["VisionSize"][0]/10))
        self.flat: nn.Flatten = nn.Flatten()
        #
        arr = []
        for i in range(len(dims)-1):
            arr.append( nn.Linear(dims[i], dims[i+1]) )
        self.lins: nn.ModuleList = nn.ModuleList(arr)
        # Doit retourner :
        # l'accélération rotationnelle (1) -> [-1, 1]
        # l'accélération directionnelle (1) -> [-1, 1]
        # l'accélération globale x et y (2) -> [-1, 1]
        # le sentiment qu'il possède (est_destructeur - neutre - est_amical) (1) -> [-1, 1]
        self.out_lin: nn.Linear = nn.Linear(dims[-1], 5)
        self.remap: nn.Softsign = nn.Softsign()
        #
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.vue(X)
        X = self.flat(X)
        X = self.lins.forward(X)
        X = self.out_lin(X)
        X = self.remap(X)
        return X


class Agent:
    def __init__(self, gene: AgentGene, brain: AgentBrain) -> None:
        # Position and mouvement
        self.pos: tuple = (0.0, 0.0)
        self.angle: float = random.uniform(0, 2*math.pi)   # En radians
        self.directionnal_velocity: float = 0.0
        self.global_velocity: float = 0.0
        # Health
        self.current_energy: float = 0.0
        self.current_age: float = 0.0
        # genes
        self.gene: AgentGene = gene
        # brain
        self.brain: AgentBrain = brain
        # sentiment
        self.sentiment = 0 # (-1 ~ -0.4) = aggressif, (-0.4 ~ 0.4)  neutre, (0.4 ~ 0.7) = amical, (0.7 ~ 1) = reproduction
        


