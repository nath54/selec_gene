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
        self.max_energy: float = random.randint(int(rules["GeneMinEnergy"]*rules["GeneEDigits"]), int(rules["GeneMaxEnergy"]*rules["GeneEDigits"]))/rules["GeneEDigits"]
        self.aging: float = random.randint(int(rules["GeneMinAging"]*rules["GeneADigits"]), int(rules["GeneMaxAging"]*rules["GeneADigits"]))/rules["GeneADigits"]
        self.max_life: float = random.randint(int(rules["GeneMinMaxLife"]*rules["GeneMLDigits"]), int(rules["GeneMaxMaxLife"]*rules["GeneMLDigits"]))/rules["GeneMLDigits"]
        self.life_recup: float = random.randint(int(rules["GeneMinLifeRecup"]*rules["GeneLRecDigits"]), int(rules["GeneMaxLifeRecup"]*rules["GeneLRecDigits"]))/rules["GeneLRecDigits"]
        # Reproduction
        self.reproduction_method: float = random.randint(int(rules["GeneMinReproductionNumber"]*rules["GeneRNDigits"]), int(rules["GeneMaxReproductionNumber"]*rules["GeneRNDigits"]))/rules["GeneRNDigits"] # La valeur entière de cette var détermine le nombre d'individus qu'il faut pour 
        # Eating
        self.eat_other_agent: float = random.randint(int(rules["GeneMinEatOthers"]*rules["GeneEODigits"]), int(rules["GeneMaxEatOthers"]*rules["GeneEODigits"]))/rules["GeneEODigits"] # Capacité à absorber les autres agents
        self.eat_vegetation: float = random.randint(int(rules["GeneMinEatVegetation"]*rules["GeneEVDigits"]), int(rules["GeneMaxEatVegetation"]*rules["GeneEVDigits"]))/rules["GeneEVDigits"] # Capacité à absorber la végétation
        self.num_veg_eat: float = random.randint(int(rules["GeneMinNumVegEat"]*rules["GeneNVEDigits"]), int(rules["GeneMaxNumVegEat"]*rules["GeneNVEDigits"]))/rules["GeneNVEDigits"] # Quantitée de vegetation capable d'absorber par frame
        # Moving
        self.rot_acc: float = random.randint(int(rules["GeneMinRotAcc"]*rules["GeneRotAccDigits"]), int(rules["GeneMaxRotAcc"]*rules["GeneRotAccDigits"]))/rules["GeneRotAccDigits"]
        self.dir_acc: float = random.randint(int(rules["GeneMinDirAcc"]*rules["GeneDADigits"]), int(rules["GeneMaxDirAcc"]*rules["GeneDADigits"]))/rules["GeneDADigits"]
        self.global_acc: float = random.randint(int(rules["GeneMinGlobalAcc"]*rules["GeneGADigits"]), int(rules["GeneMaxGlobalAcc"]*rules["GeneGADigits"]))/rules["GeneGADigits"]
        # Brain
        # Brain - Structure
        self.braindepth: int = random.randint(int(rules["GeneMinBrainDepth"]), int(rules["GeneMaxBrainDepth"]))
        self.avglayersize: int = random.randint(int(rules["GeneMinAvgLayerSize"]), int(rules["GeneMaxAvgLayerSize"]))
        self.variance: float = random.randint(0, int(self.avglayersize*0.5))
        self.brain_layers: list = [
            max(1, self.avglayersize+random.randint(-self.variance, self.variance))
            for _ in range(self.braindepth)
        ]
        # Brain - Evolution
        self.freq_learning: float = random.randint(int(rules["GeneMinFreqLearning"]*rules["GeneFLDigits"]), int(rules["GeneMaxFreqLearning"]*rules["GeneFLDigits"]))/rules["GeneFLDigits"] # 0 = n'apprend jamais, 1 = apprend tout le temps
        self.learning_rate: float = random.randint(int(rules["GeneMinLearningRate"]*rules["GeneLRDigits"]), int(rules["GeneMaxLearningRate"]*rules["GeneLRDigits"]))/rules["GeneLRDigits"] # amplitude des changements lors d'un apprentissage
        # Body
        self.size: float = self.max_energy*random.randint(50, 150)/1000.0
        self.color: tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def new_gene_from(lst_parents: list) -> AgentGene:
    new_agent = AgentGene()
    #
    new_agent.max_energy = clamp(
        sum([x.max_energy for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10.0,
        rules["GeneMinEnergy"], rules["GeneMaxEnergy"])
    new_agent.aging = clamp(
        sum([x.aging for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10.0,
        rules["GeneMinAging"], rules["GeneMaxAging"])
    new_agent.max_life = clamp(
        sum([x.max_life for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10.0,
        rules["GeneMinMaxLife"], rules["GeneMaxMaxLife"])
    new_agent.life_recup = clamp(
        sum([x.life_recup for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinLifeRecup"], rules["GeneMaxLifeRecup"])
    #
    new_agent.reproduction_method = clamp(
        sum([x.reproduction_method for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinReproductionNumber"], rules["GeneMaxReproductionNumber"])
    #
    new_agent.eat_other_agent = clamp(
        sum([x.eat_other_agent for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinEatOthers"], rules["GeneMaxEatOthers"])
    new_agent.eat_vegetation = clamp(
        sum([x.eat_vegetation for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinEatVegetation"], rules["GeneMaxEatVegetation"])
    new_agent.num_veg_eat = clamp(
        sum([x.num_veg_eat for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinNumVegEat"], rules["GeneMaxNumVegEat"])
    #
    new_agent.rot_acc = clamp(
        sum([x.rot_acc for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinRotAcc"], rules["GeneMaxRotAcc"])
    new_agent.dir_acc = clamp(
        sum([x.dir_acc for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinDirAcc"], rules["GeneMaxDirAcc"])
    new_agent.global_acc = clamp(
        sum([x.global_acc for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/1000.0,
        rules["GeneMinGlobalAcc"], rules["GeneMaxGlobalAcc"])
    #
    new_agent.braindepth = int(clamp(
        int(sum([x.braindepth for x in lst_parents])/float(len(lst_parents))
        +random.randint(-100, 100)/100.0),
        rules["GeneMinBrainDepth"], rules["GeneMaxBrainDepth"]))
    new_agent.avglayersize = int(clamp(
        int(sum([x.avglayersize for x in lst_parents])/float(len(lst_parents))
        +random.randint(-100, 100)/100.0),
        rules["GeneMinAvgLayerSize"], rules["GeneMaxAvgLayerSize"]))
    new_agent.variance = random.randint(0, int(new_agent.avglayersize*0.5))
    new_agent.brain_layers = [
            max(1, new_agent.avglayersize+random.randint(-new_agent.variance, new_agent.variance))
            for _ in range(new_agent.braindepth)
        ]
    #
    new_agent.freq_learning = clamp(
        sum([x.freq_learning for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/10000.0,
        rules["GeneMinFreqLearning"], rules["GeneMaxFreqLearning"])
    new_agent.learning_rate = clamp(
        sum([x.learning_rate for x in lst_parents])/float(len(lst_parents))
        + random.randint(-100, 100)/100000.0,
        rules["GeneMinLearningRate"], rules["GeneMaxLearningRate"])
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
        dist += abs(a.aging-b.aging)/100.0
        dist += abs(a.life_recup-b.life_recup)
        dist += abs(a.reproduction_method-b.reproduction_method)*10
        dist += abs(a.eat_other_agent-b.eat_other_agent)*10 # 0 - 20
        dist += abs(a.eat_vegetation-b.eat_vegetation)*10 # 0 - 20
        dist += abs(a.num_veg_eat - b.num_veg_eat)*2 # 0 - 10
        dist += abs(a.rot_acc - b.rot_acc) # 0 - 10
        dist += abs(a.dir_acc - b.dir_acc) # 0 - 10
        dist += abs(a.global_acc - b.global_acc) # 0 - 10
        dist += abs(a.braindepth - b.braindepth)*5 # 0 - 50
        dist += abs(a.avglayersize - b.avglayersize)/20.0 # 0 - 50
        dist += abs(a.color[0] - b.color[0])/100.0 # 0 - 2.55
        dist += abs(a.color[1] - b.color[1])/100.0 # 0 - 2.55
        dist += abs(a.color[2] - b.color[2])/100.0 # 0 - 2.55
        return dist


class AgentBrain(nn.Module):
    def __init__(self, dims):
        super(AgentBrain, self).__init__()
        ks: int = rules["VisionSize"][0]//10
        self.vue: nn.Conv2d = nn.Conv2d(3, 1, ks)
        self.prelin: nn.Linear = nn.Linear( ((rules["VisionSize"][0]+1)-ks)*((rules["VisionSize"][1]+1)-ks), dims[0])
        #
        arr = []
        for i in range(len(dims)-1):
            arr.append( nn.Linear(dims[i], dims[i+1]) )
        self.lins: nn.Sequential = nn.Sequential(*arr)
        # Doit retourner :
        # l'accélération rotationnelle (1) -> [-1, 1]
        # l'accélération directionnelle (1) -> [-1, 1]
        # l'accélération globale x et y (2) -> [-1, 1]
        # le sentiment qu'il possède (est_destructeur - neutre - est_amical) (1) -> [-1, 1]
        self.out_lin: nn.Linear = nn.Linear(dims[-1], 5)
        self.remap: nn.Softsign = nn.Softsign()
        #
        #print(self)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        #print("DEB FORWARD")
        #print("1) X.shape = ", X.shape)
        X = self.vue(X)
        #print("2) X.shape = ", X.shape)
        X = torch.flatten(X)
        #print("3) X.shape = ", X.shape)
        X = self.prelin(X)
        #print("4) X.shape = ", X.shape)
        X = self.lins.forward(X)
        #print("5) X.shape = ", X.shape)
        X = self.out_lin(X)
        #print("6) X.shape = ", X.shape)
        X = self.remap(X)
        #print("7) X.shape = ", X.shape)
        #print("END FORWARD")
        return X


class Agent:
    def __init__(self, gene: AgentGene, brain: AgentBrain) -> None:
        # genes
        self.gene: AgentGene = gene
        # brain
        self.brain: AgentBrain = brain
        # Position and mouvement
        self.pos: list = [0.0, 0.0]
        self.angle: float = random.uniform(0, 2*math.pi)   # En radians
        self.directionnal_velocity: float = 0.0
        self.global_velocity: list = [0.0, 0.0]
        self.max_speed: float = 10.0 * (10.0/(self.gene.size))
        #print("max speed : ", self.max_speed)
        # Health
        self.current_energy: float = gene.max_energy
        self.current_age: float = gene.aging
        self.current_life: float = gene.max_life
        # sentiment
        self.sentiment = 0 # (-1 ~ -0.4) = aggressif, (-0.4 ~ 0.4)  neutre, (0.4 ~ 0.7) = amical, (0.7 ~ 1) = reproduction
        


