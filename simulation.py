
import torch
import numpy as np
import time
import random
from PIL import Image, ImageDraw
import os

from world import *
from agent import *
from rules import rules

s: int = rules["VegetationGridSize"]

class Simulation:
    def __init__(self):
        # On crée les agents
        self.agents: list = []
        self.vegetation: np.matrix = np.zeros([rules["WorldSize"][0]//rules["VegetationGridSize"], rules["WorldSize"][1]//rules["VegetationGridSize"]], dtype=float)
        #
        if rules["AgentsSpawnSpots"][0] == "random":
            for _ in range(rules["AgentsSpawnSpots"][1]):
                x: int = random.randint(0, rules["WorldSize"][0])
                y: int = random.randint(0, rules["WorldSize"][1])
                r: int =  random.randint(int(rules["WorldSize"][0]*0.1), int(rules["WorldSize"][0]*0.4))
                main_gene = AgentGene()
                for _ in range(rules["AgentsSpawnSpots"][2]):
                    x2: int = clamp(random.randint(x-r, x+r), 0, rules["WorldSize"][0])
                    y2: int = clamp(random.randint(y-r, y+r), 0, rules["WorldSize"][1])
                    g: AgentGene = new_gene_from([main_gene])
                    b: AgentBrain = AgentBrain(g.brain_layers)
                    ag: Agent = Agent(g,b)
                    ag.pos = [x2, y2]
                    self.agents.append(ag)
        else:
            for c in rules["AgentsSpawnSpots"]:
                x: int = c[0]
                y: int = c[1]
                r: int = c[2]
                main_gene = AgentGene()
                for _ in range(c[3]):
                    x2: int = clamp(random.randint(x-r, x+r), 0, rules["WorldSize"][0])
                    y2: int = clamp(random.randint(y-r, y+r), 0, rules["WorldSize"][1])
                    g: AgentGene = new_gene_from([main_gene])
                    b: AgentBrain = AgentBrain(g.brain_layers)
                    ag: Agent = Agent(g,b)
                    ag.pos = [x2, y2]
                    self.agents.append(ag)
        # On crée la végétation
        for c in rules["VegetationSpawnSpots"]:
            x: int = c[0]
            y: int = c[1]
            r: int = c[2]
            for _ in range(c[3]):
                x2: int = clamp(random.randint(x-r, x+r), 0, rules["WorldSize"][0]-1)
                y2: int = clamp(random.randint(y-r, y+r), 0, rules["WorldSize"][1]-1)
                cx: int = clamp(x2//rules["VegetationGridSize"], 0, rules["WorldSize"][0]//rules["VegetationGridSize"]-1)
                cy: int = clamp(y2//rules["VegetationGridSize"], 0, rules["WorldSize"][1]//rules["VegetationGridSize"]-1)
                self.vegetation[cx, cy] += 1

def affichage(sim: Simulation) -> Image:
    img: Image = Image.new("RGB", rules["WorldSize"], (255, 255, 255))
    im: ImageDraw = ImageDraw.ImageDraw(img)
    # On affiche la végétation
    for x in range(sim.vegetation.shape[0]):
        for y in range(sim.vegetation.shape[1]):
            im.rectangle([(int(x*s), int(y*s)), (int((x+1)*s), int((y+1)*s))], fill=(0,int(sim.vegetation[x,y]/rules["VegetationGridMax"]*100),0))
    # On affiche les agents
    for a in sim.agents:
        b: float = 0.7
        st: float = (a.sentiment+1.0)/2.0
        p1: tuple = (float(a.pos[0]+a.gene.size*math.cos(a.angle)/2.0), float(a.pos[1]+a.gene.size*math.sin(a.angle)/2.0))
        p2: tuple = (float(a.pos[0]+a.gene.size*math.cos(a.angle+math.pi-b)/2.0), float(a.pos[1]+a.gene.size*math.sin(a.angle+math.pi-b)/2.0))
        p3: tuple = (float(a.pos[0]+a.gene.size*math.cos(a.angle+math.pi+b)/2.0), float(a.pos[1]+a.gene.size*math.sin(a.angle+math.pi+b)/2.0))
        im.polygon([p1, p2, p3], fill=a.gene.color, outline=(int(st*255), int((1-st)*255), 0))
    return img
    

def get_submatr(m: np.matrix, x: int, y: int, tx:int, ty:int) -> np.matrix:
    x = int(x)
    y = int(y)
    tx = int(tx)
    ty = int(ty)
    xtx = int(x+tx)
    yty = int(y+ty)
    if x > 0 and xtx < m.shape[0] and y > 0 and yty < m.shape[1]: # Le cas parfait
        return m[x:xtx, y:yty]
    #
    nm: np.matrix = np.zeros([tx, ty, 3])
    # Cas 0 : 
    if (xtx < 0) or (yty < 0) or (x > m.shape[0]) or (y > m.shape[1]):
        return nm
    # Cas 1 : 
    if x < 0 and y < 0:
        nm[-x:, -y:] = m[:xtx, :yty]
    # Cas 2 :
    if y < 0 and x > 0 and xtx < m.shape[0]:
        nm[:, -y:] = m[x:xtx, :yty]
    #TODO
    # Cas 3 : 
    if y < 0 and xtx > m.shape[0]:
        pass
    # Cas 4:
    if x < 0 and yty < m.shape[1]:
        pass
    # Cas 5:
    if xtx > m.shape[0] and yty < m.shape[1]:
        pass
    # Cas 6:
    if x < 0 and yty > m.shape[1]:
        pass
    # Cas 7:
    if xtx < m.shape[0] and yty > m.shape[1]:
        pass
    # Cas 8:
    if xtx > m.shape[0] and yty > m.shape[1]:
        pass
    #
    return nm


def main_simulation():
    name: str = input("Nom de la simulation : ")
    os.mkdir("simulations/"+name+"/")
    for f in os.listdir("simulations/"+name+"/"):
        os.remove("simulations/"+name+"/"+f)
    #
    sim: Simulation = Simulation()
    sim.img = affichage(sim) # Pour avoir une première image surlaquelle 
    #
    max_compteur: int = 150
    sim_frame: int = 0
    compteur: int = max_compteur
    temps: list = []
    i: str = input("Commencer La simulation ?    ('q' to quit)\n : ")
    while i != "q":
        t1: float = time.time()
        print("Frame ", sim_frame)
        #
        img_mat: np.matrix = np.asarray(sim.img)/255.0
        # ii: np.matrix = np.moveaxis(img_mat, 0, -1)
        # On bouge les agents
        morts: list = []
        babies: list = []
        for j in range(len(sim.agents)):
            a: Agent = sim.agents[j]
            # Vision: récuperer l'image 
            i: np.matrix = get_submatr(img_mat, a.pos[0]-rules["VisionSize"][0]/2, a.pos[1]-rules["VisionSize"][1]/2, rules["VisionSize"][0], rules["VisionSize"][1])
            i = np.moveaxis(i, -1, 0)
            t:torch.Tensor = torch.from_numpy(i).float()
            res: torch.Tensor = a.brain.forward(t)
            # On "bouge" l'agent en fonction de ses décisions
            a.angle += a.gene.rot_acc * float(res[0])
            a.directionnal_velocity += a.gene.dir_acc * float(res[1])
            a.global_velocity[0] += a.gene.global_acc * float(res[2])
            a.global_velocity[1] += a.gene.global_acc * float(res[3])
            a.sentiment = float(res[4])
            # Restriction de la vitesse
            # max speed
            a.global_velocity[0] = clamp(a.global_velocity[0], -a.max_speed, a.max_speed)
            a.global_velocity[1] = clamp(a.global_velocity[1], -a.max_speed, a.max_speed)
            a.directionnal_velocity = clamp(a.directionnal_velocity, -a.max_speed, a.max_speed)
            # friction
            a.global_velocity[0]*=rules["WorldFriction"]
            a.global_velocity[1]*=rules["WorldFriction"]
            a.directionnal_velocity*=rules["WorldFriction"]
            # Mouvement
            a.pos[0]+=int(a.global_velocity[0]+a.directionnal_velocity*math.cos(a.angle))
            a.pos[1]+=int(a.global_velocity[1]+a.directionnal_velocity*math.sin(a.angle))
            if rules["WorldType"] == "finite_closed":
                clamp(a.pos[0], a.gene.size, rules["WorldSize"][0]-a.gene.size)
                clamp(a.pos[1], a.gene.size, rules["WorldSize"][1]-a.gene.size)
            elif rules["WorldType"] == "finite_loop":
                a.pos[0],a.pos[1] = loop(a.pos[0], a.pos[1], rules["WorldSize"][0], rules["WorldSize"][1])
            #TODO: collisions (nourriture vege)
            cx: int = clamp(a.pos[0]//rules["VegetationGridSize"], 0, rules["WorldSize"][0]//rules["VegetationGridSize"]-1)
            cy: int = clamp(a.pos[1]//rules["VegetationGridSize"], 0, rules["WorldSize"][1]//rules["VegetationGridSize"]-1)
            if sim.vegetation[cx, cy] > 1:
                a.current_energy += rules["EnergyRecoverPerVegetation"]*a.gene.eat_vegetation
                sim.vegetation[cx, cy] -= 1
            #TODO: collisions (nourriture ennemi)
            pass
            #TODO: collisions (attaque ennemi)
            pass
            #TODO: collisions (reproduction)
            if math.ceil(a.gene.reproduction_method)==1:
                age_prop: float = 1.0-(a.current_age/a.gene.aging)
                if age_prop >= rules["ReproductionMinAge"] and age_prop <= rules["ReproductionMaxAge"] and a.current_energy/a.gene.max_energy >= rules["SoloReproductionEnergy"]:
                    ng: AgentGene = new_gene_from([a.gene])
                    nb: AgentBrain = AgentBrain(ng.brain_layers)
                    nag: Agent = Agent(ng, nb)
                    nag.pos = [a.pos[0]+random.randint(-rules["BabySpawnDistance"], rules["BabySpawnDistance"]), a.pos[1]+random.randint(-rules["BabySpawnDistance"], rules["BabySpawnDistance"])]
                    babies.append(nag)
                    #
                    a.current_energy -= a.gene.max_energy*rules["SoloReproductionEnergy"]
            else:
                pass
            # Diminution d'énergie
            a.current_energy -= rules["BaseEnergyLossPerFrame"]
            # Diminution de l'âge
            a.current_age -= rules["BaseAgingLossPerFrame"]
            # Récupération
            a.current_life = clamp(a.current_life+a.gene.life_recup, 0, a.gene.max_life)
            #TODO: apprentissage du cerveau
            pass
            # Mort
            if a.current_energy <= 0 or a.current_age <= 0 or a.current_life <= 0:
                morts.append(a)
        # Les morts disparaissent
        for ma in morts:
            sim.agents.remove(ma)
        # Les bébés apparaissent
        sim.agents += babies
        # On fait pousser la végétation
        veg: np.matrix = np.copy(sim.vegetation)*rules["VegetationReproductionCell"]
        for x in range(veg.shape[0]):
            for y in range(veg.shape[1]):
                if rules["WorldType"] == "finite_closed":
                    if x == 0 or x == veg.shape[0]-1 or y == 0 or y == veg.shape[1]-1:
                        continue
                    veg[x, y] += sim.vegetation[x-1, y]*rules["VegetationReproductionSideCells"]
                    veg[x, y] += sim.vegetation[x+1, y]*rules["VegetationReproductionSideCells"]
                    veg[x, y] += sim.vegetation[x, y-1]*rules["VegetationReproductionSideCells"]
                    veg[x, y] += sim.vegetation[x, y+1]*rules["VegetationReproductionSideCells"]
                elif rules["WorldType"] == "finite_loop":
                    veg[x, y] += sim.vegetation[loop(x-1, y, veg.shape[0], veg.shape[1])]*rules["VegetationReproductionSideCells"]
                    veg[x, y] += sim.vegetation[loop(x+1, y, veg.shape[0], veg.shape[1])]*rules["VegetationReproductionSideCells"]
                    veg[x, y] += sim.vegetation[loop(x, y-1, veg.shape[0], veg.shape[1])]*rules["VegetationReproductionSideCells"]
                    veg[x, y] += sim.vegetation[loop(x, y+1, veg.shape[0], veg.shape[1])]*rules["VegetationReproductionSideCells"]
                veg[x,y]=min(veg[x,y], rules["VegetationGridMax"])
        sim.vegetation = veg
        # On affiche l'image
        sim.img = affichage(sim)
        sim.img.save("simulations/"+name+"/frame"+str(sim_frame)+".png")
        #
        t2: float = time.time()
        temps.append(t2-t1)
        sim_frame+=1
        compteur -= 1
        if compteur == 0:
            i = input(f"Simulation Frame = {sim_frame}\nTemps des calculs (Moyenne)= {sum(temps)/len(temps)} sec\nContinuer La simulation ?    ('q' to quit)\n : ")
            if i != "q": compteur, temps = max_compteur, []


if __name__ == "__main__":
    main_simulation()
