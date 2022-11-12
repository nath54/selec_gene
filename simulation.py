
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
                    ag.pos = (x2, y2)
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
                    ag.pos = (x2, y2)
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
            im.rectangle([(int(x*s), int(y*s)), (int((x+1)*s), int((y+1)*s))], fill=(0,int(sim.vegetation[x,y]/rules["VegetationGridMax"]*255),0))
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
    nm: np.matrix = np.zeros([tx, ty])
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
    sim_frame: int = 0
    compteur: int = 50
    temps: list = []
    i: str = input("Commencer La simulation ?    ('q' to quit)\n : ")
    while i != "q":
        t1: float = time.time()
        #
        img_mat: np.matrix = np.asarray(sim.img)
        ii: np.matrix = np.moveaxis(img_mat, 0, -1)
        # On bouge les agents
        for a in sim.agents:
            # Vision: récuperer l'image 
            i: np.matrix = get_submatr(img_mat, a.pos[0]-rules["VisionSize"][0]/2, a.pos[1]-rules["VisionSize"][1]/2, rules["VisionSize"][0], rules["VisionSize"][1])
            i = np.moveaxis(i, -1, 0)
            t: torch.Tensor = torch.zeros((1, 3, rules["VisionSize"][0], rules["VisionSize"][1]))
            t[0] = torch.from_numpy(i)
            res: torch.Tensor = a.brain.forward(t)
            print("test : ", res)
            #TODO: collisions
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
            if i != "q": compteur, temps = 50, []



if __name__ == "__main__":
    main_simulation()
