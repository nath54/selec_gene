import numpy as np
import torch

from agent import *

# for _ in range(20):
#     ga = AgentGene()
#     g1 = new_gene_from([ga])
#     g2 = new_gene_from([ga])
#     for x in range(1, 11):
#         print(f"parent - enfant1 gen {x} : ",gene_dist(ga, g1))
#         print(f"parent - enfant2 gen {x} : ",gene_dist(ga, g2))
#         print(f"enfant1 - enfant2 gen {x} : ",gene_dist(g2, g1))
#         print()
#         g1 = new_gene_from([g1])
#         g2 = new_gene_from([g2])
#     print("\n"+"-"*20+"\n")

g = AgentGene()
b = AgentBrain([100, 50])
ag = Agent(g, b)

print(ag)
t: torch.Tensor = torch.rand((3, rules["VisionSize"][0], rules["VisionSize"][1]))

x = ag.brain.forward(t)

print(x)












