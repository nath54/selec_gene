
rules = {
    # World
    "WorldType": "finite_closed", # "finite_loop", "infinite", 
    "WorldSize": (2048, 2048),

    # Spawn
    "AgentsSpawnSpots": ["random", 10, 5], # ["random", number of species, number of entities for each species] or  [Circles : (x, y, radius, number of species that will be close enought to reproduce)]
    "VegetationSpawnSpots": [(0, 0, 2048, 1000)],

    # Vegetation
    "VegetationGridSize": 64, # Pour éviter les problèmes, on préferera que ce soit un diviseur de WorldSize x et y
    "VegetationGridMax": 100,
    "VegetationReproductionCell": 1.04, # à chaque frame, la végétation est multipliée par cette valeur
    "VegetationReproductionSideCells": 0.01,
    
    # Vision
    "VisionSize": (100, 100), # Les agents voient un carré de 100x100 avec eux au centre

    # Reproduction
    "GeneDistSameSpecie": 10.0,
    "ReproductionMinAge": 0.1, # L'âge minimum pour se reproduire
    "ReproductionMaxAge": 0.8, # l'âge maximal pour se reproduire
    "SoloReproductionEnergy": 0.7, # L'énergie dépensée pour faire un enfant

    # World friction
    "WorldFriction": 0.99 # Les vitesses sont multipliées par la friction chaque frame
}





