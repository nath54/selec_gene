
rules = {
    # World
    "WorldType": "finite_closed", # "finite_loop", "infinite", 
    "WorldSize": (3840, 2048),

    # Spawn
    "AgentsSpawnSpots": ["random", 20, 3], # ["random", number of species, number of entities for each species] or  [Circles : (x, y, radius, number of species that will be close enought to reproduce)]
    "VegetationSpawnSpots": [(3840/2, 2048/2, 1024/2, 3000)],

    # Agent count
    "AgentCellSize": 256,

    # Vegetation
    "VegetationGridSize": 64, # Pour éviter les problèmes, on préferera que ce soit un diviseur de WorldSize x et y
    "VegetationGridMax": 100,
    "VegetationReproductionCell": 1.02, # à chaque frame, la végétation est multipliée par cette valeur
    "VegetationReproductionSideCells": 0.005,
    
    # Vision
    "VisionSize": (50, 50), # Les agents voient un carré de 100x100 avec eux au centre

    # Reproduction
    "GeneDistSameSpecie": 10.0,
    "ReproductionMinAge": 0.5, # L'âge minimum pour se reproduire
    "ReproductionMaxAge": 0.9, # l'âge maximal pour se reproduire
    "SoloReproductionEnergy": 0.85, # L'énergie dépensée pour faire un enfant
    "BabySpawnDistance": 30,

    # World friction
    "WorldFriction": 0.95, # Les vitesses sont multipliées par la friction chaque frame

    # Energy
    "BaseEnergyLossPerFrame": 20.0,
    "EnergyRecoverPerVegetation": 10.0,

    # Aging
    "BaseAgingLossPerFrame": 1.0,

    # Min-Max agent gene parameters
    ## Energy
    "GeneMinEnergy": 50.0,
    "GeneMaxEnergy": 400.0,
    "GeneEDigits": 10.0,
    ## Aging
    "GeneMinAging": 5.0,
    "GeneMaxAging": 100.0,
    "GeneADigits": 10.0,
    ## Max Life
    "GeneMinMaxLife": 5.0,
    "GeneMaxMaxLife": 100.0,
    "GeneMLDigits": 10.0,
    ## Life Recup
    "GeneMinLifeRecup": 0.001,
    "GeneMaxLifeRecup": 1.0,
    "GeneLRecDigits": 1000.0,
    ## MaxAgentInSameCell
    "GeneMinMaxAgentInSameCell": 0.0,
    "GeneMaxMaxAgentInSameCell": 25.0,
    "GeneMAISCDigits": 10,
    ## Reproduction
    "GeneMinReproductionNumber": 0.0,
    "GeneMaxReproductionNumber": 1.0,
    "GeneRNDigits": 100.0,
    ## Eat Others
    "GeneMinEatOthers": 0.0,
    "GeneMaxEatOthers": 2.0,
    "GeneEODigits": 100.0,
    ## Eat Vegetation
    "GeneMinEatVegetation": 0.5,
    "GeneMaxEatVegetation": 2.0,
    "GeneEVDigits": 100.0,
    ## NumVegEat
    "GeneMinNumVegEat": 0.5,
    "GeneMaxNumVegEat": 10.0,
    "GeneNVEDigits": 100.0,
    ## Rot Acc
    "GeneMinRotAcc": 0.0,
    "GeneMaxRotAcc": 0.2,
    "GeneRotAccDigits": 10.0,
    ## Dir acc
    "GeneMinDirAcc": 0.0,
    "GeneMaxDirAcc": 30.0,
    "GeneDADigits": 10.0,
    ## Global acc
    "GeneMinGlobalAcc": 0.0,
    "GeneMaxGlobalAcc": 30.0,
    "GeneGADigits": 10.0,
    ## Brain Depth
    "GeneMinBrainDepth": 2,
    "GeneMaxBrainDepth": 4,
    ## Avg LayerSize
    "GeneMinAvgLayerSize": 1,
    "GeneMaxAvgLayerSize": 15,
    ## Freq Learning
    "GeneMinFreqLearning": 0.0,
    "GeneMaxFreqLearning": 1.0,
    "GeneFLDigits": 100.0,
    ## Learning rate
    "GeneMinLearningRate": 0.00001,
    "GeneMaxLearningRate": 0.1,
    "GeneLRDigits": 100000.0,
}





