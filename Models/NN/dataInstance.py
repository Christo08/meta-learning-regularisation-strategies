from dataclasses import dataclass

@dataclass
class DataInstance:
    datasets: str
    seed: int

    # Simple meta-feature
    # Number of attributes
    numberOfAttributes: int
    # Proportion of category to numeric attributes
    proportionCategory: float
    # Proportion of numeric to category attributes
    proportionNumeric: float
    # Number of instances
    numberOfInstances: int
    # Number of classes
    numberOfClasses: int
    # Proportion of attributes per instances
    proportionAttributes: float
    # Number of classes per attributes
    proportionClasses: float
    # Number of instances per class
    proportionInstances: float
    # Frequency of instances in each class
    frequencyOfInstancesPerClass: float

    # Information-theoretic meta-features
    # Mutual information
    minMutualInformation: float
    maxMutualInformation: float
    # Equivalent number of attributes
    equivalentAttributes: float
    # Noise signal ratio
    nsr: float

    #target
    bestTechnique: str