import random

_GENERIC_SENTENCES = [
    "The weather was pleasant today with mild temperatures.",
    "A new movie premiered this weekend to mixed reviews.",
    "Local sports team won their match in extra time.",
    "Scientists discovered a new species of beetle in the Amazon.",
    "The annual flower show attracted thousands of visitors.",
    "Road works on the high street are expected to finish next week.",
    "A popular restaurant opened a second branch across town.",
    "Volunteers cleaned up the beach ahead of the summer season.",
    "The library extended its opening hours on weekday evenings.",
    "A documentary about penguins topped streaming charts.",
    "Farmers reported a good harvest due to favourable rainfall.",
    "The city council approved plans for a new cycling lane.",
    "A local artist unveiled a mural in the town centre.",
    "School children planted trees as part of an environmental project.",
    "The annual marathon drew over five thousand participants.",
    "A community garden was opened in the west end of the park.",
    "Archaeologists found Roman coins during building works.",
    "The museum extended its exhibition on ancient civilisations.",
    "A travelling circus set up camp on the outskirts of town.",
    "Heavy snowfall caused delays on mountain roads overnight.",
    "The birdwatching club recorded a rare sighting near the estuary.",
    "A new coffee shop opened next to the train station.",
    "Local schools celebrated international languages day.",
    "The theatre company announced its programme for next season.",
]


class RandomNewsStore:
    """Returns random non-financial sentences — used in Experiment 2 (random news control)."""

    def retrieve(self, query, k=3):
        return random.sample(_GENERIC_SENTENCES, min(k, len(_GENERIC_SENTENCES)))
