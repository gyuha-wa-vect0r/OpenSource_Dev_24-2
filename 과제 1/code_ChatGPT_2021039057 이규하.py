import random

# Define the cards
suits = ['H', 'D', 'S', 'C']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

# Create a deck of cards
deck = [rank + suit for suit in suits for rank in ranks]

# Shuffle the deck and deal 5 cards
random.shuffle(deck)
hand = deck[:5]

hand