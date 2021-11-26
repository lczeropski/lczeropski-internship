
import random


class Card:
    col = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    values = ['A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    def __init__(self, value, color):
        if value in Card.values and color in Card.col:
            self.value = value
            self.color = color
        else:
            raise ValueError("Try to define a proper card")
    def __repr__(self):
        return f"{self.value} of {self.color}"

class Deck:
    def __init__(self):
        self.cards=[v +" of "+ c for v in Card.values for c in Card.col]
    def count(self):
        return len(self.cards)
    def __repr__(self):
        return f"Deck of {self.count()} cards" 
    def _deal(self,num):
        if not self.count():
            raise ValueError("All cards have been dealt")
        if num > self.count():
            num = self.count()
        for i in range(num):
            self.cards.pop()
    def shufflee(self):
        if self.count() == 56:
            random.shuffle(self.cards)
            return self.cards
        raise ValueError("Only full decks can be shuffled")
    def deal_card(self):
        de = self.cards[self.count()-1]
        self._deal(1)
        return de
    def deal_hand(self,num):
        hand=[self.cards[self.count()-1-i] for i in range(0,num)]
        self._deal(num)
        return hand

