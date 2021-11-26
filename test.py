import unittest
from talia import Deck, Card

class CardTest(unittest.TestCase):
    def setUp(self):
        self.card = Card("A","Clubs")
    
    def test_init(self):
        """cards should have a color and a value"""
        self.assertEqual(self.card.color, "Clubs")
        self.assertEqual(self.card.value, "A")
    def test_repr(self):
        """repr should return a string of the form 'value of color'"""
        self.assertEqual(repr(self.card), "A of Clubs")

class DeckTest(unittest.TestCase):
    def setUp(self):
        self.deck=Deck()
    def test_init(self):
        """decks should have a cards attribute"""
        self.assertTrue(isinstance(self.deck.cards, list))
        self.assertEqual(len(self.deck.cards),56)
    def test_repr(self):
        """repr should return a string of the form 'Deck of 56 cards'"""
        self.assertEqual(repr(self.deck), "Deck of 56 cards")
        
    def test_count(self):
        """count should return a count of the numbers of cards"""
        self.assertEqual(self.deck.count(), 56)
        self.deck.cards.pop()
        self.assertEqual(self.deck.count(), 55)

    def test_deal_no_cards(self):
        """_deal should throw a ValueError"""
        self.deck._deal(self.deck.count())
        with self.assertRaises(ValueError):
            self.deck._deal(1)
    def test_deal_card(self):
        """should deal a single card"""
        card = self.deck.cards[-1]
        dealt_card = self.deck.deal_card()
        self.assertEqual(card, dealt_card)
        self.assertEqual(self.deck.count(), 55)
    def test_deal_hand(self):
        """should deal the number of cards"""
        cards = self.deck.deal_hand(20)
        self.assertEqual(len(cards), 20)
        self.assertEqual(self.deck.count(), 36)
    
    def test_shuffle_full_deck(self):
        """shuffle should shuffle the deck"""
        cards = self.deck.cards[:]
        self.deck.shufflee()
        self.assertNotEqual(cards, self.deck.cards)
        self.assertEqual(self.deck.count(),56)
        
    def tesh_shuffle_not_full(self):
        """shouldn't shuffle the deck"""
        self.deck._deal(1)
        with self.assertRaises(ValueError):
            self.deck.shufflee()


if __name__ == '__main__':
    unittest.main()