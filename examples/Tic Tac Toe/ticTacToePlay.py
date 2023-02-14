import numpy as np
import math
import random
import time

from ticTacToeClasses import *

#this is just for testing
def play():
    userSym = changeSym(TRAINSIDE)
    t = TicTacToe()
    user = UserPlayer(userSym)
    b = Minimax(changeSym(userSym))
    while t.status is None:
        if user.symbol == t.turn:
            t.printout()
            t.makeMove(user.getMove(t), user.symbol)
        t.check()
        if b.symbol == t.turn and t.status is None:
            t.printout()
            botMove = b.getMove(t)
            print('Bot moved:', botMove)
            t.makeMove(botMove, b.symbol)
        t.check()
    t.printout()
    print(['Tie!', 'X Wins!', 'O Wins!'][t.status])

play()
