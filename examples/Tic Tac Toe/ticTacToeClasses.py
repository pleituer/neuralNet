import numpy as np
import math
import random
import time

#global vars
x = 1
o = -1
_ = 0
changeSym = lambda sym: {x:o, o:x}[sym]

#train side set to 2nd player (O)
TRAINSIDE = o

#the board
class TicTacToe():
    def __init__(self):
        self.turn = x
        self.size = 3
        self.board = np.zeros((self.size, self.size))
        self.status = None #None -> not ended, 0 -> tie, non-zero number -> player of this number won
    
    def squish(self, board=None):
        if board is None: return np.reshape(self.board, (self.size**2,))
        return np.reshape(board, (self.size**2,))

    #returns all empty spots on the board
    def emptySpots(self):
        emptySpot = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == _: emptySpot.append((x,y))
        return emptySpot
    
    #prints
    def printout(self):
        print('\n'.join('|'+'|'.join(list(map(lambda c:('_','x','o')[int(c)], row)))+'|' for row in self.board))
        return 1

    #checks if anyone won
    def checkWin(self, player):
        playerList = np.array([player for __ in range(self.size)])
        rowWin = [all(row == playerList) for row in self.board]
        columnWin = [all(column == playerList) for column in self.board.T]
        diagWin = [all(np.array([self.board[i][i] for i in range(self.size)]) == playerList), all(np.array([self.board[i][self.size-i-1] for i in range(self.size)]) == playerList)]
        win = any(rowWin) or any(columnWin) or any(diagWin)
        if win: self.status = player
        return win
    
    #checks if there is a tie
    def checkTie(self):
        if len(self.emptySpots()) == 0 and self.status == None: 
            self.status = 0
            return 1
        return 0
    
    #puts stuff in front tgt to check if anyone won, or if there is a tie
    def check(self):
        self.checkWin(x)
        self.checkWin(o)
        self.checkTie()

    #makes move
    def makeMove(self, pos, player):
        if pos in self.emptySpots():
            self.board[pos[1]][pos[0]] = player
            self.turn = changeSym(self.turn)
            return 1
        return 0
    
    #undo previous move
    def undoMove(self, pos, undoStatus=True):
        self.board[pos[1]][pos[0]] = _
        self.turn = changeSym(self.turn)
        if undoStatus: self.status = None
        return 1
    
    #resets the board
    def reset(self):
        self.__init__()
        return 1
    
    #sets the board to some other 
    def set(self, board=0, status=None):
        if type(board) == int: self.board = np.zeros((self.size, self.size))
        else: self.board = board
        self.status = status

#player base class
class Player():
    def __init__(self, symbol):
        self.symbol = symbol
    
    def getMove(self, board):
        pass

#minimax algorithm implemented to train the neural network
#this provides data to be trained upon
class Minimax(Player):
    #this just makes it unbeatable
    def checkNearWin(self, board):
        availables = board.emptySpots()
        for move in availables:
            board.makeMove(move, self.symbol)
            aboutToWin = board.checkWin(self.symbol)
            board.undoMove(move)
            if aboutToWin: return move
        for move in availables:
            board.makeMove(move, changeSym(self.symbol))
            aboutToWin = board.checkWin(changeSym(self.symbol))
            board.undoMove(move)
            if aboutToWin: return move

    #the minimax algorithm
    def minimax(self, move, board, sym):
        board.makeMove(move, sym)
        availables = board.emptySpots()

        if board.checkWin(x): 
            board.undoMove(move)
            return x*(len(availables)+1)
        if board.checkWin(o):
            board.undoMove(move)
            return o*(len(availables)+1)
        if board.checkTie():
            board.undoMove(move)
            return _

        if sym == x:
            best = -math.inf
            for mv in availables: best = max(best, self.minimax(mv, board, changeSym(sym)))
        elif sym == o:
            best = math.inf
            for mv in availables: best = min(best, self.minimax(mv, board, changeSym(sym)))
        board.undoMove(move)
        return best
    
    #gets move, using minimax
    def getMove(self, board, randomize=False):
        Muhahahaha = self.checkNearWin(board)
        if Muhahahaha is not None: return Muhahahaha
        availables = board.emptySpots()
        moves = []
        for move in availables:
            score = self.minimax(move, board, self.symbol)
            moves.append((score, move))
        moves = sorted(moves, key=(lambda m:m[0]))
        if self.symbol == x:
            bestMove = moves[-1]
            bestMoves = list(filter((lambda m:m[0]==bestMove[0]), moves))
        elif self.symbol == o:
            bestMove = moves[0]
            bestMoves = list(filter((lambda m:m[0]==bestMove[0]), moves))
        if randomize: return random.choice(bestMoves)[1]
        return bestMove[1]

#user class
class UserPlayer(Player):
    def getMove(self, board): 
        move = tuple(map(int, input(f"You are {('','X','O')[self.symbol]}\nEnter column row\n> ").split()))
        while move not in board.emptySpots(): 
            print('That move is invalid, please enter again')
            move = tuple(map(int, input(f"You are {('','X','O')[self.symbol]}\nEnter column row\n> ").split()))
        return move