# Upgraded deepBlue Version 
import deepbluecoursemology
from deepbluecoursemology import PlayerAI

# # Primitive Version - just IDS alone
# import idscoursemology
# from idscoursemology import PlayerAI


from playerNaive import PlayerNaive
import utils

if __name__ == "__main__":
    board = utils.generate_init_state()
    res = utils.play(PlayerAI(), PlayerNaive(), board)
    print(res)