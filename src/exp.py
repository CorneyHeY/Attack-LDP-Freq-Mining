from data import Data
from svsm import SVSM
from atk import Adversary

def main():
    top_k = 64
    epsilon = 4
    data = Data(dataname='IBM',limit=-1)
    attacker = Adversary(data, top_k, epsilon)
    finder = SVSM(data, attacker, top_k, epsilon)
    cand_dict = finder.find(mode=0)

    print(cand_dict)

main()
