import argparse
import Agent
import PolicyValueFn
import Board

def main():
    # Training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-batch-size', type=int, default=15000)
    parser.add_argument('--channels',type=int,default=4)
    parser.add_argument('--size',type=int,default=8)
    parser.add_argument('--numOfIterations',type=int,default=1600)
    parser.add_argument('--numberForWin',type=int,default=4)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--show-size', type=int, default=15000)
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    model = PolicyValueFn.PolicyValueFn(args)
    agent1 = Agent.SelfplayAgent(args.numOfIterations,model,"test.txt")
    agent2 = Agent.RandomAgent()
    b = Board.Board(args.size,args.numberForWin)


if __name__ == '__main__':
    main()
