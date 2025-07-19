import sys
from debug.dqn_debug_v1 import debug
from train.dqn_train_v1 import train
from evaluate.dqn_evaluate_v1 import evaluate

if len(sys.argv) != 2:
    print("Usage: python main.py [ debug | train |eval ]")
    sys.exit(1)

if sys.argv[1] == "run":
    debug(step_by_step=False)

elif sys.argv[1] == "debug":
    debug()

elif sys.argv[1] == "train":
    train("test", 100_000)

elif sys.argv[1] == "eval":
    evaluate("test_3000", 10, debug=False)
