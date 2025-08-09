import sys

if len(sys.argv) != 2:
    print("Usage: python main.py [debug | train]")
    sys.exit(1)

if sys.argv[1] == "debug":
    from debug import debug

    debug()

elif sys.argv[1] == "train":
    from train import train

    train(
        episodes=1000,
        hyperparameters={
            "lr": 0.001,
            "epsilon_decay": 0.995,
            "gamma": 0.95,
            "epsilon_min": 0.1,
            "memory_size": 100000,
        },
    )
