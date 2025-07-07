# CRAIS-MARL

## Setup steps for Linux

- Clone the repo
```bash
git clone https://github.com/nishantHolla/cRAIS-MARL.git
cd  ./cRAIS-MARL
```

- Create env and activate it
```bash
python -m venv .cRAIS-MARL
source ./.cRAIS-MARL/bin/activate
```

- Install libraries
```bash
pip install -r requirements.txt
```

- Run the simulation step-by-step
```bash
python main.py debug
```

- Train the DQN network with the parameters defined in `main.py`
```bash
python main.py train
```

- Evaluate the trained model with the parameters defined in `main.py`
```bash
python main.py eval
```
