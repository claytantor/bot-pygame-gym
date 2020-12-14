# bot-pysc2
work is based on this DQN example from openai: 

* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


## using python 3
you need the ssl system packages because IOT requires ssl

```
brew install python@3.9.0
pyenv install 3.9.0
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## runing the move to beacon simulation
`python runner_ple.py`