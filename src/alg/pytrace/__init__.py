from . import tracer
from . import init
from . import optim
from .tracer import tracer
from .nn import Sequential
from .utils import load, save, set_seed, from_numpy
from .utils import prYellow, prGreen

__doc__ = """
Tracer, a simple auto-grad tools only designed for Sequential model
Construct Dynamic Graph
Author: Jianbai Ye skr~skr~
"""

try:
    logo = """
                                                         
        ,------.          ,--------.                             
        |  .--. ',--. ,--.'--.  .--',--.--. ,--,--. ,---. ,---.  
        |  '--' | \  '  /    |  |   |  .--'' ,-.  || .--'| .-. : 
        |  | --'   \   '     |  |   |  |   \ '-'  |\ `--.\   --. 
        `--'     .-'  /      `--'   `--'    `--`--' `---' `----' 
                 `---'               
    """
    # font: soft
    # refer to http://patorjk.com/software/taag/#p=display&f=Soft&t=PyTrace
    prGreen(logo)
except:
    prYellow("Welcome to PyTrace")