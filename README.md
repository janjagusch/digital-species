# On the Origin of Digital Species

This repository contains the intelligent agent for the Atari game Breakout, presented at Tedx IST Alamdeda 2018 in Lisbon, Portugal. In my talk, I explained the analogy between natural evolution and simulated evolution. I illustrated, how humanity has bred wolves into dogs and how we can use the same principles in computer science to develop artificial intelligence that excels at playing video games from the 1970s.

## Getting Started

If you want to run the code on your personal machine, please follow these instructions.

### Prerequisites

The backbone of the program in the [OpenAI Gym](https://gym.openai.com/). Please follow the [official installation guidelines](https://github.com/openai/gym#installation). If you work on a Windows machine, please follow the instruction [here](https://github.com/openai/gym/issues/11).	

The underlying algorithm is [Neuroevolution of Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). We use the [Python implementation](http://neat-python.readthedocs.io/en/latest/) of this algorithm. The installation guidelines can be found [here](http://neat-python.readthedocs.io/en/latest/installation.html).

Additional third party dependencies are [numpy](http://neat-python.readthedocs.io/en/latest/installation.html), [matplotlib](https://matplotlib.org/) and [tqdm](https://pypi.org/project/tqdm/). They can be installed as follows, using [Anaconda](https://anaconda.org/):

```
conda install numpy
conda install matplotlib
pip install tqdm
```

## Running the program

The evolve.py module in the src folder is the central module for this program. From there, you can start new runs and continue from an existing checkpoint.

### Starting a new run.

If you want to start with a new run, call the start_run() function. This creates a new NEAT population, based on the defined [configuration](http://neat-python.readthedocs.io/en/latest/config_file.html).

### Continuing from existing checkpoint.

If you want to continue from an existing checkpoint, call the continue_run(filename) function. This continues an existing run from a checkpoint file, described in filename.

## Authors

* **Jan-Benedikt Jagusch** - [jbj2505](https://github.com/jbj2505)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
