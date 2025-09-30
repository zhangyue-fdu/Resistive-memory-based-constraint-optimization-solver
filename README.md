# Resistive-memory-based-constraint-optimization-solver
Code for "Resistive memory based constraint optimizaation solver for robotic manipulation"

## Abstract
The core of impedance regulation, model-predictive control, and manipulation planning lies in ensuring that the target policy satisfies constraints imposed by both robot hardware and the physical environment. This allows the system to approach the target object while avoiding collisions with others, naturally formulating the problem as a constrained optimization task. However, solving such optimization problems on von Neumann processors-- which separate memory from logic, rely on serial constraint validation, and depend on iterative numerical methods with discrete time steps and truncation errors-- remains orders of magnitude slower and less energy-efficient than biological motor systems. To address this gap, we introduce a resistive-memory-based computing-in-memory solver that embeds constraint optimization directly into analog circuits. The collocation of memory and processing is achieved using analog in-memory computing. We map the problem to a system of differential equations and solve the equations using the continous-time dynamics of the circuit, processing all constraints in parallel. The system naturally relaxes to the constrained optimum, eliminating the need for serial iterative loops as well as intermediate discretization and digitization. Fabricated in a 180 nm resistive memory array, the solver achieves a 90 µs response time and consumes only 14.7 µJ when executing a manipulation task on a seven-degree-of-freedom UR-5 robot,  cutting latency and energy by factors of $1.3 \times 10^{4}$ and $5.9 \times 10^{6}$, respectively, compared with state-of-the-art digital hardware. This resistive memory based solver demonstrates a scalable pathway toward brain-level action selection and manipulation performance for embodied intelligence.

# Requirements
The codes run on Ubuntu 20.04 and Windows10, cuda 11.2 with the following packages:

## Setup Instructions

The keypoint and constraints generation is from the official demo code for ReKep implemented in OmniGibson [Project Page]](https://rekep-robot.github.io/). Note that this codebase is best run with a display. For running in headless mode, refer to the [instructions in OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html).

- Install [OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html). This code is tested on [this commit](https://github.com/StanfordVL/OmniGibson/tree/cc0316a0574018a3cb2956fcbff3be75c07cdf0f).

NOTE: If you encounter the warning `We did not find Isaac Sim under ~/.local/share/ov/pkg.` when running `./scripts/setup.sh` for OmniGibson, first ensure that you have installed Isaac Sim. Assuming Isaac Sim is installed in the default directory, then provide the following path `/home/[USERNAME]/.local/share/ov/pkg/isaac-sim-2023.1.1` (replace `[USERNAME]` with your username).

- Install ReKep in the same conda environment:
```Shell
conda activate omnigibson
cd ..
git clone https://github.com/huangwl18/ReKep.git
cd ReKep
pip install -r requirements.txt
```

- Obtain an [OpenAI API](https://openai.com/blog/openai-api) key and set it up as an environment variable:
```Shell
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

