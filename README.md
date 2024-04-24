# GGH24_NOC
This repository contains the code implementation for optimizing a Network on Chip (NOC) design using a simulator and Reinforcement Learning (RL). The goal is to design an efficient NOC for a System on a Chip (SoC), optimizing for latency, bandwidth, buffer occupancy, and throttling.

Files and Directory Structure
src/: Contains the source code for the simulator and RL agent.
main.py: Main script for simulator interaction and RL training.
simulator.py: Simulator interface and APIs.
rl_agent.py: Proximal Policy Optimization (PPO) RL agent.
docs/: Contains documentation and results.
README.md: Instructions and setup guide.
design_document.pdf: Detailed design document.
results/: Directory for storing result files.
results.txt: Sample result file showing RL agent's performance.
requirements.txt: List of required dependencies.
Setup and Installation
Clone this repository:
git clone https://github.com/your-username/noc-rl-optimization.git cd noc-rl-optimization


Install dependencies:
pip install -r requirements.txt


Run the main script:
python src/main.py


Running the Code
The main.py script interacts with the simulator, trains the RL agent, and evaluates its performance.
The simulator is defined in simulator.py which simulates the NOC components and interactions.
The RL agent, using PPO algorithm, is implemented in rl_agent.py with a policy and value network.
Results
The results/ directory contains a sample result file (results.txt) showing the RL agent's performance on test workloads.
You can update this file with your actual results obtained during training.
Documentation
The docs/ directory contains:
design_document.pdf: Detailed design document outlining the problem statement, simulator setup, RL framework, and implementation details.
README.md: Instructions for running the code and details about the repository.
References
Reinforcement Learning: An Introduction by Sutton and Barto
PyTorch Documentation
OpenAI Spinning Up in Deep RL
