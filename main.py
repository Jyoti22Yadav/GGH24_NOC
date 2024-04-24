import numpy as np
from simulator import Simulator
from rl_agent import RLAgent

NUM_EPISODES = 1000  # Number of episodes for training

def main():
    # Initialize simulator and RL agent
    simulator = Simulator()
    rl_agent = RLAgent()

    # Training loop
    for episode in range(NUM_EPISODES):
        state = simulator.reset()  # Reset simulator and get initial state
        done = False

        while not done:
            # RL agent selects action based on current state
            action = rl_agent.select_action(state)

            # Execute action in simulator and get next state, reward, done flag
            next_state, reward, done = simulator.step(action)

            # RL agent learns from experience
            rl_agent.train(state, action, reward, next_state, done)

            # Update current state
            state = next_state

        # Evaluate model periodically
        if episode % 50 == 0:
            evaluate_model(rl_agent, simulator)

    # Save trained model
    rl_agent.save_model("rl_model.pth")

def evaluate_model(rl_agent, simulator):
    total_rewards = []
    for _ in range(10):  # Evaluate on 10 test workloads
        state = simulator.reset()
        done = False
        total_reward = 0

        while not done:
            action = rl_agent.select_action(state)
            next_state, reward, done = simulator.step(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    average_reward = np.mean(total_rewards)
    print(f"Average Reward: {average_reward}")

if __name__ == "__main__":
    main()



