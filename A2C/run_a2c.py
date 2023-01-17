import gym
import random
import wandb

from torch_a2c import A2C_agent

ENV = "CartPole-v1"
TRAIN_EPISODES = 2000
SEED = 42

random.seed(SEED)

wandb.init(project="cartpole-torch-a2c")


def train_agent():
    env = gym.make(ENV, render_mode="human")

    dim_observation_space = env.observation_space.shape[0]
    dim_action_space = env.action_space.n

    agent = A2C_agent(dim_observation_space, dim_action_space)

    episode = 0

    while episode < TRAIN_EPISODES:
        print(f"\n===== EPISODE {episode} =====")
        done = False
        episode_reward = 0
        state, _ = env.reset()
        steps = 0
        while not done:
            action = agent.pred_action(state)
            state_next, reward, done, _, _ = env.step(action)

            agent.save_experience(state, action, reward, state_next, done)

            state = state_next
            steps += 1
            episode_reward += reward

            if done:
                print(f"Cumulative reward = {episode_reward}")
                wandb.log({"episode_reward_sum": episode_reward, "episode": episode})
                state, _ = env.reset()
                episode += 1


if __name__ == '__main__':
    train_agent()
