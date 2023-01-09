import gym
import random
import wandb

from torch_agents import DQN_agent
# from keras_agents import DQN_agent

TRAIN_EPISODES = 2000
SEED = 42

random.seed(SEED)

# wandb.init(project="cartpole-dqn")


def train_dqn():
    env = gym.make("CartPole-v1", render_mode="human")

    dim_observation_space = env.observation_space.shape[0]
    dim_action_space = env.action_space.n

    agent = DQN_agent(dim_observation_space, dim_action_space, alpha=0.00025)

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

            reward = reward if not done else -reward
            # print(f'Step {steps}: {state}, {reward}, {done}')

            agent.save_experience(state, action, reward, state_next, done)

            state = state_next
            steps += 1
            episode_reward += reward

            if done:
                print(f"Cumulative reward = {episode_reward}")
                # wandb.log({"episode_reward_sum": episode_reward, "episode": episode})
                state, _ = env.reset()
                episode += 1


if __name__ == '__main__':
    train_dqn()
