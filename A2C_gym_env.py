import gym
from A2C import *
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")
""" 4 states, 2 actions """

agent = Agent()

n_games = 500
episodes = []
scores = []
episode_counter = 0

for i_episode in range(n_games):
    state = env.reset()
    done = False
    rewards = []
    dones = []
    score = 0.
    while not done:
        if episode_counter >= n_games-3:  # Show last 3 attempts
            env.render()

        action = agent.step(state)
        next_state, reward, done, info = env.step(action)
        score += reward
        rewards.append(reward)
        dones.append(done)
        state = next_state
        if done:
            agent.sample_trajectory(rewards, dones)
            scores.append(score)
            episodes.append(episode_counter)
            episode_counter += 1
            print(episode_counter, ' FINAL SCORE: ', score)
            break

env.close()

plt.title('Score vs Episode')
plt.plot(episodes, scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid()
plt.show()
