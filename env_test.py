# import gym
# env = gym.make('SpaceInvaders-ram-v0')
# # print(env.action_space)
# #> Discrete(2)
# # print(env.observation_space)
# # print(env.action_space.sample())
# from gym import spaces 
# space = spaces.Discrete(7)
# x = space.sample()
# print (env.action_space)
# # print (space.n)

import gym
env = gym.make('SpaceInvaders-ram-v0')
observation = env.reset()
print(observation)
print(observation.shape)
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation.shape)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         # if done:
#         #     print("Episode finished after {} timesteps".format(t+1))
#         #     break
env.close()