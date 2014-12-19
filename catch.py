import abc
import collections
import functools
import random

import matplotlib.pyplot as plt


class Environment(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def available_actions(self):
    return []

  @abc.abstractmethod
  def reset(self):
    pass

  @abc.abstractmethod
  def observe(self):
    return []

  @abc.abstractmethod
  def act(self, action):
    """Returns (reward, is_terminal)."""
    return 0.0, True


class Catch(Environment):

  def __init__(self, width):
    self.width = width
    self.height = width
    self.reset()

  def available_actions(self):
    return [-1, 0, 1]

  def reset(self):
    self.ball_position = [0, random.randint(0, self.width - 1)]
    self.bat_position = random.randint(0, self.width - 1)

  def observe(self):
    return self._render(self.ball_position[0], self.ball_position[1],
                        self.bat_position)

  def _render(self, ball_h, ball_w, bat_w):
    screen = []
    for h in range(self.height):
      row = []
      for w in range(self.width):
        if h == ball_h and w == ball_w:
          row.append('*')
        elif w == self.bat_position and h == self.height - 1:
          row.append('_')
        else:
          row.append(' ')
      screen.append(row)
    return screen

  def act(self, action):
    self.bat_position = max(0, min(self.width - 1, self.bat_position + action))
    self.ball_position[0] += 1
    if self.ball_position[0] == self.height - 1:
      if self.bat_position == self.ball_position[1]:
        return 1, True
      else:
        return -1, True
    return 0, False


class QLearner(object):
  """Off-Policy TD Control, page 158."""

  def __init__(self, available_actions, learning_rate, discount_factor):
    # action_value_estimates
    self.Q = collections.defaultdict(lambda: collections.defaultdict(int))
    self.available_actions = available_actions
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor

  def act(self, observation):
    q_action = self.Q[self._condense_observation(observation)]
    return functools.reduce(lambda a, b: a if q_action[a] > q_action[b] else b,
                            self.available_actions)

  def learn(self, observation_raw, action, new_observation_raw, reward):
    observation = self._condense_observation(observation_raw)
    new_observation = self._condense_observation(new_observation_raw)
    max_q = max([self.Q[new_observation][a] for a in self.available_actions])
    self.Q[observation][action] += self.learning_rate * \
        (reward + self.discount_factor * max_q - self.Q[observation][action])

  def _condense_observation(self, observation):
    return '\n'.join([''.join(row) for row in observation])

  def __str__(self):
    return 'GreedyAgent'

num_episodes = 350
num_repetitions = 200

plt.ion()
x = range(num_episodes)
rewards = [0 for _ in range(num_episodes)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([-1.1, 1.1])
line = ax.plot(x, rewards, '-')[0]


for repetition in range(num_repetitions):
  catch = Catch(5)
  agent = QLearner(catch.available_actions(), learning_rate=0.1,
                   discount_factor=0.9)
  reward_ma = 0
  for i in range(num_episodes):
    terminal = False
    catch.reset()

    while not terminal:
      observation = catch.observe()
      action = agent.act(observation)
      reward, terminal = catch.act(action)
      new_observation = catch.observe()
      agent.learn(observation, action, new_observation, reward)

    reward_ma = 0.05 * reward + 0.95 * reward_ma

    rewards[i] += 1.0 / (repetition + 1) * (reward_ma - rewards[i])
  if repetition % 3 == 0:
    line.set_ydata(rewards)
    fig.canvas.draw()

raw_input()
