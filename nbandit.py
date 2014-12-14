#!/usr/bin/python2
import abc
import functools
import random
import time

import matplotlib.pyplot as plt


class Environment(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def available_actions(self):
    return []

  @abc.abstractmethod
  def act(self, action):
    return 0.0


class Bandit(Environment):

  def __init__(self, rewards, noise_std):
    self.rewards = rewards
    self.n = len(rewards)
    self.noise_std = noise_std

  def available_actions(self):
    return range(self.n)

  def act(self, action):
    return self.rewards[action] + random.gauss(0, self.noise_std)

  def __str__(self):
    return 'Bandit(%s, noise_std=%s)' % (self.rewards, self.noise_std)


class GreedyAgent(object):

  def __init__(self, num_actions):
    # action_value_estimates
    self.Q = [0 for _ in range(num_actions)]
    self.action_count = [1 for _ in range(num_actions)]

  def act(self):
    return functools.reduce(lambda a, b: a if self.Q[a] > self.Q[b] else b,
                            range(len(self.Q)))

  def learn(self, action, reward):
    self.Q[action] += 1.0 / self.action_count[action] * \
        (reward - self.Q[action])
    self.action_count[action] += 1

  def __str__(self):
    return 'GreedyAgent'


class EpsilonGreedyAgent(GreedyAgent):

  def __init__(self, num_actions, epsilon):
    super(EpsilonGreedyAgent, self).__init__(num_actions)
    self.epsilon = epsilon
    self.num_actions = num_actions

  def act(self):
    if random.random() < self.epsilon:
      return random.randint(0, self.num_actions - 1)
    else:
      return super(EpsilonGreedyAgent, self).act()

  def __str__(self):
    return 'EpsilonGreedyAgent(epsilon=%f)' % self.epsilon


class Testbed(object):

  def __init__(self, environment_fn, agent_fns):
    self.environment_fn = environment_fn
    self.agent_fns = agent_fns

  def evaluate(self, num_episodes, episode_length, plot=False):
    if plot:
      plt.ion()
      x = range(episode_length)
      self.rewards = [
          [0 for _ in range(episode_length)] for _ in self.agent_fns]
      self.fig = plt.figure()
      ax = self.fig.add_subplot(111)
      ax.set_ylim([0, 1.5])

      # Create some fake agents for the labels.
      environment = self.environment_fn()
      agents = [f(len(environment.available_actions()))
                for f in self.agent_fns]
      self.lines = [ax.plot(x, rewards, '-', label=str(agent))[0]
                    for rewards, agent in zip(self.rewards, agents)]
      ax.legend(loc=8)

    start = time.time()
    for episode in xrange(num_episodes):
      environment = self.environment_fn()
      agents = [f(len(environment.available_actions()))
                for f in self.agent_fns]
      for step in xrange(episode_length):
        for i, agent in enumerate(agents):
          action = agent.act()
          reward = environment.act(action)
          agent.learn(action, reward)

          if plot:
            self.rewards[i][
                step] += (1.0 / (episode + 1)) * (reward - self.rewards[i][step])

      if plot and episode % 20 == 0:
        for line, rewards in zip(self.lines, self.rewards):
          line.set_ydata(rewards)
        self.fig.canvas.draw()

      if episode and episode % 100 == 0:
        print 'Episode %d: %.2f episodes/second' % (episode, episode / (time.time() - start))


testbed = Testbed(
    environment_fn=lambda: Bandit([random.gauss(0, 1) for _ in range(10)], 1),
    agent_fns=[lambda n: GreedyAgent(n),
               lambda n: EpsilonGreedyAgent(n, 0.1),
               lambda n: EpsilonGreedyAgent(n, 0.01)])
testbed.evaluate(num_episodes=2000, episode_length=1000, plot=True)

print 'Done, press enter to exit.'
raw_input()
