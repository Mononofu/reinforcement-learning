require 'gnuplot'
require 'profiler'
require 'torch'

local nnlearner = require 'nnlearner'
local randomlearner = require 'random'
local qlearner = require 'qlearner'


local Catch = {}
Catch.__index = Catch

function Catch.create(width)
  local catch = {}
  setmetatable(catch, Catch)
  catch.width = width
  catch.height = width
  return catch
end

-- actions
function Catch:availableActions()
  return {-1, 0, 1}
end

function Catch:reset()
  self.ballPosition = {0, math.floor(torch.uniform(0, self.width))}
  self.batPosition = math.floor(torch.uniform(0, self.width))
end

-- observations
function Catch:observe()
  return {self.ballPosition[1], self.ballPosition[2], self.batPosition}
end

-- reward, isTerminal
function Catch:act(action)
  self.batPosition = math.max(0, math.min(self.width - 1, self.batPosition + action))
  self.ballPosition[1] = self.ballPosition[1] + 1
  if self.ballPosition[1] == self.height - 1 then
    if self.batPosition == self.ballPosition[2] then
      return 1.0, true
    else
      return -1.0, true
    end
  end
  return 0.0, false
end


local numRepetitions = 2
local numEpisodes = 40000
local agentDefinitions = {
  {
    name = 'QLearner',
    factory = function(availableActions)
      return qlearner.create(availableActions, 0.1, 0.9)
    end,
  },
  {
    name = 'NNLearner 0.01',
    factory = function(availableActions)
      return nnlearner.create(availableActions, 0.01, 0.9, 0.05)
    end,
  },
  {
    name = 'NNLearner 0.05',
    factory = function(availableActions)
      return nnlearner.create(availableActions, 0.05, 0.9, 0.05)
    end,
  },
  {
    name = 'NNLearner 0.10',
    factory = function(availableActions)
      return nnlearner.create(availableActions, 0.10, 0.9, 0.05)
    end,
  },
  {
    name = 'Random',
    factory = function(availableActions)
      return randomlearner.create(availableActions)
    end,
  },
}
local rewards = torch.zeros(#agentDefinitions, numEpisodes)
gnuplot.axis({0, numEpisodes, -1, 1})
profiler.start()

for repetition = 1, numRepetitions do
  local catch = Catch.create(5)
  local agents = {}
  for _, definition in pairs(agentDefinitions) do
    agents[#agents + 1] = definition.factory(catch:availableActions())
  end

  for agent_i = 1, #agents do
    local timer = torch.Timer()
    local agent = agents[agent_i]
    local reward_ma = 0
    for i = 1, numEpisodes do
      catch:reset()
      reward = 0.0
      terminal = false
      while not terminal do
        local observation = catch:observe()
        local action = agent:act(observation)
        reward, terminal = catch:act(action)
        local new_observation = catch:observe()
        agent:learn(observation, action, new_observation, reward, terminal)
      end

      reward_ma = 0.01 * reward + 0.99 * reward_ma
      -- if i % 100 == 0 then
      --   print(i .. ': ' .. reward_ma)
      -- end
      rewards[agent_i][i] = (rewards[agent_i][i] + 1.0 / (repetition + 1) *
        (reward_ma - rewards[agent_i][i]))

      -- if i % 100 == 0 and agentDefinitions[agent_i]['name'] == 'NNLearner' then
      --   agent:visualize()
      -- end
    end
    print(agentDefinitions[agent_i].name .. ' took ' .. timer:time().real)

    if repetition % 1 == 0 then
      data = {}
      for i = 1, #agents do
        local reward = rewards[i]
        table.insert(data, {
          agentDefinitions[i]['name'],
          torch.range(1, numEpisodes),
          reward,
        })
      end
      gnuplot.figure(0)
      gnuplot.plot(data)
    end

  end
end

profiler.stop()
