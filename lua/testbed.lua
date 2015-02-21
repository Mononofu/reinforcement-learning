require 'gnuplot'
require 'torch'

-- Environments
local catch = require 'catch'
local easy21 = require 'easy21'

-- Agents
local nnlearner = require 'nnlearner'
local randomlearner = require 'random'
local qlearner = require 'qlearner'
local monteCarlo = require 'monte_carlo'


local numRepetitions = 50
local numEpisodes = 20000000
local agentDefinitions = {
  -- {
  --   name = 'QLearner',
  --   factory = function(availableActions)
  --     return qlearner.create(availableActions, 0.1, 0.9)
  --   end,
  -- },
  -- {
  --   name = 'Random',
  --   factory = function(availableActions)
  --     return randomlearner.create(availableActions)
  --   end,
  -- },
  {
    name = 'MonteCarlo',
    factory = function(availableActions)
      return monteCarlo.create(availableActions)
    end,
  },
  -- {
  --   name = 'NNLearner 1',
  --   factory = function(availableActions)
  --     return nnlearner.create(availableActions, {hiddenLayers=1})
  --   end,
  -- },
  -- {
  --   name = 'NNLearner 2',
  --   factory = function(availableActions)
  --     return nnlearner.create(availableActions, {hiddenLayers=2})
  --   end,
  -- },
}
local rewards = torch.zeros(#agentDefinitions, numEpisodes)

for repetition = 1, numRepetitions do
  -- local environment = catch.create(5)
  local environment = easy21.create()
  local agents = {}
  for _, definition in pairs(agentDefinitions) do
    agents[#agents + 1] = definition.factory(environment:availableActions())
  end

  print(repetition)

  for agent_i = 1, #agents do
    local timer = torch.Timer()
    local agent = agents[agent_i]
    local reward_ma = 0
    for i = 1, numEpisodes do
      environment:reset()
      reward = 0.0
      terminal = false
      while not terminal do
        local observation = environment:observe()
        local action = agent:act(observation, reward_ma)
        reward, terminal = environment:act(action)
        local new_observation = environment:observe()
        agent:learn(observation, action, new_observation, reward, terminal)
      end

      reward_ma = 0.01 * reward + 0.99 * reward_ma

      -- iterative mean from http://www.heikohoffmann.de/htmlthesis/node134.html
      rewards[agent_i][i] = (rewards[agent_i][i] + 1.0 / repetition *
        (reward_ma - rewards[agent_i][i]))

      if i % 10000 == 0 then
        local name = agentDefinitions[agent_i].name
        print(name .. ': ' .. rewards[agent_i][i])
        if name == 'MonteCarlo' then
          agent:visualizeQState()
        end
      end

    end

    if repetition % 1 == 0 then
      data = {}
      for i = 1, #agents do
        local reward = rewards[i]
        table.insert(data, {
          agentDefinitions[i]['name'],
          torch.range(1, numEpisodes),
          reward,
          '-',
        })
      end
      gnuplot.axis({0, numEpisodes, -1, 1})
      gnuplot.figure(0)
      gnuplot.plot(data)
    end

  end
end
