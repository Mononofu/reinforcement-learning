require 'nn'
local gfx = require 'gfx.js'

function defaultdict(default_value_factory)
  local t = {}
  local metatable = {}
  metatable.__index = function(t, key)
    if not rawget(t, key) then
      rawset(t, key, default_value_factory(key))
    end
    return rawget(t, key)
  end
  return setmetatable(t, metatable)
end

function reduce(f, xs)
  acc = xs[1]
  for i = 2, #xs do
    acc = f(acc, xs[i])
  end
  return acc
end

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

local QLearner = {}
QLearner.__index = QLearner

function QLearner.create(availableActions, learningRate, discountFactor)
  local learner = {}
  setmetatable(learner, QLearner)
  learner.Q = defaultdict(function(_) return defaultdict(function(_) return 0 end) end)
  learner.availableActions = availableActions
  learner.learningRate = learningRate
  learner.discountFactor = discountFactor
  return learner
end

function observationToString(observation)
  return table.concat(observation, "")
end

function QLearner:act(observation)
  q_action = self.Q[observationToString(observation)]
  return reduce(function(a, b) return q_action[a] > q_action[b] and a or b end,
                self.availableActions)
end

function QLearner:learn(observationRaw, action, newObservationRaw, reward)
  local max_q = - math.huge
  for i, a in pairs(self.availableActions) do
    local q = self.Q[observationToString(newObservationRaw)][a]
    if q > max_q then
      max_q = q
    end
  end
  local observation = observationToString(observationRaw)
  self.Q[observation][action] = (self.Q[observation][action] + self.learningRate *
    (reward + self.discountFactor * max_q - self.Q[observation][action]))
end

local numRepetitions = 200
local numEpisodes = 350
local rewards = torch.zeros(numEpisodes)
chartWindow = gfx.chart(rewards, {chart='line'})

for repetition = 1, numRepetitions do
  local catch = Catch.create(5)
  local agent = QLearner.create(catch:availableActions(), 0.1, 0.9)

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
      agent:learn(observation, action, new_observation, reward)
      if terminal then
        break
      end
    end

    reward_ma = 0.05 * reward + 0.95 * reward_ma
    rewards[i] = rewards[i] + 1.0 / (repetition + 1) * (reward_ma - rewards[i])
  end

  if repetition % 5 == 0 then
    gfx.chart(rewards, {chart='line', win=chartWindow})
  end
end
