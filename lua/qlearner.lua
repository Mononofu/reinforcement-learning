require 'common'

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

function QLearner:act(observation)
  q_action = self.Q[observationToString(observation)]
  return reduce(function(a, b) return q_action[a] > q_action[b] and a or b end,
                self.availableActions)
end

function QLearner:learn(observationRaw, action, newObservationRaw, reward, terminal)
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

return QLearner
