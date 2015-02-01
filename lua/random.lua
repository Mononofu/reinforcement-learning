require 'common'

local Random = {}
Random.__index = Random

function Random.create(availableActions)
  local learner = {}
  setmetatable(learner, Random)
  learner.availableActions = availableActions
  return learner
end


function Random:act(observation)
  return self.availableActions[math.floor(
    torch.uniform(1, #self.availableActions))]
end

function Random:learn(observationRaw, action, newObservationRaw, reward, terminal)
end

return Random
