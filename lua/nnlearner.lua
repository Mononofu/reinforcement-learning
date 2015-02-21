require 'nn'
require 'torch'

local NNLearner = {}
NNLearner.__index = NNLearner

function NNLearner.create(availableActions, args)
  local learner = {}
  setmetatable(learner, NNLearner)
  learner.availableActions = availableActions
  learner.learningRate = args.learningRate or 0.2
  learner.discountFactor = args.discountFactor or 0.9
  learner.explorationRate = args.explorationRate or 0.05

  learner.screen = torch.zeros(25)
  learner.criterion = nn.MSECriterion()
  learner.target = torch.zeros(3)

  learner.mlp = nn.Sequential();  -- make a multi-layer perceptron
  local inputs = 25
  local outputs = #availableActions
  local HUs = args.HUs or 20
  learner.mlp:add(nn.Linear(inputs, HUs))
  learner.mlp:add(args.transfer or nn.ReLU())
  for i = 1, args.hiddenLayers - 1 do
    learner.mlp:add(nn.Linear(HUs, HUs))
    learner.mlp:add(args.transfer or nn.ReLU())
  end
  learner.mlp:add(nn.Linear(HUs, outputs))
  return learner
end

function NNLearner:q(observation)
  for i = 1, 25 do
    self.screen[i] = 0
  end
  self.screen[observation[1] * 5 + observation[2] + 1] = 1
  self.screen[21 + observation[3]] = -1
  return self.mlp:forward(self.screen)
end

local function maxAction(q, availableActions)
  local max_q = - math.huge
  local action_index = -1
  for i, a in pairs(availableActions) do
    if q[i] > max_q then
      max_q = q[i]
      action_index = i
    end
  end
  return max_q, action_index
end

function NNLearner:act(observation, rewardMa)
  if torch.uniform() < math.min(self.explorationRate, (1 - rewardMa) / 4) then
    return self.availableActions[math.floor(
      torch.uniform(1, #self.availableActions + 1))]
  end
  local _, action_index = maxAction(self:q(observation), self.availableActions)
  return self.availableActions[action_index]
end

function NNLearner:learn(observation, action, newObservation, reward, terminal)
  local q = self:q(newObservation)
  local max_q, _ = maxAction(q, self.availableActions)
  local action_index = -1
  for i, a in pairs(self.availableActions) do
    if a == action then
      action_index = i
    end
  end


  local pred = self:q(observation)
  for i = 1, pred:size(1) do
    self.target[i] = pred[i] + self.learningRate * (q[i] - pred[i])
  end
  if terminal then
    self.target[action_index] = reward
  else
    self.target[action_index] = pred[action_index] + self.learningRate * (reward + self.discountFactor * max_q - pred[action_index])
  end

  local err = self.criterion:forward(pred, self.target)
  local gradCriterion = self.criterion:backward(pred, self.target)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  self.mlp:zeroGradParameters()
  -- (2) accumulate gradients
  self.mlp:backward(self.screen, gradCriterion)
  -- (3) update parameters with learning rate
  self.mlp:updateParameters(self.learningRate)
end

function NNLearner:visualize()
  local bat = 2
  for action = 1, 3 do
    local gradient = torch.zeros(5, 5)
    for ballH = 0, 4 do
      for ballW = 0, 4 do
        local q = self:q({ballH, ballW, bat})
        gradient[ballH + 1][ballW + 1] = q[action]
      end
    end
    gnuplot.figure(action + 1)
    gnuplot.imagesc(gradient, 'color')
  end
  -- body
end

return NNLearner
