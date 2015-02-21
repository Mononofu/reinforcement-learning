require 'common'

local MonteCarlo = {}
MonteCarlo.__index = MonteCarlo

function MonteCarlo.create(availableActions, args)
  args = args or {}
  local learner = {}
  setmetatable(learner, MonteCarlo)
  learner.Q = defaultdict(function(_) return defaultdict(function(_) return 0 end) end)
  learner.availableActions = availableActions
  learner.n0 = args.n0 or 100
  learner.n_state = defaultdict(function(_) return 0 end)
  learner.n_action = defaultdict(function(_) return defaultdict(function(_) return 0 end) end)
  learner.visited = {}
  learner.reward = 0
  return learner
end

function MonteCarlo:act(observation)
  local state = observationToString(observation)
  self.n_state[state] = self.n_state[state] + 1
  local epsilon = self.n0 / (self.n0 + self.n_state[state])

  if torch.uniform() < epsilon then
    -- pick random action
    return self.availableActions[math.floor(
      torch.uniform(1, #self.availableActions + 1))]
  else
    local q_act = self.Q[state]
    -- pick highest value action
    return reduce(function(a, b) return q_act[a] > q_act[b] and a or b end,
                self.availableActions)
  end
end

function MonteCarlo:learn(observationRaw, action, newObservationRaw, reward, terminal)
  self.visited[#self.visited + 1] = {observationRaw, action}
  self.reward = self.reward + reward
  if not terminal then
    -- Monte Carlo only updates at the end of the episode, towards the actual
    -- return. (In contrast to TD, which updates at every step towards the
    -- estimated return).
    return
  end


  for i, visited in ipairs(self.visited) do
    local observationRaw, action = unpack(visited)
    local state = observationToString(observationRaw)
    self.n_action[state][action] = self.n_action[state][action] + 1
    local alpha = 1 / self.n_action[state][action]
    local q = self.Q[state][action]
    self.Q[state][action] = q + alpha * (self.reward - q)
  end

  self.visited = {}
  self.reward = 0
end


function MonteCarlo:visualizeQState()
  local q = torch.zeros(21, 21)
  for player_value = 1, 21 do
    for dealer_value = 1, 21 do
      local state = observationToString{player_value, dealer_value}
      local q_act = self.Q[state]
      local max_action = reduce(
        function(a, b) return q_act[a] > q_act[b] and a or b end,
        self.availableActions)
      q[player_value][dealer_value] = q_act[max_action]
    end
  end


  gnuplot.xlabel('Player')
  gnuplot.ylabel('Dealer')
  gnuplot.axis({1, 21, 1, 21})
  gnuplot.figure(1)
  gnuplot.splot(q)
end

return MonteCarlo
