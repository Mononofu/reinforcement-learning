require 'torch'

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

return Catch
