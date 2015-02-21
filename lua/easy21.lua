require 'torch'

local Easy21 = {}
Easy21.__index = Easy21

function Easy21.create(width)
  local easy21 = {}
  setmetatable(easy21, Easy21)
  return easy21
end

-- actions
function Easy21:availableActions()
  return {'stick', 'hit'}
end

function Easy21:reset()
  -- start off with one black card each
  self.player = math.ceil(torch.uniform(0, 10))
  self.dealer = math.ceil(torch.uniform(0, 10))
end

-- observations
function Easy21:observe()
  return {self.player, self.dealer}
end

function drawCard(initial_score)
  local card_value = math.ceil(torch.uniform(0, 10))
  if torch.uniform() < 1 / 3 then
    -- red card, subtract
    return -1 * card_value
  else
    -- black card, add
    return  1 * card_value
  end
end

function isBust(score)
  return score < 1 or score > 21
end

-- reward, isTerminal
function Easy21:act(action)
  if action == 'hit' then
    self.player = self.player + drawCard()
    if isBust(self.player) then
      return -1, true
    else
      return 0, false
    end
  elseif action == 'stick' then
    -- play out dealer. Dealer sticks on 17 or higher, hits otherwise.
    while self.dealer < 17 do
      self.dealer = self.dealer + drawCard()
    end
    if isBust(self.dealer) or self.player > self.dealer then
      return 1, true
    elseif self.dealer > self.player then
      return -1, true
    else
      return 0, true
    end
  else
    error('unknown action: ' .. action)
  end
end

return Easy21
