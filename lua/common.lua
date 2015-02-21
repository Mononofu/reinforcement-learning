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

function observationToString(observation)
  return table.concat(observation, "")
end
