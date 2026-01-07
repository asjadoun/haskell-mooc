{-
---
fulltitle: "In class exercise: Practice with Maybe Monad"
---

The goal of this short in-class exercise is to get a bit more practice with using the `Maybe` Monad.
-}

module MaybePractice where

import Control.Monad qualified as Monad
import Data.Map (Map)
import Data.Map qualified as Map
import Text.Read qualified as Text
import Control.Applicative ( Alternative((<|>)) )

{-
Part 1
------

Consider the `Weather` datatype from the Kata problem from HW #3.
-}

data Weather = Weather
  { dayNumber :: Int,
    maxTemp :: Int,
    minTemp :: Int
  }
  deriving (Eq, Show)

{-
Use `Map.lookup` and `Text.readMaybe` to write a function that takes a
dictionary (i.e. a finite map from String keys to String values) and converts
it into weather data. Your implementation should not use pattern matching;
use `do` notation instead.

If the provided dictionary does not have the appropriate data, then the
function should return `Nothing`. In other words, this function should behave
according to the examples below.
-}

-- >>> parseWeather (Map.fromList [("day", "2"), ("maxTemp", "78"), ("minTemp", "62")])
-- Just (Weather {dayNumber = 2, maxTemp = 78, minTemp = 62})
-- >>> parseWeather (Map.fromList [("day", "2")])
-- Nothing
-- >>> parseWeather (Map.fromList [("day", "two"), ("maxTemp", "78"), ("minTemp", "62")])
-- Nothing

parseWeather :: Map String String -> Maybe Weather
parseWeather dict = do
  dayStr <- Map.lookup "day" dict
  maxTempStr <- Map.lookup "maxTemp" dict
  minTempStr <- Map.lookup "minTemp" dict

  day <- Text.readMaybe dayStr :: Maybe Int
  maxTemp <- Text.readMaybe maxTempStr :: Maybe Int
  minTemp <- Text.readMaybe minTempStr :: Maybe Int
  
  return (Weather day maxTemp minTemp)

{-
Part 2
------

Consider the following operations that combine two `Maybe`s. Implement them both.
One can be defined using `(>>=)` from the `Maybe` monad, and one cannot. Which is which?

-}

-- | Left-biased choice on maybes
--
-- >>> firstJust (Just 1) (Just 2)
-- Just 1
-- >>> firstJust Nothing (Just 2)
-- Just 2
-- >>> firstJust (Just 1) Nothing
-- Just 1
-- >>> firstJust Nothing Nothing
-- Nothing
firstJust :: Maybe a -> Maybe a -> Maybe a
firstJust x y = case x of
  Just _ -> x
  Nothing -> y

-- using Alternative
-- firstJust x y = x <|> y

-- pattern match solution
-- firstJust f@(Just _) _ = f
-- firstJust Nothing f@(Just _) = f
-- firstJust _ _ = Nothing

-- | Ensure that both Maybes are 'Just' and retain the first one
--
-- >>> sequenceFirst (Just 1) (Just 'a')
-- Just 1
-- >>> sequenceFirst Nothing (Just 'a')
-- Nothing
-- >>> sequenceFirst (Just 1) Nothing
-- Nothing
-- >>> sequenceFirst Nothing Nothing
-- Nothing
sequenceFirst :: Maybe a -> Maybe b -> Maybe a
sequenceFirst v1 v2 = v2 >>= return v1

-- sequenceFirst v1 v2 = v2 >>= return (v1 >>= (return ))

-- sequenceFirst mx my = mx >>= \x -> my >>= \_ -> return x