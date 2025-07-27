-- Exercise set 4a:
--
--  * using type classes
--  * working with lists
--
-- Type classes you'll need
--  * Eq
--  * Ord
--  * Num
--  * Fractional
--
-- Useful functions:
--  * maximum
--  * minimum
--  * sort

module Set4a where

import Mooc.Todo
import Data.List
import Data.Ord
import qualified Data.Map as Map
import Data.Array
import Data.Bool (not)

------------------------------------------------------------------------------
-- Ex 1: implement the function allEqual which returns True if all
-- values in the list are equal.
--
-- Examples:
--   allEqual [] ==> True
--   allEqual [1,2,3] ==> False
--   allEqual [1,1,1] ==> True
--
-- PS. check out the error message you get with your implementation if
-- you remove the Eq a => constraint from the type!

allEqual :: Eq a => [a] -> Bool
allEqual [] = True
allEqual [_] = True
allEqual (x:y:ys) = (x==y) && allEqual (y:ys)

------------------------------------------------------------------------------
-- Ex 2: implement the function distinct which returns True if all
-- values in a list are different.
--
-- Hint: a certain function from the lecture material can make this
-- really easy for you.
--
-- Examples:
--   distinct [] ==> True
--   distinct [1,1,2] ==> False
--   distinct [1,2] ==> True

distinct :: Eq a => [a] -> Bool
distinct [] = True
distinct (x:xs) = check x xs && distinct xs
    where
        check _ [] = True
        check a (b:bs) = (a /= b) && check a bs

------------------------------------------------------------------------------
-- Ex 3: implement the function middle that returns the middle value
-- (not the smallest or the largest) out of its three arguments.
--
-- The function should work on all types in the Ord class. Give it a
-- suitable type signature.
--
-- Examples:
--   middle 'b' 'a' 'c'  ==> 'b'
--   middle 1 7 3        ==> 3
--   middle 7 3 1        ==> 3
--   middle 3 7 1        ==> 3
--   middle 3 1 7        ==> 3
middle :: Ord a => a -> a -> a -> a
middle a b c
    | a <= b && b <= c = b
    | a > b && b > c = middle c b a
    | a < b && b > c = middle c a b
    | otherwise      = middle b a c

------------------------------------------------------------------------------
-- Ex 4: return the range of an input list, that is, the difference
-- between the smallest and the largest element.
--
-- Your function should work on all suitable types, like Float and
-- Int. You'll need to add _class constraints_ to the type of range.
--
-- It's fine if your function doesn't work for empty inputs.
--
-- Examples:
--   rangeOf [4,2,1,3]          ==> 3
--   rangeOf [1.5,1.0,1.1,1.2]  ==> 0.5

rangeOf :: (Ord a, Num a) => [a] -> a
rangeOf [] = 0
rangeOf [x] = x
-- OP#1
-- rangeOf xs = maximum xs - minimum xs
-- OP#2
-- rangeOf (x:xs) = max x xs - min x xs
--     where
--         max a [] = a
--         max a (b:bs)
--             | a >= b = max a bs
--             | otherwise = max b bs
--         min a [] = a
--         min a (b:bs)
--             | a <= b = min a bs
--             | otherwise = min b bs
-- OP#3
rangeOf (x:xs) = maxVal - minVal
    where
        (maxVal, minVal) = foldl maxMin (x,x) xs
        maxMin (a, b) y = (max a y, min b y)

------------------------------------------------------------------------------
-- Ex 5: given a (non-empty) list of (non-empty) lists, return the longest
-- list. If there are multiple lists of the same length, return the list that
-- has the smallest _first element_.
--
-- (If multiple lists have the same length and same first element,
-- you can return any one of them.)
--
-- Give the function "longest" a suitable type.
--
-- Challenge: Can you solve this exercise without sorting the list of lists?
--
-- Examples:
--   longest [[1,2,3],[4,5],[6]] ==> [1,2,3]
--   longest ["bcd","def","ab"] ==> "bcd"

longest :: (Ord a) => [[a]] -> [a]
longest [] = []
longest (x:xs) = go x xs
  where
    go best [] = best
    go best (y:ys)
      | length y > length best = go y ys
      | length y == length best && head y < head best = go y ys
      | otherwise = go best ys
------------------------------------------------------------------------------
-- Ex 6: Implement the function incrementKey, that takes a list of
-- (key,value) pairs, and adds 1 to all the values that have the given key.
--
-- You'll need to add _class constraints_ to the type of incrementKey
-- to make the function work!
--
-- The function needs to be generic and handle all compatible types,
-- see the examples.
--
-- Examples:
--   incrementKey True [(True,1),(False,3),(True,4)] ==> [(True,2),(False,3),(True,5)]
--   incrementKey 'a' [('a',3.4)] ==> [('a',4.4)]

incrementKey :: (Eq k, Num v) => k -> [(k,v)] -> [(k,v)]
-- OP#1
-- incrementKey k [] = []
-- incrementKey k ((k', v): xs) 
--     | k==k' = (k, v+1):incrementKey k xs
--     | otherwise = (k', v):incrementKey k xs
-- OP#2
incrementKey k = map (\(k', v) -> if k == k' then (k,v+1) else (k',v))
------------------------------------------------------------------------------
-- Ex 7: compute the average of a list of values of the Fractional
-- class.
--
-- There is no need to handle the empty list case.
--
-- Hint! since Fractional is a subclass of Num, you have all
-- arithmetic operations available
--
-- Hint! you can use the function fromIntegral to convert the list
-- length to a Fractional

average :: Fractional a => [a] -> a
average xs =  sum xs / fromIntegral(length xs)

------------------------------------------------------------------------------
-- Ex 8: given a map from player name to score and two players, return
-- the name of the player with more points. If the players are tied,
-- return the name of the first player (that is, the name of the
-- player who comes first in the argument list, player1).
--
-- If a player doesn't exist in the map, you can assume they have 0 points.
--
-- Hint: Map.findWithDefault can make this simpler
--
-- Examples:
--   winner (Map.fromList [("Bob",3470),("Jane",2130),("Lisa",9448)]) "Jane" "Lisa"
--     ==> "Lisa"
--   winner (Map.fromList [("Mike",13607),("Bob",5899),("Lisa",5899)]) "Lisa" "Bob"
--     ==> "Lisa"

winner :: Map.Map String Int -> String -> String -> String
winner scores player1 player2
    | Map.findWithDefault 0 player1 scores >= 
        Map.findWithDefault 0 player2 scores = player1
    | otherwise = player2

------------------------------------------------------------------------------
-- Ex 9: compute how many times each value in the list occurs. Return
-- the frequencies as a Map from value to Int.
--
-- Challenge 1: try using Map.alter for this
--
-- Challenge 2: use foldr to process the list
--
-- Example:
--   freqs [False,False,False,True]
--     ==> Map.fromList [(False,3),(True,1)]

freqs :: (Eq a, Ord a) => [a] -> Map.Map a Int
-- OP#1
--freqs = foldr (\x -> Map.insertWith (+) x 1) Map.empty
-- OP#2
--freqs [] = Map.empty
--freqs (x:xs) = (Map.insertWith (+) x 1) (freqs xs)
-- OP#3
freqs = foldr (\x -> Map.alter increment x) Map.empty
    where
        increment Nothing = Just 1
        increment (Just n) = Just (n+1)
------------------------------------------------------------------------------
-- Ex 10: recall the withdraw example from the course material. Write a
-- similar function, transfer, that transfers money from one account
-- to another.
--
-- However, the function should not perform the transfer if
-- * the from account doesn't exist,
-- * the to account doesn't exist,
-- * the sum is negative,
-- * or the from account doesn't have enough money.
--
-- Hint: there are many ways to implement this logic. Map.member or
-- Map.notMember might help.
--
-- Examples:
--   let bank = Map.fromList [("Bob",100),("Mike",50)]
--   transfer "Bob" "Mike" 20 bank
--     ==> fromList [("Bob",80),("Mike",70)]
--   transfer "Bob" "Mike" 120 bank
--     ==> fromList [("Bob",100),("Mike",50)]
--   transfer "Bob" "Lisa" 20 bank
--     ==> fromList [("Bob",100),("Mike",50)]
--   transfer "Lisa" "Mike" 20 bank
--     ==> fromList [("Bob",100),("Mike",50)]

transfer :: String -> String -> Int -> Map.Map String Int -> Map.Map String Int
transfer from to amount bank
    | amount < 0 = bank
    | Map.notMember from bank = bank
    | Map.notMember to bank = bank
    | Map.findWithDefault 0 from bank < amount = bank
    | otherwise = Map.adjust (+ amount) to (Map.adjust (subtract amount) from bank)

------------------------------------------------------------------------------
-- Ex 11: given an Array and two indices, swap the elements in the indices.
--
-- Example:
--   swap 2 3 (array (1,4) [(1,"one"),(2,"two"),(3,"three"),(4,"four")])
--         ==> array (1,4) [(1,"one"),(2,"three"),(3,"two"),(4,"four")]

swap :: Ix i => i -> i -> Array i a -> Array i a
swap i j arr =
    let 
        vali = arr ! i
        valj = arr ! j
        elems = [(i, valj), (j, vali)]
    in arr // elems


------------------------------------------------------------------------------
-- Ex 12: given an Array, find the index of the largest element. You
-- can assume the Array isn't empty.
--
-- You may assume that the largest element is unique.
--
-- Hint: check out Data.Array.indices or Data.Array.assocs

maxIndex :: (Ix i, Ord a) => Array i a -> i
maxIndex arr = 
    let
        ls = assocs arr --[(1,"One"),(2,"Two"),(3,"Three")]
        vs = map (\(x,y) -> y) ls --["One","Two","Three"]
        maxV = maximum vs --"Two"
        filtered = filter (\(x,y) -> y==maxV) ls -- [(2,"Two")]
    in head (map (\(x,y) -> x) filtered) -- 2
