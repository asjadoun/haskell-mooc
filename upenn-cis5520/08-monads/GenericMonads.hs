{-
---
fulltitle: "In class exercise: General Monadic Functions"
date: October 23, 2024
---
-}

module GenericMonads where

import Data.Char qualified as Char
import Test.HUnit
import Text.Read (readMaybe)
import Prelude hiding (mapM, sequence, sequenceA, liftA, liftA2)

{-
Generic Monad Operations
========================

This problem asks you to recreate some of the operations in the
[Control.Monad](https://hackage.haskell.org/package/base-4.14.2.0/docs/Control-Monad.html)
library. You should *not* use any of the functions defined in that library to
solve this problem.  (These functions also appear in more general forms
elsewhere, so other libraries that are off limits for this problem include
`Control.Applicative`, `Data.Traversable` and `Data.Foldable`.)

NOTE: because these operations are so generic, the types will really help you
figure out the implementation, even if you don't quite know what the function should do.

For that reason you should also test *each* of these functions with at least two unit test
cases, one using the `Maybe` monad, and one using the `List` monad.  After you
you have tried the function out, try to describe in words what each operation does
for that specific monad.

Here is the first one as an example.
-}

-- (a)

{-
Given the type signature:
-}

mapM :: (Monad m) => (a -> m b) -> [a] -> m [b]
{-
We implement it by recursion on the list argument.
-}

mapM _ [] = return []
mapM f (x : xs) = do
  b <- f x
  bs <- mapM f xs
  return (b : bs)

{-
Then define the following test cases, which make use of the following
helper functions.
-}

maybeUpper :: Char -> Maybe Char
maybeUpper x = if Char.isAlpha x then Just (Char.toUpper x) else Nothing

onlyUpper :: [Char] -> [Char]
onlyUpper = filter Char.isUpper

-- >>> mapM maybeUpper "sjkdhf"
-- Just "SJKDHF"
-- >>> mapM maybeUpper "sa2ljsd"
-- Nothing

-- >>> mapM onlyUpper ["QuickCheck", "Haskell"]
-- ["QH","CH"]
-- >>> mapM onlyUpper ["QuickCheck", ""]
-- []

{-
Finally, we observe that this function is a generalization of List.map, where
the mapped function can return its value in some monad m.
-}

-- (b)

foldM :: (Monad m) => (a -> b -> m a) -> a -> [b] -> m a
-- foldM _ a [] = return a
-- foldM f a (b:bs) = f a b >>= (\a' -> foldM f a' bs)
-- OR
foldM _ a [] = return a
foldM f a (b:bs) = do
  a' <- f a b
  foldM f a' bs

-- (c)
-- >>> sequence [(Just 1), (Just 3)]

sequence :: (Monad m) => [m a] -> m [a]
-- sequence [] = return []
-- sequence (a:as) = do
--   a'  <- a             -- Extract value 'a' from the first action
--   as' <- sequence as   -- Recursively sequence the rest of the list
--   return (a' : as')    -- Cons them together and wrap in 'return'
-- OR
sequence [] = return []
sequence (a:as) = a >>= (\a' -> sequence as >>= (\as' -> return (a' : as')))

-- (d) This one is the Kleisli "fish operator"
--

-- >>> safeParsePositive "10"
(>=>) :: (Monad m) => (a -> m b) -> (b -> m c) -> a -> m c
-- (>=>) f g a = do
--   b <- f a
--   g b

(>=>) f g a = f a >>= \b -> g b

-- 1. Parse a String into an Int
parse :: String -> Maybe Int
parse s = readMaybe s

-- 2. Ensure the Int is positive
ensurePositive :: Int -> Maybe Int
ensurePositive n
  | n > 0     = Just n
  | otherwise = Nothing

-- 3. Compose them using >=> operator
-- This creates a function: String -> Maybe Int
safeParsePositive :: String -> Maybe Int
safeParsePositive = parse >=> ensurePositive

-- (e)

join :: (Monad m) => m (m a) -> m a
join mma = mma >>= \ma -> ma
-- OR
-- join mma = mma >>= id

-- (f) Define the 'liftM' function

-- >>> liftM (+1) (Just 2)

liftM :: (Monad m) => (a -> b) -> m a -> m b
-- liftM f ma = ma >>= \a -> return (f a)
-- liftM f ma = (>>=) ma (\a -> return (f a))
liftM f ma = (>>=) ma (return . f)

-- Thought question: Is the type of `liftM` similar to that of another
-- function we've discussed recently?

-- (g) And its two-argument version ...
-- >>> liftM2 (+) (Just 10) (Just 5)

liftM2 :: (Monad m) => (a -> b -> r) -> m a -> m b -> m r
liftM2 f ma mb = ma >>=
  \a -> mb >>=
    \b -> return (f a b)

{-
-------------------------------------------------------------------------

General Applicative Functions
=============================

Which of these functions above can you equivalently rewrite using `Applicative`?
i.e. for which of the definitions below, can you replace `undefined` with
a definition that *only* uses members of the `Applicative` (and `Functor`)
type class. (Again, do not use functions from `Control.Applicative`,
`Data.Foldable` or `Data.Traversable` in your solution.)

  (<*>) :: f (a -> b) -> f a -> f b
  (<$>) :: (a -> b) -> f a -> f b

If you provide a definition, you should write test cases that demonstrate that it
has the same behavior on `List` and `Maybe` as the monadic versions above.
-}

-- NOTE: you will not be able to define all of these, but be sure to test the
-- ones that you do

-- >>> mapA Just [1,2]

-- >>> mapA (\x -> if x > 0 then Just x else Nothing) [1,2,-1]

mapA :: (Applicative f) => (a -> f b) -> [a] -> f [b]
mapA _ []     = pure []
-- mapA f (x:xs) = (:) <$> f x <*> mapA f xs
-- OR
mapA f (x:xs) = pure (:) <*> f x <*> mapA f xs
{-
Explanation:
  pure (:) lifts the list constructor (:) into the applicative.
  Then <*> f x feeds the effectful result of f x to (:).
  Then <*> mapA f xs feeds the effectful list from the recursive call.
  Together they form f (x':xs'), i.e. an effectful list.
-}

foldA :: forall f a b. (Applicative f) => (a -> b -> f a) -> a -> [b] -> f a
foldA = undefined

-- >>> sequenceA [Just 1, Just 2]

sequenceA :: (Applicative f) => [f a] -> f [a]
-- sequenceA = foldr (\ x -> (<*>) (pure (:) <*> x)) (pure [])
-- OR
-- sequenceA = foldr (\ x -> (<*>) ((:) <$> x)) (pure [])
-- OR
sequenceA []     = pure []
sequenceA (x:xs) = pure (:) <*> x <*> sequenceA xs
-- OR
-- sequenceA (x:xs) = ((:) <$> x) <*> sequenceA xs

kleisliA :: (Applicative f) => (a -> f b) -> (b -> f c) -> a -> f c
kleisliA = error "cannot be implemented using Applicative/Functor"

joinA :: (Applicative f) => f (f a) -> f a
joinA = error "cannot be implemented using Applicative/Functor"

-- >>> liftA (+1) (Just 3)

-- >>> liftA (*2) [1,2,3]

liftA :: (Applicative f) => (a -> b) -> f a -> f b
-- liftA f fa = f <$> fa
-- OR
liftA f fa = pure f <*> fa

-- >>>  liftA2 (*) [1,2] [10,20]

liftA2 :: (Applicative f) => (a -> b -> r) -> f a -> f b -> f r
-- liftA2 f fa fb = ((pure f) <*> fa) <*> fb
{-
This is how it works -
  pure f                 :: f (a -> b -> r)
  pure f <*> fa          :: f (b -> r)
  (pure f <*> fa) <*> fb :: f r
-}
liftA2 f fa fb = f <$> fa <*> fb
{-
This is how it works -
  f <$> fa          :: f (b -> r)
  (f <$> fa) <*> fb :: f r
-}