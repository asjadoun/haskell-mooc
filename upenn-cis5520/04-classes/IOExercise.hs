{-
---
fulltitle: "In class exercise: IOExercise"
---

In-class exercise (IO Monad)
-}

module IOExercise where

import System.FilePath

{-
Rewrite these programs so that they do not use 'do'.
(Make sure that you do not change their behavior!)
-}

simpleProgram :: IO ()
simpleProgram =
  putStrLn "This is a simple program that does IO." >>
    putStrLn "What is your name?" >>
    getLine >>=
      \inpStr -> putStrLn ("Welcome to Haskell, " ++ inpStr ++ "!")

-- OR With "do" notation

-- simpleProgram = do
--   putStrLn "This is a simple program that does IO."
--   putStrLn "What is your name?"
--   inpStr <- getLine
--   putStrLn ("Welcome to Haskell, " ++ inpStr ++ "!")

lengthProgram :: IO Int
lengthProgram =
  return (length [1, 2, 3, 4, 5, 6]) >>=
    \x -> putStrLn ("The length of the list is" ++ show x) >>
     return x

-- OR With "do" notation

-- lengthProgram = do
--   let x = length [1, 2, 3, 4, 5, 6]
--   putStrLn ("The length of the list is" ++ show x)
--   return x

{-
>
-}

anotherProgram :: IO ()
anotherProgram =
  putStrLn "What is your name?" >>
  getLine >>=
    \inpStr -> 
    (
        if inpStr == "Haskell"
        then do
          putStrLn "You rock!" >> 
            putStrLn "Really!!"
        else putStrLn ("Hello " ++ inpStr)
        ) >>
    putStrLn "That's all!"

-- OR With "do" notation

-- anotherProgram = do
--   putStrLn "What is your name?"
--   inpStr <- getLine
--   if inpStr == "Haskell"
--     then do
--       putStrLn "You rock!"
--       return ()
--       putStrLn "Really!!"
--     else putStrLn ("Hello " ++ inpStr)
--   putStrLn "That's all!"
