interpreter :: [String] -> [String]
interpreter commands = go (0, 0) commands
    where
        go _   []         = []
        go (x, y) (cmd:cmds) = case cmd of
            "up" -> go (x, y + 1) cmds
            "down" -> go(x, y - 1) cmds
            "right" -> go (x + 1, y) cmds
            "left" -> go (x - 1, y) cmds
            "printX" -> show x : go (x, y) cmds
            "printY" -> show y : go (x, y) cmds
            _ -> go (x, y) cmds

exec :: String -> Int -> Int
exec cmd acc = case cmd of
    "up" ->  acc + 1
    "down" -> acc - 1
    _ -> acc

main :: IO() 
main = do 
    print "Welcome..."
    print (interpreter ["up", "up", "right", "right", "left", "printX", "printY"])
