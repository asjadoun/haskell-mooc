**custom Parser type**
## 1. **Type Definition**
```haskell
newtype Parser a = P { doParse :: String -> Maybe (a, String) }
```
* We are defining a newtype named Parser a.
* The newtype keyword in Haskell is used to define a new type that is isomorphic to an existing one, but at compile time it is treated as a distinct type.

* It has the runtime representation of the type it wraps, but it is a separate type at compile time, which helps in preventing mixing up different types that have the same underlying representation.

* In this case, we are defining a Parser a as a type that is isomorphic to a function from a String to a Maybe (a, String).

* The Parser a type is defined with a single constructor named P. However, note that in the definition we have:

newtype Parser a = P { doParse :: String -> Maybe (a, String) }

* This uses record syntax. So P is the constructor, and doParse is a field accessor function that, given a Parser a, returns the underlying function of type String -> Maybe (a, String).

Let's break down the type:

String -> Maybe (a, String)

* This is a function that takes a string (the input to be parsed) and returns a Maybe (a, String). The Maybe type represents either success or failure:

* If the parsing is successful, it returns Just (a, remainingString), where a is the parsed value of type a and remainingString is the part of the input string that was not consumed.

If the parsing fails, it returns Nothing.

The newtype definition is equivalent to:

newtype Parser a = P (String -> Maybe (a, String))

but with the added benefit that we get the function doParse for free, which acts as a getter to extract the parsing function from the Parser type.

* Example of using this:

Suppose we have a parser for integers. Then Parser Int would be a parser that, if successful, returns an Int and the remaining string.

We can define:

text
 parseInt :: Parser Int
 parseInt = P $ \s -> ...  -- implementation here
Then, to run the parser, we can use:

text
 doParse parseInt "123abc"
which would return something like:

text
 Just (123, "abc")
if the parser successfully parsed the integer 123 and left "abc" as the remaining string.

* Why use a newtype? Because we want to define instances for this type, like Functor, Applicative, Monad, etc., to build complex parsers from simple ones. By using a newtype, we can define these instances without causing an infinite loop (which might happen if we used a type alias) and also get the benefits of type safety.

Also note that because it's a newtype, the runtime overhead is zero. The P constructor and the doParse function are compiled away, and at runtime it's just the function.

Let's see how the Functor instance might look like:

instance Functor Parser where
fmap f (P p) = P $ \s -> case p s of
Just (a, s') -> Just (f a, s')
Nothing -> Nothing

This allows us to apply a function to the result of a parser, if the parser succeeds.

Similarly, we can define Applicative and Monad instances to chain parsers together.

Now, let's break it down:

## 1. **Type Definition**
```haskell
newtype Parser a = P { doParse :: String -> Maybe (a, String) }
```

- `Parser a` is a type that represents a parser for values of type `a`
- The `a` is a **type parameter** - it could be `Parser Int`, `Parser String`, `Parser Bool`, etc.

## 2. **Constructor & Field**
```haskell
P { doParse :: String -> Maybe (a, String) }
```
- `P` is the constructor
- `doParse` is a **field/accessor function** that extracts the underlying parsing function
- This uses **record syntax** to automatically create the `doParse` function

## 3. **The Core Idea**
A `Parser a` is essentially a **function** that:
- Takes a `String` (input to parse)
- Returns `Maybe (a, String)`:
  - `Just (parsedValue, remainingString)` if successful
  - `Nothing` if parsing fails

## 4. **Why `newtype` instead of `type` or `data`?**
- `newtype` has zero runtime overhead (compile-time wrapper only)
- Used when wrapping exactly one type (here: `String -> Maybe (a, String)`)
- Allows defining custom typeclass instances (like `Functor`, `Applicative`, `Monad`)

## 5. **Example Usage**
```haskell
-- Create a parser that consumes one character
charParser :: Parser Char
charParser = P $ \input -> case input of
    (c:rest) -> Just (c, rest)
    ""       -> Nothing

-- Use the parser
parseResult = doParse charParser "hello"
-- parseResult = Just ('h', "ello")
```

## 6. **Common Parser Combinator Pattern**
This is the classic pattern for building parser combinators:
```haskell
-- Map a function over parser result
instance Functor Parser where
    fmap f (P p) = P $ \input -> case p input of
        Just (val, rest) -> Just (f val, rest)
        Nothing -> Nothing

-- Chain parsers together
instance Monad Parser where
    return x = P $ \input -> Just (x, input)
    P p >>= f = P $ \input -> case p input of
        Just (val, rest) -> let P p' = f val in p' rest
        Nothing -> Nothing
```

## 7. **Practical Example**
```haskell
-- Parse two characters and return them as a pair
twoChars :: Parser (Char, Char)
twoChars = do
    c1 <- charParser
    c2 <- charParser
    return (c1, c2)

-- Usage
result = doParse twoChars "abc"
-- result = Just (('a','b'), "c")
```

This pattern is foundational for many parsing libraries in Haskell (like Parsec, Attoparsec, or Megaparsec). It allows building complex parsers from simple ones through composition.