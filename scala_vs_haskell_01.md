# Scala (Cats / Cats-Effect / FS2) vs Haskell Cheat-Sheet

While no single cheat sheet exists for directly translating between Scala (Cats/Cats-Effect/FS2) and Haskell, the concepts, typeclasses, and patterns map closely. The primary differences lie in syntax and ecosystem specifics.

## Core Concepts

| Concept | Scala (Cats/Cats-Effect/FS2) | Haskell |
|---------|------------------------------|----------|
| Monadic composition | for comprehension | do notation |
| Effects | Effects are captured in the IO type from Cats-Effect. The IO type describes a program that will eventually perform a side effect when run. | Effects are captured in the IO type. main must have type IO (). |
| Typeclasses | A typeclass is an interface that can be implemented by many types. It's often used with implicit parameters or the ContextBound syntax (F: Monad). | A typeclass is a set of functions that are defined for a type, and that can be used with any type that implements the typeclass. |
| Asynchronous operations | The Concurrent and Async typeclasses in Cats-Effect handle concurrency and non-blocking operations. Fibers manage lightweight green threads. | The Control.Concurrent library and async package offer tools for concurrency. Lightweight green threads are managed by the GHC runtime. |
| Streaming | FS2 (Functional Streams for Scala) is the standard streaming library. Key types are Stream[F, A]. | The pipes or conduit libraries provide functional streaming primitives. |

## Typeclass Mapping

This table shows some common typeclasses and their equivalent syntax in each language.

| Typeclass | Scala (Cats syntax) | Haskell |
|-----------|---------------------|----------|
| Functor (map) | `fa.map(f)` | `fmap f fa` or `f <$> fa` |
| Applicative (apply) | `fa.map2(fb)(f)` or `(f.pure[F] <*> fa) <*> fb` | `pure x`, `<*>` |
| Monad (flatMap) | `fa.flatMap(f)` | `fa >>= f` or do notation |
| MonadError (error handling) | `fa.handleErrorWith(h)` or `fa.attempt` | `catchError`, `try` |
| Monoid (combine) | `fa.combine(fb)` or `fa |+| fb` | `<>` |
| Semigroup (combine) | `fa.combine(fb)` or `fa |+| fb` | `<>` |

## Code Snippet Comparison

### Monadic Composition

**Scala (Cats)**
```scala
import cats.effect._
import cats.implicits._

def foo(): IO[Int] = IO.pure(1)
def bar(x: Int): IO[String] = IO.pure(x.toString)

val program: IO[String] = for {
  x <- foo()
  y <- bar(x)
} yield y
```

**Haskell**
```haskell
import Control.Monad.IO.Class

foo :: IO Int
foo = return 1

bar :: Int -> IO String
bar x = return (show x)

program :: IO String
program = do
  x <- foo
  y <- bar x
  return y
```

### Error Handling

**Scala (Cats-Effect)**
```scala
import cats.effect._
import cats.implicits._

val unsafeProgram: IO[Int] = IO(throw new RuntimeException("fail"))

val handledProgram: IO[Either[Throwable, Int]] = unsafeProgram.attempt
```

**Haskell**
```haskell
import Control.Exception

unsafeProgram :: IO Int
unsafeProgram = throwIO (ErrorCall "fail")

handledProgram :: IO (Either SomeException Int)
handledProgram = try unsafeProgram
```

### Parallel Execution

**Scala (Cats-Effect)**
```scala
import cats.effect._
import cats.implicits._

val task1 = IO.sleep(1.second) *> IO.println("task1 done")
val task2 = IO.sleep(1.second) *> IO.println("task2 done")

val parallelProgram: IO[Unit] = (task1, task2).parTupled.void
```

**Haskell**
```haskell
import Control.Concurrent.Async
import Control.Monad

task1 :: IO ()
task1 = threadDelay 1000000 >> putStrLn "task1 done"

task2 :: IO ()
task2 = threadDelay 1000000 >> putStrLn "task2 done"

main :: IO ()
main = void (concurrently task1 task2)
```

### Streaming

**Scala (FS2)**
```scala
import cats.effect._
import fs2.Stream

val s: Stream[IO, Int] = Stream(1, 2, 3)

val processedStream: Stream[IO, String] = s.map(_.toString)
```

**Haskell (pipes)**
```haskell
import Pipes

s :: Producer Int IO ()
s = each [1, 2, 3]

processedStream :: Producer String IO ()
processedStream = s >-> Pipes.map show
```

## Key Differences

- **Syntax and language foundation**: Scala is an object-oriented language with functional features, while Haskell is a purely functional language. The Scala ecosystem with Cats, Cats-Effect, and FS2 is designed to bring Haskell-like functional purity to a more traditional OO language.

- **Laziness**: Haskell is a lazy language by default, meaning expressions are not evaluated until their results are needed. Scala with Cats/Cats-Effect is strict by default, but the IO monad introduces an explicit form of laziness, ensuring effects are deferred.

- **Ecosystem and interoperability**: Scala benefits from seamless interoperability with the vast Java ecosystem, allowing it to leverage mature, battle-tested libraries. Haskell's ecosystem is more self-contained and purely functional.

- **Purity**: Haskell enforces purity at the language level, with the type system separating pure and impure code. In Scala, purity is a convention upheld by libraries like Cats-Effect, but it is not enforced by the compiler.

- **Learning curve**: Scala is often considered more accessible to programmers coming from an OOP background, as it blends familiar syntax with advanced functional concepts. Haskell's purely functional nature and unique syntax can be a steeper initial learning curve for some. 