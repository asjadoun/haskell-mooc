# Scala (Cats / Cats-Effect) vs Haskell Cheat-Sheet

A side-by-side comparison of **Scala (with Cats / Cats-Effect)** and **Haskell**:  
types, typeclasses, methods, monad transformers, effects, error handling, concurrency, and streaming.  

---

## Table of Contents

1. [Core Types](#1-core-types)  
2. [Collections](#2-collections)  
3. [Typeclasses & Methods](#3-typeclasses--methods)  
   - [Semigroup](#semigroup)  
   - [Monoid](#monoid)  
   - [Functor](#functor)  
   - [Applicative](#applicative)  
   - [Monad](#monad)  
   - [Bifunctor](#bifunctor)  
   - [Arrow](#arrow)  
4. [Effects (Cats-Effect vs Haskell IO)](#4-effects-cats-effect-vs-haskell-io)  
5. [Monad Transformers](#5-monad-transformers)  
6. [Advanced Patterns](#6-advanced-patterns)  
7. [Error Handling](#7-error-handling)  
8. [Concurrency & Parallelism](#8-concurrency--parallelism)  
9. [Streaming](#9-streaming)  

---

## 1. Core Types

| Concept             | Scala (Cats / Cats-Effect)          | Haskell                  |
|---------------------|-------------------------------------|--------------------------|
| Int                 | `Int`                               | `Int`                    |
| Double              | `Double`                            | `Double`                 |
| String              | `String`                            | `String`                 |
| Unit                | `Unit`                              | `()`                     |
| Function            | `(A) => B`                          | `A -> B`                 |
| Option / Maybe      | `Option[A]`                         | `Maybe a`                |
| Either              | `Either[E, A]`                      | `Either e a`             |
| Try / Exception     | `scala.util.Try[A]`                 | `IO (Either SomeException a)` |
| Tuple               | `(A, B)`                            | `(a, b)`                 |

---

## 2. Collections

| Concept       | Scala                          | Haskell                |
|---------------|-------------------------------|------------------------|
| List          | `List[A]`                      | `[a]`                  |
| Vector        | `Vector[A]`                    | `Vector a` (Data.Vector) |
| Map           | `Map[K, V]`                    | `Map k v` (Data.Map)   |
| Set           | `Set[A]`                       | `Set a` (Data.Set)     |
| Stream        | `LazyList[A]`                  | `Stream a` (deprecated, prefer Conduit/Pipes) |

---

## 3. Typeclasses & Methods

### Semigroup

| Method | Scala (Cats) | Haskell |
|--------|--------------|---------|
| combine | `Semigroup[A].combine(x, y)` | `(<>) :: a -> a -> a` |

```scala
import cats.Semigroup
import cats.syntax.semigroup._

val res = "Hello " |+| "World" // "Hello World"

| Method  | Scala (Cats)      | Haskell  |   |        |
| ------- | ----------------- | -------- | - | ------ |
| empty   | `Monoid[A].empty` | `mempty` |   |        |
| combine | `                 | +        | ` | `(<>)` |

import cats.Monoid

val m = Monoid[List[Int]]
val res = m.empty |+| List(1,2) // List(1,2)

res = mempty <> [1,2]
