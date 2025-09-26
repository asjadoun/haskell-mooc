# Functional Programming: Scala (Cats Ecosystem) vs Haskell - Complete Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Core Language Concepts](#core-language-concepts)
3. [Type System Comparison](#type-system-comparison)
4. [Typeclass Mapping](#typeclass-mapping)
5. [Monadic Programming](#monadic-programming)
6. [Error Handling](#error-handling)
7. [Concurrency & Parallelism](#concurrency--parallelism)
8. [Stream Processing](#stream-processing)
9. [Effects System](#effects-system)
10. [Advanced Patterns](#advanced-patterns)
11. [Real-World Integration](#real-world-integration)
12. [Performance & Ecosystem](#performance--ecosystem)
13. [Key Differences Summary](#key-differences-summary)

---

## Introduction

This guide provides a comprehensive comparison between **Scala (with Cats/Cats-Effect/FS2)** and **Haskell** for functional programming. Both languages offer powerful abstractions for building reliable, scalable systems, but with different approaches and trade-offs.

---

## Core Language Concepts

### Basic Data Types

| Concept | Scala (Cats) | Haskell |
|---------|--------------|---------|
| Integer | `Int` | `Int` |
| Floating Point | `Double` | `Double` |
| Boolean | `Boolean` | `Bool` |
| String | `String` | `String` |
| Unit/Void | `Unit` | `()` |
| Function | `A => B` | `A -> B` |
| Optional Value | `Option[A]` | `Maybe a` |
| Either | `Either[E, A]` | `Either e a` |
| Tuple | `(A, B)` | `(a, b)` |

### Collections

| Type | Scala | Haskell |
|------|-------|---------|
| List | `List[A]` | `[a]` |
| Vector | `Vector[A]` | `Vector a` |
| Map | `Map[K, V]` | `Map k v` |
| Set | `Set[A]` | `Set a` |
| Non-Empty List | `NonEmptyList[A]` | `NonEmpty a` |

### Evaluation Strategy

- **Scala**: Strict by default, lazy with `IO` and explicit lazy constructs
- **Haskell**: Lazy by default, strict when explicitly requested

---

## Type System Comparison

### Type Definitions

**Scala**
```scala
// Product types (case classes)
case class Person(name: String, age: Int)

// Sum types (sealed traits)
sealed trait Shape
case class Circle(radius: Double) extends Shape
case class Rectangle(width: Double, height: Double) extends Shape

// Type aliases
type UserId = Int
type Email = String
```

**Haskell**
```haskell
-- Product types
data Person = Person { name :: String, age :: Int }

-- Sum types (Algebraic Data Types)
data Shape = Circle Double | Rectangle Double Double

-- Type synonyms
type UserId = Int
type Email = String

-- Newtype (zero-cost abstraction)
newtype UserId = UserId Int deriving (Show, Eq)
```

---

## Typeclass Mapping

### Core Typeclasses

| Typeclass | Scala (Cats) | Haskell |
|-----------|--------------|---------|
| **Semigroup** | `x |+| y` or `Semigroup[A].combine(x, y)` | `x <> y` |
| **Monoid** | `Monoid[A].empty`, `x |+| y` | `mempty`, `x <> y` |
| **Functor** | `fa.map(f)` | `fmap f fa` or `f <$> fa` |
| **Applicative** | `fa.map2(fb)(f)` | `f <$> fa <*> fb` |
| **Monad** | `fa.flatMap(f)` | `fa >>= f` |
| **MonadError** | `fa.handleErrorWith(h)` | `catchError` |

### Detailed Examples

**Semigroup**
```scala
// Scala
import cats.Semigroup
import cats.syntax.semigroup._
val result = "Hello " |+| "World" // "Hello World"
```

```haskell
-- Haskell
result = "Hello " <> "World"
```

**Monoid**
```scala
// Scala
import cats.Monoid
val m = Monoid[List[Int]]
val result = m.empty |+| List(1,2) // List(1,2)
```

```haskell
-- Haskell
result = mempty <> [1,2]
```

---

## Monadic Programming

### Basic Monadic Operations

**Scala**
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

### Monad Transformers

**Scala**
```scala
import cats.data.OptionT

val optionTExample: OptionT[List, Int] = 
  OptionT(List(Some(1), None, Some(2)))

// Sequence operations
val listSequence: Option[List[Int]] = List(Some(1), Some(2)).sequence
```

**Haskell**
```haskell
import Control.Monad.Trans.Maybe (MaybeT(..))

optionTExample :: MaybeT [] Int
optionTExample = MaybeT [Just 1, Nothing, Just 2]

-- Sequence operations
listSequence :: Maybe [Int]
listSequence = sequence [Just 1, Just 2]
```

---

## Error Handling

### Either-based Error Handling

**Scala**
```scala
def divide(a: Int, b: Int): Either[String, Int] =
  if (b == 0) Left("Division by zero") else Right(a / b)

// With IO
def safeDivide(a: Int, b: Int): IO[Int] =
  if (b == 0) IO.raiseError(new ArithmeticException("Division by zero"))
  else IO.pure(a / b)

// EitherT for monadic error handling
import cats.data.EitherT

def divideEitherT(a: Int, b: Int): EitherT[IO, String, Int] =
  EitherT(if (b == 0) IO.pure(Left("Division by zero")) 
          else IO.pure(Right(a / b)))
```

**Haskell**
```haskell
divide :: Int -> Int -> Either String Int
divide a b = if b == 0 then Left "Division by zero" else Right (a `div` b)

-- With IO
safeDivide :: Int -> Int -> IO Int
safeDivide a b = if b == 0 then throwIO (userError "Division by zero") 
                 else return (a `div` b)

-- ExceptT for monadic error handling
import Control.Monad.Except

divideExceptT :: Monad m => Int -> Int -> ExceptT String m Int
divideExceptT a b = if b == 0 then throwError "Division by zero" 
                    else return (a `div` b)
```

---

## Concurrency & Parallelism

### Fiber-based Concurrency

**Scala (Cats-Effect)**
```scala
import cats.effect._
import cats.effect.implicits._
import scala.concurrent.duration._

// Fiber-based concurrency
val concurrentProgram: IO[String] = for {
  fiber <- IO.println("Starting task") >> 
           IO.sleep(1.second) >> 
           IO.pure("Result").start
  _ <- IO.println("Doing other work")
  result <- fiber.join
} yield result

// Race conditions
val raceExample: IO[Either[String, Int]] = 
  IO.sleep(500.millis).as("Timeout").race(IO.sleep(1.second).as(42))

// Shared state with Ref
def counterProgram: IO[Unit] = 
  Ref[IO].of(0).flatMap { ref =>
    ref.update(_ + 1) >> ref.get.flatMap(n => IO.println(s"Count: $n"))
  }
```

**Haskell**
```haskell
import Control.Concurrent.Async
import Control.Concurrent

-- Async operations
concurrentProgram :: IO String
concurrentProgram = do
  putStrLn "Starting task"
  fiber <- async $ threadDelay 1000000 >> return "Result"
  putStrLn "Doing other work"
  wait fiber

-- Race conditions
raceExample :: IO (Either String Int)
raceExample = race (threadDelay 500000 >> return "Timeout")
                   (threadDelay 1000000 >> return 42)

-- Shared state with MVar
counterProgram :: IO ()
counterProgram = do
  mvar <- newMVar (0 :: Int)
  modifyMVar_ mvar (\n -> return (n + 1))
  n <- readMVar mvar
  putStrLn $ "Count: " ++ show n
```

---

## Stream Processing

### Functional Streams

**Scala (FS2)**
```scala
import fs2._
import cats.effect._
import scala.concurrent.duration._

val streamExample: Stream[IO, Int] = 
  Stream(1, 2, 3) ++ Stream.eval(IO.pure(4))

val processingStream: Stream[IO, Int] = 
  Stream.range(1, 10)
    .filter(_ % 2 == 0)
    .map(_ * 2)
    .evalMap(n => IO.println(s"Processing: $n").as(n))

val concurrentStream: Stream[IO, Int] = 
  Stream(1, 2, 3).mapAsync(2) { n =>
    IO.sleep(1.second).as(n * 2)
  }
```

**Haskell (Conduit)**
```haskell
import Conduit
import Control.Monad.IO.Class

-- Basic stream
streamExample :: Monad m => ConduitT i Int m ()
streamExample = yieldMany [1, 2, 3] >> yield 4

-- Stream processing
processingStream :: MonadIO m => ConduitT i Int m ()
processingStream = yieldMany [1..9]
    .| filterC even
    .| mapC (*2)
    .| mapMC (\n -> liftIO (putStrLn ("Processing: " ++ show n)) >> return n)
```

---

## Effects System

### IO and Effect Management

**Scala (Cats-Effect)**
```scala
import cats.effect._

// Effects are captured in the IO type
val effectfulProgram: IO[Unit] = for {
  _ <- IO.println("Hello")
  _ <- IO.sleep(1.second)
  _ <- IO.println("World")
} yield ()

// Resource management
def resourceExample: Resource[IO, String] =
  Resource.make(IO.println("Acquire") *> IO.pure("resource"))(
    _ => IO.println("Release")
  )
```

**Haskell**
```haskell
-- Effects are captured in the IO type
effectfulProgram :: IO ()
effectfulProgram = do
  putStrLn "Hello"
  threadDelay 1000000
  putStrLn "World"

-- Resource management with bracket
import Control.Exception (bracket)

resourceExample :: IO String
resourceExample = bracket
  (putStrLn "Acquire" >> return "resource")
  (\_ -> putStrLn "Release")
  (\r -> return r)
```

---

## Advanced Patterns

### Arrows and Function Composition

**Scala**
```scala
import cats.arrow._
import cats.implicits._

val f: Int => Int = _ + 1
val g: Int => Int = _ * 2
val composed: Int => Int = f >>> g
```

**Haskell**
```haskell
import Control.Arrow

f :: Int -> Int
f = (+1)

g :: Int -> Int
g = (*2)

composed :: Int -> Int
composed = f >>> g
```

---

## Real-World Integration

### HTTP Services

**Scala (http4s)**
```scala
import org.http4s._
import org.http4s.dsl.io._
import org.http4s.implicits._

val helloService = HttpRoutes.of[IO] {
  case GET -> Root / "hello" / name =>
    Ok(s"Hello, $name")
}
```

**Haskell (Servant)**
```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}

import Servant

type API = "hello" :> Capture "name" String :> Get '[PlainText] String

server :: Server API
server name = return $ "Hello, " ++ name
```

### Kafka Integration

**Scala (FS2 Kafka)**
```scala
import fs2.kafka._
import cats.effect._
import cats.syntax.all._

case class UserEvent(userId: String, action: String, timestamp: Long)

val producerSettings: ProducerSettings[IO, String, String] =
  ProducerSettings[IO, String, String]
    .withBootstrapServers("localhost:9092")
    .withProperties(
      "acks" -> "all",
      "retries" -> "3"
    )

def createUserEventProducer: Resource[IO, KafkaProducer[IO, String, String]] =
  KafkaProducer.resource(producerSettings)

def produceUserEvents(events: List[UserEvent]): IO[Unit] = {
  val records = events.map { event =>
    val json = s"""{"userId":"${event.userId}","action":"${event.action}","timestamp":${event.timestamp}}"""
    ProducerRecord("user-events", event.userId, json)
  }

  createUserEventProducer.use { producer =>
    records.traverse_(record => 
      producer.produce(ProducerRecords.one(record)).flatten
        .flatMap(metadata => IO.println(s"Produced to ${metadata.topic}-${metadata.partition}"))
    )
  }
}

// Consumer example
val consumerSettings: ConsumerSettings[IO, String, String] =
  ConsumerSettings[IO, String, String]
    .withAutoOffsetReset(AutoOffsetReset.Earliest)
    .withBootstrapServers("localhost:9092")
    .withGroupId("user-events-group")

def consumerStream: Stream[IO, Unit] = {
  KafkaConsumer.stream(consumerSettings)
    .subscribeTo("user-events")
    .records
    .evalMap { committable =>
      IO.println(s"Processing: ${committable.record.value}") >>
      committable.offset.commit
    }
}
```

**Haskell (kafka-client)**
```haskell
{-# LANGUAGE OverloadedStrings #-}

import Kafka.Producer
import Kafka.Consumer
import Control.Monad (forever, void)

data UserEvent = UserEvent
  { userId :: String
  , action :: String
  , timestamp :: Int
  } deriving (Show)

-- Producer
producerConfig :: ProducerProperties
producerConfig = brokersList ["localhost:9092"]

produceUserEvents :: [UserEvent] -> IO ()
produceUserEvents events = do
  eitherProducer <- newProducer producerConfig
  case eitherProducer of
    Left err -> putStrLn $ "Producer error: " ++ show err
    Right producer -> do
      mapM_ (produceEvent producer) events
      void $ closeProducer producer

-- Consumer
consumerConfig :: ConsumerProperties
consumerConfig = brokersList ["localhost:9092"]
              <> groupId (ConsumerGroupId "user-events-group")
              <> autoOffsetReset Earliest

basicConsumer :: IO ()
basicConsumer = do
  eitherConsumer <- newConsumer consumerConfig (Subscription [TopicName "user-events"])
  case eitherConsumer of
    Left err -> putStrLn $ "Consumer error: " ++ show err
    Right consumer -> forever $ do
      eitherMessages <- pollMessage consumer (Timeout 1000)
      case eitherMessages of
        Left err -> putStrLn $ "Poll error: " ++ show err
        Right messages -> mapM_ processMessage messages
```

### Elasticsearch Integration

**Scala (elastic4s)**
```scala
import com.sksamuel.elastic4s.ElasticClient
import com.sksamuel.elastic4s.ElasticProperties
import com.sksamuel.elastic4s.circe._
import io.circe.generic.auto._

case class Product(id: String, name: String, price: Double, category: String)

val elasticProperties = ElasticProperties("http://localhost:9200")
val client: ElasticClient = ElasticClient(elasticProperties)

def indexProduct(product: Product): IO[IndexResponse] = {
  val request = 
    indexInto("products")
      .id(product.id)
      .doc(product)
      .refresh(RefreshPolicy.WaitFor)

  IO.fromFuture(IO(client.execute(request))).map(_.result)
}

def searchProducts(query: String): IO[SearchResponse] = {
  val searchRequest = search("products")
    .query(matchQuery("name", query))
    .limit(10)

  IO.fromFuture(IO(client.execute(searchRequest))).map(_.result)
}
```

**Haskell (bloodhound)**
```haskell
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Database.Bloodhound
import Data.Aeson (FromJSON, ToJSON)
import GHC.Generics (Generic)

data Product = Product
  { productId :: String
  , name :: String
  , price :: Double
  , category :: String
  } deriving (Generic, Show, Eq)

instance ToJSON Product
instance FromJSON Product

indexProduct :: Product -> BHIO (Reply IndexResponse)
indexProduct product = 
  indexDocument (IndexName "products") 
                (DocId (productId product)) 
                product 
                defaultIndexSettings

searchProducts :: String -> BHIO (Reply (SearchResult Product))
searchProducts query = 
  let search' = Search { queryBody = Just $ MatchQuery (MatchQueryField "name") (QueryString query) Nothing
                       , size = Size 10
                       }
  in searchByIndex (IndexName "products") search'
```

### JSON Processing

**Scala (Circe)**
```scala
import io.circe._
import io.circe.generic.auto._
import io.circe.parser._
import io.circe.syntax._

case class Person(name: String, age: Int, email: Option[String])
case class Address(street: String, city: String, zipCode: String)
case class User(id: Long, person: Person, addresses: List[Address])

// Automatic derivation
val user = User(1, Person("John", 30, Some("john@example.com")), 
               List(Address("123 Main St", "NYC", "10001")))

// Encoding to JSON
val json: Json = user.asJson
val jsonString: String = user.asJson.noSpaces

// Decoding from JSON
val jsonStr = """{
  "id": 1,
  "person": {"name": "John", "age": 30, "email": "john@example.com"},
  "addresses": [{"street": "123 Main St", "city": "NYC", "zipCode": "10001"}]
}"""

val decoded: Either[Error, User] = decode[User](jsonStr)

// Custom encoders/decoders
implicit val personEncoder: Encoder[Person] = Encoder.forProduct3("name", "age", "email")(p => 
  (p.name, p.age, p.email))

implicit val personDecoder: Decoder[Person] = Decoder.forProduct3("name", "age", "email")(Person.apply)

// Working with JSON in IO
def parseUserFromApi: IO[User] = 
  IO("""api response json""").flatMap { jsonStr =>
    IO.fromEither(decode[User](jsonStr))
  }
```

**Haskell (Aeson)**
```haskell
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Data.Aeson
import GHC.Generics (Generic)
import qualified Data.ByteString.Lazy as L

data Person = Person
  { name :: String
  , age :: Int
  , email :: Maybe String
  } deriving (Generic, Show)

data Address = Address
  { street :: String
  , city :: String
  , zipCode :: String
  } deriving (Generic, Show)

data User = User
  { userId :: Int
  , person :: Person
  , addresses :: [Address]
  } deriving (Generic, Show)

-- Automatic derivation
instance ToJSON Person
instance FromJSON Person
instance ToJSON Address
instance FromJSON Address
instance ToJSON User
instance FromJSON User

-- Usage
user :: User
user = User 1 (Person "John" 30 (Just "john@example.com")) 
           [Address "123 Main St" "NYC" "10001"]

-- Encoding to JSON
jsonByteString :: L.ByteString
jsonByteString = encode user

-- Decoding from JSON
jsonStr :: L.ByteString
jsonStr = "{\"userId\":1,\"person\":{\"name\":\"John\",\"age\":30,\"email\":\"john@example.com\"},\"addresses\":[{\"street\":\"123 Main St\",\"city\":\"NYC\",\"zipCode\":\"10001\"}]}"

decoded :: Maybe User
decoded = decode jsonStr

-- Custom instances
instance ToJSON Person where
  toJSON (Person n a e) = object ["name" .= n, "age" .= a, "email" .= e]

instance FromJSON Person where
  parseJSON = withObject "Person" $ \o -> Person
    <$> o .: "name"
    <*> o .: "age" 
    <*> o .:? "email"

-- Working with JSON in IO
parseUserFromApi :: IO (Either String User)
parseUserFromApi = do
  jsonResponse <- readFile "api_response.json"
  return $ eitherDecode (L.pack jsonResponse)
```

### Advanced JSON Processing

**Scala (Circe Advanced)**
```scala
import io.circe.optics.JsonPath._
import cats.syntax.either._

// JSON transformation
val transform: Json => Json = root.person.age.int.modify(_ + 1)

// Streaming JSON parsing
import fs2.Stream
import io.circe.fs2._

def processJsonStream: Stream[IO, User] = 
  Stream.resource(Files[IO].readAll(Path("users.json")))
    .through(stringArrayParser)
    .through(decoder[IO, User])

// Error handling with accumulating errors
import io.circe.Decoder.Result
import cats.data.Validated

def validateUser(json: Json): Validated[List[String], User] = {
  val cursor = json.hcursor
  (
    cursor.get[Long]("id").toValidatedNel,
    cursor.get[Person]("person").toValidatedNel,
    cursor.get[List[Address]]("addresses").toValidatedNel
  ).mapN(User.apply)
}
```

**Haskell (Aeson Advanced)**
```haskell
import Data.Aeson.Lens
import Control.Lens
import qualified Data.Aeson.Types as AT

-- JSON transformation using lens
transformAge :: Value -> Value
transformAge = person . key "age" . _Integer %~ (+1)

-- Streaming JSON parsing
import qualified Data.Conduit as C
import Data.Conduit.Aeson

processJsonStream :: C.ConduitT () User IO ()
processJsonStream = 
  C.sourceFile "users.json"
  C..| C.conduitParser json
  C..| C.awaitForever (\v -> case fromJSON v of
    Success user -> C.yield user
    Error _ -> return ())

-- Custom parsing with better error messages
parseUserWithErrors :: Value -> AT.Parser User
parseUserWithErrors = withObject "User" $ \o -> do
  uid <- o .: "id" <?> AT.Key "id"
  p <- o .: "person" <?> AT.Key "person" 
  addrs <- o .: "addresses" <?> AT.Key "addresses"
  return $ User uid p addrs

-- Alternative parsing
instance FromJSON User where
  parseJSON v = parseUserWithErrors v <|> parseUserLegacy v
    where
      parseUserLegacy = -- fallback parser for old format
```

### Database Integration (Oracle)

**Scala (Doobie)**
```scala
import doobie._
import doobie.implicits._
import cats.effect._

case class User(id: Long, name: String, email: String)

val xa: Transactor[IO] = Transactor.fromDriverManager[IO](
  "oracle.jdbc.OracleDriver",
  "jdbc:oracle:thin:@localhost:1521:xe",
  "username",
  "password"
)

def insertUser(user: User): IO[Int] =
  sql"INSERT INTO users (id, name, email) VALUES (${user.id}, ${user.name}, ${user.email})"
    .update
    .run
    .transact(xa)

def findUserById(id: Long): IO[Option[User]] =
  sql"SELECT id, name, email FROM users WHERE id = $id"
    .query[User]
    .option
    .transact(xa)

def findUsersByName(name: String): IO[List[User]] =
  sql"SELECT id, name, email FROM users WHERE name LIKE ${s"%$name%"}"
    .query[User]
    .to[List]
    .transact(xa)
```

**Haskell (postgresql-simple style)**
```haskell
{-# LANGUAGE OverloadedStrings #-}

import Database.Oracle.Simple
import Data.Int (Int64)

data User = User
  { userId :: Int64
  , userName :: String
  , userEmail :: String
  } deriving (Show, Eq)

instance FromRow User where
  fromRow = User <$> field <*> field <*> field

instance ToRow User where
  toRow (User uid name email) = [toField uid, toField name, toField email]

connectInfo :: ConnectInfo
connectInfo = defaultConnectInfo
  { connectHost = "localhost"
  , connectPort = 1521
  , connectUser = "username"
  , connectPassword = "password"
  , connectDatabase = "xe"
  }

insertUser :: Connection -> User -> IO Int64
insertUser conn user = do
  execute conn "INSERT INTO users (id, name, email) VALUES (?, ?, ?)" user

findUserById :: Connection -> Int64 -> IO (Maybe User)
findUserById conn uid = do
  results <- query conn "SELECT id, name, email FROM users WHERE id = ?" (Only uid)
  case results of
    [user] -> return (Just user)
    _ -> return Nothing

findUsersByName :: Connection -> String -> IO [User]
findUsersByName conn name = 
  query conn "SELECT id, name, email FROM users WHERE name LIKE ?" (Only ("%" ++ name ++ "%"))
```

### Complete Data Pipeline: Kafka + Elasticsearch

**Scala (FS2 + Kafka + Elasticsearch)**
```scala
import fs2.kafka._
import com.sksamuel.elastic4s.ElasticClient
import io.circe.parser._

case class PipelineConfig(
  kafkaBootstrapServers: String,
  kafkaTopic: String,
  kafkaGroupId: String,
  elasticsearchUrl: String
)

def createDataPipeline(config: PipelineConfig): Stream[IO, Unit] = {
  val consumerSettings = ConsumerSettings[IO, String, String]
    .withBootstrapServers(config.kafkaBootstrapServers)
    .withGroupId(config.kafkaGroupId)

  val esClient = Stream.resource(
    Resource.make(IO(ElasticClient(ElasticProperties(config.elasticsearchUrl))))(
      client => IO.fromFuture(IO(client.close())).void
    )
  )

  esClient.flatMap { client =>
    KafkaConsumer.stream(consumerSettings)
      .subscribeTo(config.kafkaTopic)
      .records
      .mapAsync(25) { committable =>
        parse(committable.record.value)
          .flatMap(_.as[Product])
          .fold(
            error => 
              IO.println(s"Failed to parse: $error") >>
              committable.offset.commit,
            product => 
              indexProduct(client, product)
                .attempt
                .flatMap {
                  case Right(_) =>
                    IO.println(s"Indexed product ${product.id}") >>
                    committable.offset.commit
                  case Left(error) =>
                    IO.println(s"Failed to index: $error") >>
                    committable.offset.commit
                }
          )
      }
  }
}
```

**Haskell (Complete Pipeline)**
```haskell
import Conduit
import qualified Data.Aeson as A

data PipelineConfig = PipelineConfig
  { kafkaBrokers :: [String]
  , kafkaTopic :: TopicName
  , kafkaGroupId :: ConsumerGroupId
  , esServer :: Server
  }

runDataPipeline :: PipelineConfig -> IO ()
runDataPipeline config = withElasticClient $ \esClient -> do
  let kafkaConfig = brokersList (kafkaBrokers config)
                   <> groupId (kafkaGroupId config)
  
  eitherConsumer <- newConsumer kafkaConfig (Subscription [kafkaTopic config])
  case eitherConsumer of
    Left err -> putStrLn $ "Kafka error: " ++ show err
    Right consumer -> 
      runConduit $ sourceKafka consumer 
                .| processMessages esClient
                .| sinkCommits consumer

processMessages :: BHIO -> ConduitT (ConsumerRecord (Maybe String) (Maybe String)) Void IO ()
processMessages esClient = awaitForever $ \record -> 
  case crValue record of
    Nothing -> liftIO $ putStrLn "Empty message"
    Just value -> case A.eitherDecode' value of
      Left err -> liftIO $ putStrLn $ "JSON error: " ++ err
      Right product -> do
        result <- liftIO $ runBH esClient $ indexProduct product
        case result of
          Right _ -> liftIO $ putStrLn "Indexed successfully"
          Left esErr -> liftIO $ putStrLn $ "ES error: " ++ show esErr
```

### Complete Enterprise Pipeline: Kafka + Database + Elasticsearch

**Scala (Full Stack)**
```scala
import fs2.kafka._
import doobie._
import doobie.implicits._
import com.sksamuel.elastic4s._
import io.circe.parser._

case class UserActivity(
  userId: Long,
  action: String,
  timestamp: Long,
  metadata: Map[String, String]
)

def enterprisePipeline: Stream[IO, Unit] = {
  val kafkaSettings = ConsumerSettings[IO, String, String]
    .withBootstrapServers("localhost:9092")
    .withGroupId("enterprise-pipeline")

  KafkaConsumer.stream(kafkaSettings)
    .subscribeTo("user-activities")
    .records
    .evalMap { committable =>
      for {
        activity <- IO.fromEither(decode[UserActivity](committable.record.value))
        _ <- storeInDatabase(activity)
        _ <- indexInElasticsearch(activity)
        _ <- committable.offset.commit
      } yield ()
    }
}

def storeInDatabase(activity: UserActivity): IO[Unit] =
  sql"INSERT INTO user_activities (user_id, action, timestamp) VALUES (${activity.userId}, ${activity.action}, ${activity.timestamp})"
    .update
    .run
    .transact(xa)
    .void

def indexInElasticsearch(activity: UserActivity): IO[Unit] =
  IO.fromFuture(IO(esClient.execute(
    indexInto("activities").doc(activity)
  ))).void
```

**Haskell (Full Stack)**
```haskell
import qualified Database.Oracle.Simple as DB
import qualified Database.Bloodhound as ES
import Kafka.Consumer

data UserActivity = UserActivity
  { activityUserId :: Int64
  , activityAction :: String
  , activityTimestamp :: Int64
  } deriving (Show, Generic)

instance FromJSON UserActivity
instance ToJSON UserActivity
instance FromRow UserActivity where
  fromRow = UserActivity <$> field <*> field <*> field

enterprisePipeline :: DB.Connection -> ES.BHEnv -> IO ()
enterprisePipeline dbConn esEnv = do
  consumer <- newConsumer consumerConfig (Subscription [TopicName "user-activities"])
  case consumer of
    Right c -> forever $ do
      messages <- pollMessage c (Timeout 1000)
      case messages of
        Right msgs -> mapM_ (processActivity dbConn esEnv) msgs
        Left err -> putStrLn $ "Poll error: " ++ show err
    Left err -> putStrLn $ "Consumer error: " ++ show err

processActivity :: DB.Connection -> ES.BHEnv -> ConsumerRecord -> IO ()
processActivity dbConn esEnv record = 
  case crValue record >>= A.decode of
    Just activity -> do
      storeInDatabase dbConn activity
      indexInElasticsearch esEnv activity
    Nothing -> putStrLn "Failed to decode activity"

storeInDatabase :: DB.Connection -> UserActivity -> IO ()
storeInDatabase conn activity = void $
  DB.execute conn 
    "INSERT INTO user_activities (user_id, action, timestamp) VALUES (?, ?, ?)"
    activity

indexInElasticsearch :: ES.BHEnv -> UserActivity -> IO ()
indexInElasticsearch env activity = void $
  ES.runBH env $ ES.indexDocument (ES.IndexName "activities") 
                                  (ES.DocId $ show $ activityUserId activity)
                                  activity
                                  ES.defaultIndexSettings
```

---

## Performance & Ecosystem

### Performance Characteristics

| Aspect | Scala (Cats) | Haskell |
|--------|--------------|---------|
| **Runtime** | JVM - mature, optimized | GHC - advanced optimizations |
| **Memory** | JVM garbage collection | Lazy evaluation, efficient GC |
| **Concurrency** | JVM threads + Cats-Effect fibers | Lightweight green threads |
| **Startup** | JVM warmup time | Fast startup |

### Ecosystem Maturity

**Scala Strengths:**
- Seamless Java ecosystem integration
- Strong enterprise adoption
- Excellent big data tools (Spark, Kafka)
- Rich IDE support
- More job opportunities

**Haskell Strengths:**
- Pure functional paradigm
- Advanced type system
- Strong in compiler technology
- Excellent for correctness-critical applications
- Growing web development ecosystem

---

## Key Differences Summary

### Language Philosophy

| Feature | Scala (Cats) | Haskell |
|---------|--------------|---------|
| **Paradigm** | Multi-paradigm (OOP + FP) | Pure functional |
| **Purity** | Convention-based with Cats-Effect | Language-enforced |
| **Evaluation** | Strict by default | Lazy by default |
| **Type System** | Sophisticated with implicits | Advanced with type classes |
| **Learning Curve** | Gradual FP adoption | Steep but rewarding |

### Practical Considerations

**Choose Scala when:**
- Need Java ecosystem integration
- Team has OOP background
- Building enterprise applications
- Working with big data

**Choose Haskell when:**
- Want pure functional programming
- Need maximum correctness
- Building compilers/DSLs
- Learning FP concepts deeply

### Syntax Comparison Summary

| Operation | Scala | Haskell |
|-----------|-------|---------|
| **Function application** | `f(x)` | `f x` |
| **Method chaining** | `x.map(f).filter(p)` | `filter p (map f x)` |
| **Composition** | `f >>> g` | `f . g` |
| **Monadic bind** | `fa.flatMap(f)` | `fa >>= f` |
| **For comprehension** | `for { x <- fa } yield f(x)` | `do { x <- fa; return (f x) }` |

---

## Conclusion

Both Scala (with Cats ecosystem) and Haskell offer powerful functional programming capabilities. Scala provides a pragmatic approach with excellent Java interoperability, while Haskell offers a pure functional experience with advanced type system features. The choice depends on your team's background, project requirements, and ecosystem needs.

This guide serves as a comprehensive reference for understanding the similarities and differences between these two excellent functional programming languages.