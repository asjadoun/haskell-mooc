Functional Programming Ecosystem: Scala (Cats) vs Haskell - Comprehensive Guide

markdown
# Functional Programming Ecosystem: Scala (Cats) vs Haskell - Comprehensive Guide

## Table of Contents
- [Basic Data Types](#basic-data-types)
- [Type Definitions](#type-definitions)
- [Functor, Applicative, Monad](#functor-applicative-monad)
- [Monoids](#monoids)
- [Arrows](#arrows)
- [Error Handling](#error-handling)
- [Concurrency](#concurrency)
- [Stream Processing](#stream-processing)
- [HTTP Libraries](#http-libraries)
- [Kafka Integration](#kafka-integration)
- [Elasticsearch Integration](#elasticsearch-integration)
- [Key Differences Summary](#key-differences-summary)

## Basic Data Types

### Scala
```scala
// Primitive types
val int: Int = 42
val double: Double = 3.14
val boolean: Boolean = true
val string: String = "Hello"

// Collection types
val list: List[Int] = List(1, 2, 3)
val option: Option[Int] = Some(42)
val either: Either[String, Int] = Right(42)
val map: Map[String, Int] = Map("a" -> 1, "b" -> 2)

// Cats data types
import cats.data._
val validated: Validated[String, Int] = Validated.valid(42)
val nonEmptyList: NonEmptyList[Int] = NonEmptyList(1, List(2, 3))
val chain: Chain[Int] = Chain(1, 2, 3)
Haskell
haskell
-- Primitive types
int :: Int
int = 42

double :: Double
double = 3.14

boolean :: Bool
boolean = True

string :: String
string = "Hello"

-- Built-in types
list :: [Int]
list = [1, 2, 3]

maybe :: Maybe Int
maybe = Just 42

either :: Either String Int
either = Right 42

map' :: Map String Int
map' = fromList [("a", 1), ("b", 2)]

-- Common data types
import Data.List.NonEmpty (NonEmpty(..))
nonEmpty :: NonEmpty Int
nonEmpty = 1 :| [2, 3]
Type Definitions
Scala
scala
// Case classes (product types)
case class Person(name: String, age: Int)

// Sealed traits (sum types)
sealed trait Shape
case class Circle(radius: Double) extends Shape
case class Rectangle(width: Double, height: Double) extends Shape

// Type aliases
type UserId = Int
type Email = String

// Newtypes using tagged types or libraries
import cats.data._
type UserId = Int @@ UserIdTag
Haskell
haskell
-- Product types
data Person = Person { name :: String, age :: Int }

-- Sum types
data Shape = Circle Double | Rectangle Double Double

-- Type synonyms
type UserId = Int
type Email = String

-- Newtype (zero-cost abstraction)
newtype UserId = UserId Int deriving (Show, Eq)

-- Record syntax
data Person = Person
  { name :: String
  , age :: Int
  } deriving (Show, Eq)
Functor, Applicative, Monad
Scala
scala
import cats._
import cats.implicits._

// Functor
val functorExample: Option[Int] = Some(5).map(_ + 1)

// Applicative
val applicativeExample: Option[Int] = (Some(5), Some(6)).mapN(_ + _)

// Monad
val monadExample: Option[Int] = Some(5).flatMap(x => Some(x + 1))

// Monad transformers
import cats.data.OptionT
val optionTExample: OptionT[List, Int] = 
  OptionT(List(Some(1), None, Some(2)))

// Sequence operations
val listSequence: Option[List[Int]] = List(Some(1), Some(2)).sequence
val mapMExample: Option[List[Int]] = List(1, 2).traverse(x => Some(x + 1))
Haskell
haskell
import Control.Applicative
import Control.Monad

-- Functor
functorExample :: Maybe Int
functorExample = fmap (+1) (Just 5)

-- Applicative
applicativeExample :: Maybe Int
applicativeExample = (+) <$> Just 5 <*> Just 6

-- Monad
monadExample :: Maybe Int
monadExample = Just 5 >>= \x -> Just (x + 1)

-- Monad transformers
import Control.Monad.Trans.Maybe (MaybeT(..))
import Control.Monad.Trans.Class (lift)

optionTExample :: MaybeT [] Int
optionTExample = MaybeT [Just 1, Nothing, Just 2]

-- Sequence operations
listSequence :: Maybe [Int]
listSequence = sequence [Just 1, Just 2]

mapMExample :: Maybe [Int]
mapMExample = mapM (\x -> Just (x + 1)) [1, 2]
Monoids
Scala
scala
import cats.Monoid
import cats.implicits._

val intMonoid: Monoid[Int] = Monoid[Int]
val combineInts: Int = intMonoid.combine(1, 2) // 3

val stringMonoid: Monoid[String] = Monoid[String]
val combineStrings: String = stringMonoid.combine("a", "b") // "ab"

// Fold with monoid
val sumList: Int = List(1, 2, 3).foldMap(identity)
val combined: Int = List(1, 2, 3).combineAll
Haskell
haskell
import Data.Monoid

intMonoid :: Monoid Int => Int
intMonoid = 1 <> 2  -- Uses Sum monoid

stringMonoid :: String
stringMonoid = "a" <> "b"

-- Fold with monoid
sumList :: Int
sumList = getSum $ foldMap Sum [1, 2, 3]

combined :: Int
combined = mconcat [1, 2, 3]  -- Uses Sum monoid
Arrows
Scala
scala
import cats.arrow._
import cats.implicits._

// Function composition
val f: Int => Int = _ + 1
val g: Int => Int = _ * 2
val composed: Int => Int = f >>> g

// Using Arrow type class
val arrowExample: (Int, Int) => Int = 
  Arrow[Function1].lift((x: Int) => x + 1)
Haskell
haskell
import Control.Arrow

-- Function composition with arrows
f :: Int -> Int
f = (+1)

g :: Int -> Int
g = (*2)

composed :: Int -> Int
composed = f >>> g

-- Arrow operations
arrowExample :: (Int, Int) -> Int
arrowExample = arr (+1)
Error Handling
Scala
scala
import cats.effect._
import cats.implicits._

// Either for error handling
def divide(a: Int, b: Int): Either[String, Int] =
  if (b == 0) Left("Division by zero") else Right(a / b)

// IO for effectful error handling
def safeDivide(a: Int, b: Int): IO[Int] =
  if (b == 0) IO.raiseError(new ArithmeticException("Division by zero"))
  else IO.pure(a / b)

// Using EitherT for monadic error handling
import cats.data.EitherT
import cats.effect.IO

def divideEitherT(a: Int, b: Int): EitherT[IO, String, Int] =
  EitherT(if (b == 0) IO.pure(Left("Division by zero")) 
          else IO.pure(Right(a / b)))

// Resource handling
def resourceExample: Resource[IO, String] =
  Resource.make(IO.println("Acquire") *> IO.pure("resource"))(
    _ => IO.println("Release")
  )
Haskell
haskell
import Control.Exception
import Control.Monad.Except

-- Either for error handling
divide :: Int -> Int -> Either String Int
divide a b = if b == 0 then Left "Division by zero" else Right (a `div` b)

-- IO with exceptions
safeDivide :: Int -> Int -> IO Int
safeDivide a b = if b == 0 then throwIO (userError "Division by zero") 
                 else return (a `div` b)

-- ExceptT for monadic error handling
divideExceptT :: Monad m => Int -> Int -> ExceptT String m Int
divideExceptT a b = if b == 0 then throwError "Division by zero" 
                    else return (a `div` b)

-- Resource handling
import Control.Exception (bracket)

resourceExample :: IO String
resourceExample = bracket
  (putStrLn "Acquire" >> return "resource")
  (\_ -> putStrLn "Release")
  (\r -> return r)
Concurrency
Scala (Cats-effect)
scala
import cats.effect._
import cats.effect.implicits._
import cats.implicits._
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

// Ref for shared state
def counterProgram: IO[Unit] = 
  Ref[IO].of(0).flatMap { ref =>
    ref.update(_ + 1) >> ref.get.flatMap(n => IO.println(s"Count: $n"))
  }

// Semaphore for coordination
def semaphoreExample: IO[Unit] = 
  Semaphore[IO](1).flatMap { semaphore =>
    semaphore.permit.use { _ =>
      IO.println("Critical section")
    }
  }
Haskell
haskell
import Control.Concurrent
import Control.Concurrent.Async
import Control.Monad

-- Async operations
concurrentProgram :: IO String
concurrentProgram = do
  putStrLn "Starting task"
  fiber <- async $ threadDelay 1000000 >> return "Result"
  putStrLn "Doing other work"
  wait fiber

-- Race conditions
import Control.Concurrent.Timeout
raceExample :: IO (Either String Int)
raceExample = race (threadDelay 500000 >> return "Timeout")
                   (threadDelay 1000000 >> return 42)

-- MVar for shared state
counterProgram :: IO ()
counterProgram = do
  mvar <- newMVar (0 :: Int)
  modifyMVar_ mvar (\n -> return (n + 1))
  n <- readMVar mvar
  putStrLn $ "Count: " ++ show n

-- Software Transactional Memory
import Control.Concurrent.STM

stmExample :: IO ()
stmExample = do
  tvar <- atomically $ newTVar (0 :: Int)
  atomically $ modifyTVar tvar (+1)
  n <- atomically $ readTVar tvar
  putStrLn $ "STM Count: " ++ show n
Stream Processing
Scala (FS2)
scala
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

val resourceStream: Stream[IO, Byte] = 
  Stream.resource(Blocker[IO]).flatMap { blocker =>
    io.file.readAll[IO](java.nio.file.Paths.get("test.txt"), blocker, 4096)
  }

val concurrentStream: Stream[IO, Int] = 
  Stream(1, 2, 3).mapAsync(2) { n =>
    IO.sleep(1.second).as(n * 2)
  }
Haskell
haskell
import Conduit
import Control.Monad.Trans.Resource
import qualified Data.ByteString as B

-- Basic stream
streamExample :: Monad m => ConduitT i Int m ()
streamExample = yieldMany [1, 2, 3] >> yield 4

-- Stream processing
processingStream :: MonadIO m => ConduitT i Int m ()
processingStream = yieldMany [1..9]
    .| filterC even
    .| mapC (*2)
    .| mapMC (\n -> liftIO (putStrLn ("Processing: " ++ show n)) >> return n)

-- Resource handling
resourceStream :: IO B.ByteString
resourceStream = runResourceT $ 
  sourceFile "test.txt" .| sinkLazy

-- Concurrent processing
import qualified Data.Conduit.Async as Async

concurrentStream :: IO [Int]
concurrentStream = runConduit $
  yieldMany [1, 2, 3] .| Async.mapMC 2 (\n -> threadDelay 1000000 >> return (n * 2)) .| sinkList
HTTP Libraries
Scala (http4s)
scala
import org.http4s._
import org.http4s.dsl.io._
import org.http4s.implicits._
import org.http4s.server.Router
import org.http4s.server.blaze.BlazeServerBuilder
import cats.effect._

val helloService = HttpRoutes.of[IO] {
  case GET -> Root / "hello" / name =>
    Ok(s"Hello, $name")
  case GET -> Root / "users" :? UserIdParam(userId) =>
    Ok(s"User ID: $userId")
}

object UserIdParam extends QueryParamDecoderMatcher[Int]("userId")

val httpApp = Router("/api" -> helloService).orNotFound

val server = BlazeServerBuilder[IO]
  .bindHttp(8080, "localhost")
  .withHttpApp(httpApp)
  .serve
Haskell (Servant/Warp)
haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}

import Servant
import Network.Wai
import Network.Wai.Handler.Warp

type API = "hello" :> Capture "name" String :> Get '[PlainText] String
      :<|> "users" :> QueryParam "userId" Int :> Get '[PlainText] String

server :: Server API
server = helloHandler :<|> usersHandler
  where
    helloHandler name = return $ "Hello, " ++ name
    usersHandler (Just userId) = return $ "User ID: " ++ show userId
    usersHandler Nothing = return "No user ID provided"

api :: Proxy API
api = Proxy

app :: Application
app = serve api server

main :: IO ()
main = run 8080 app
Kafka Integration
Scala (FD4s - FS2 Kafka)
scala
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
Haskell (kafka-client)
haskell
{-# LANGUAGE OverloadedStrings #-}

import Kafka.Producer
import Kafka.Consumer
import Control.Monad (forever, void)
import Data.Aeson (encode, object, (.=))
import Data.ByteString.Lazy (toStrict)

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
Elasticsearch Integration
Scala (elastic4s)
scala
import com.sksamuel.elastic4s.ElasticClient
import com.sksamuel.elastic4s.ElasticProperties
import com.sksamuel.elastic4s.circe._
import com.sksamuel.elastic4s.requests.indexes.IndexResponse
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
Haskell (bloodhound)
haskell
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Database.Bloodhound
import Network.HTTP.Client (newManager, defaultManagerSettings)
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
Complete Pipeline: Kafka + Elasticsearch
Scala (FD4s + elastic4s + FS2)
scala
import fs2.kafka._
import com.sksamuel.elastic4s.ElasticClient
import com.sksamuel.elastic4s.ElasticProperties
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
      .through(commitBatchWithin(100, 10.seconds))
  }
}
Haskell (Complete Pipeline)
haskell
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
    Just value -> case eitherDecode' value of
      Left err -> liftIO $ putStrLn $ "JSON error: " ++ err
      Right product -> do
        result <- liftIO $ runBH esClient $ indexProduct product
        case result of
          Right _ -> liftIO $ putStrLn "Indexed successfully"
          Left esErr -> liftIO $ putStrLn $ "ES error: " ++ show esErr
Key Differences Summary
Feature	Scala (Cats Ecosystem)	Haskell
Evaluation Strategy	Strict by default, lazy with IO	Lazy by default
Type Classes	Implicit instances, context bounds	Native type classes
Effect System	IO monad from cats-effect	Built-in IO, STM
Syntax	Method chaining, for-comprehensions	Function composition, do-notation
Concurrency	Fibers via cats-effect	Green threads, async
Error Handling	EitherT, IO.raiseError	Either, ExceptT, exceptions
Streaming	FS2 streams	Conduit, pipes, streaming
Kafka Integration	FD4s (FS2 Kafka)	kafka-client, haskell-warp
Elasticsearch	elastic4s	bloodhound
JSON Handling	Circe	Aeson
Resource Safety	Cats Effect Resource	ResourceT, bracket
Build Tool	sbt	Stack, Cabal
Performance Considerations
Scala Strengths:

JVM ecosystem and tooling

Better integration with existing Java libraries

Strong typing with less runtime overhead than dynamic languages

Excellent for data processing pipelines

Haskell Strengths:

Lazy evaluation can optimize certain operations

Pure functional nature enables aggressive optimizations

Lightweight threads (green threads)

Advanced type system catches more errors at compile time

Learning Curve
Scala:

Easier for developers coming from OOP backgrounds

Gradual adoption of functional programming

Rich IDE support

More job opportunities in industry

Haskell:

Steeper learning curve due to pure functional paradigm

Powerful type system requires deeper understanding

Strong academic and research community

Excellent for learning functional programming concepts

Ecosystem Maturity
Scala:

Mature ecosystem with extensive enterprise adoption

Strong support from Typelevel and ZIO communities

Excellent integration with big data tools (Spark, Kafka, etc.)

Active commercial support

Haskell:

Mature mathematical foundations

Strong in compiler technology and DSLs

Growing web development ecosystem

Excellent for correctness-critical applications

This guide provides a comprehensive comparison of functional programming concepts and practical applications between Scala's Cats ecosystem and Haskell. Both languages offer robust solutions for building reliable, scalable systems with different trade-offs in terms of learning curve, ecosystem, and performance characteristics.
