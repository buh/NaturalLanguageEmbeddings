import Foundation
import Testing
@testable import NaturalLanguageEmbeddings

@Suite("BasicEmbeddingServiceTests", .serialized)
struct BasicEmbeddingServiceTests {
    let service : BasicEmbeddingService

    init() async throws {
        service = try await BasicEmbeddingService(specific: .script(.latin))
    }

    @Test("Similar sentences have higher similarity than dissimilar ones")
    func testSimilarityOrdering() async throws {
        // Create embeddings for test sentences
        let sentences = [
            "The cat sits on the mat.",           // 0 - about cats
            "A dog is playing in the yard.",      // 1 - about dogs
            "Feline animals love to sleep.",      // 2 - about cats (feline)
            "Python is a programming language.",  // 3 - about programming
            "Swift is used for iOS development.", // 4 - about programming
        ]

        var embeddings: [[Double]] = []
        for sentence in sentences {
            let embedding = try await service.generateEmbeddings(sentence, language: .english)
            embeddings.append(embedding)
        }

        // Query about cats - should find cat-related sentences first
        let results = try await service.search(query: "cats and kittens", in: embeddings)

        print("\n=== Test: Similar sentences have higher similarity ===")
        print("Query: 'cats and kittens'")
        print("Expected: Cat-related sentences (0, 2) should rank higher than others")
        print("\nResults:")
        for (index, similarity) in results.prefix(5) {
            print("  [\(index)] \(sentences[index])")
            print("      Similarity: \(String(format: "%.4f", similarity))")
        }

        // The top result should be one of the cat-related sentences (0 or 2)
        let topIndex = results[0].0
        #expect(topIndex == 0 || topIndex == 2,
                "Top result should be cat-related (index 0 or 2), but got index \(topIndex)")

        // Programming sentences should rank lower than cat sentences
        let catIndices = Set([0, 2])
        let programmingIndices = Set([3, 4])

        // Find best ranking for cat sentences and programming sentences
        var bestCatRank = Int.max
        var bestProgrammingRank = Int.max

        for (rank, (index, _)) in results.enumerated() {
            if catIndices.contains(index) && rank < bestCatRank {
                bestCatRank = rank
            }
            if programmingIndices.contains(index) && rank < bestProgrammingRank {
                bestProgrammingRank = rank
            }
        }

        #expect(bestCatRank < bestProgrammingRank,
                "Cat sentences should rank higher than programming sentences")

        print("\n✅ Cat sentences ranked higher than programming sentences")
        print("===\n")
    }

    @Test("Search returns results in descending order of similarity")
    func testSearchOrdering() async throws {
        let sentences = [
            "Apple pie recipe",
            "Banana smoothie",
            "Cherry tart",
            "Date cookies"
        ]

        var embeddings: [[Double]] = []
        for sentence in sentences {
            let embedding = try await service.generateEmbeddings(sentence, language: .english)
            embeddings.append(embedding)
        }

        let results = try await service.search(query: "dessert recipes", in: embeddings)

        // Verify results are sorted in descending order
        for i in 0..<(results.count - 1) {
            let currentSimilarity = results[i].1
            let nextSimilarity = results[i + 1].1
            #expect(currentSimilarity >= nextSimilarity,
                    "Results should be sorted by similarity (descending)")
        }
    }

    @Test("Identical sentences have very high similarity (≈ 1.0)")
    func testIdenticalSentences() async throws {
        let sentence = "This is a test sentence for embeddings."

        let embedding1 = try await service.generateEmbeddings(sentence, language: .english)
        let embedding2 = try await service.generateEmbeddings(sentence, language: .english)

        let results = try await service.search(query: sentence, in: [embedding1, embedding2])

        // Both should have similarity very close to 1.0
        for (_, similarity) in results {
            #expect(similarity > 0.99, "Identical sentences should have similarity ≈ 1.0, got \(similarity)")
        }
    }

    @Test("generateEmbeddings returns L2-normalized vectors")
    func testEmbeddingsAreNormalized() async throws {
        let sentences = ["hello world", "machine learning", "the quick brown fox"]
        for sentence in sentences {
            let embedding = try await service.generateEmbeddings(sentence, language: .english)
            let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
            #expect(abs(norm - 1.0) < 1e-6, "Embedding for '\(sentence)' should be unit norm, got \(norm)")
        }
    }

    @Test("generateEmbeddings throws on empty string")
    func testEmptyStringThrows() async throws {
        await #expect(throws: EmbeddingError.self) {
            try await service.generateEmbeddings("")
        }
    }

    @Test("Test for indirect topics with expected rankings")
    func indirectTopics() async throws {
        let sentences = [
            "The car battery was completely dead this morning.",           // 0
            "I need to buy a new charger for my electric scooter.",        // 1
            "Our company is hiring iOS developers with SwiftUI experience.", // 2
            "Apple just announced a new update for macOS.",                // 3
            "He brewed a cup of coffee and started reading the news.",     // 4
            "The weather forecast predicts heavy rain tomorrow.",          // 5
            "I visited Amsterdam last summer and loved the canals.",       // 6
            "The patient needs to book an appointment with a cardiologist.", // 7
            "This recipe uses fresh basil and olive oil for the dressing.", // 8
            "The system crashed after the last software update.",          // 9
            "We need a meeting room with a projector for the presentation.", // 10
            "The electric vehicle market is growing rapidly.",             // 11
            "He's been running every morning to improve his stamina."      // 12
        ]

        var embeddings: [[Double]] = []
        for sentence in sentences {
            let embedding = try await service.generateEmbeddings(sentence, language: .english)
            embeddings.append(embedding)
        }

        // Test case: Looking for Swift engineers
        let swiftQuery = "Looking for Swift engineers"
        let swiftResults = try await service.search(query: swiftQuery, in: embeddings)

        print("\n=== Test: '\(swiftQuery)' ===")
        print("Expected: Index 2 (hiring iOS developers) should rank high")
        for (index, similarity) in swiftResults.prefix(3) {
            print("  [\(index)] \(String(format: "%.4f", similarity)) - \(sentences[index])")
        }

        // The hiring sentence (index 2) should be in top 3
        let swiftTopIndices = swiftResults.prefix(3).map { $0.0 }
        #expect(swiftTopIndices.contains(2),
                "Hiring iOS developers sentence should be in top 3 results")
        print("✅ Passed\n")

        // Test case: Weather/rain query
        let rainQuery = "Rain is expected next week"
        let rainResults = try await service.search(query: rainQuery, in: embeddings)

        print("=== Test: '\(rainQuery)' ===")
        print("Expected: Index 5 (weather forecast rain) should rank high")
        for (index, similarity) in rainResults.prefix(3) {
            print("  [\(index)] \(String(format: "%.4f", similarity)) - \(sentences[index])")
        }

        let rainTopIndices = rainResults.prefix(3).map { $0.0 }
        #expect(rainTopIndices.contains(5),
                "Weather forecast sentence should be in top 3 results")
        print("✅ Passed\n")
    }

    @Test("Test quotes with specific expected results")
    func testQuotesSimilarity() async throws {
        // Use a smaller subset for clearer tests
        let testQuotes = [
            "Success is walking from failure to failure with no loss of enthusiasm. ~Winston Churchill",
            "The distance between insanity and genius is measured only by success. ~Bruce Feirstein",
            "Knowledge is being aware of what you can do. Wisdom is knowing when not to do it. ~Anonymous",
            "Courage is resistance to fear, mastery of fear - not absense of fear. ~Mark Twain",
            "Innovation distinguishes between a leader and a follower. ~Steve Jobs",
        ]

        var embeddings: [[Double]] = []
        for quote in testQuotes {
            let embedding = try await service.generateEmbeddings(quote, language: .english)
            embeddings.append(embedding)
        }

        // Test: "insanity" should find the Bruce Feirstein quote (index 1)
        let insanityResults = try await service.search(query: "insanity", in: embeddings)

        print("\n=== Test: 'insanity' query ===")
        print("Expected: Index 1 (insanity and genius quote) should rank #1")
        for (index, similarity) in insanityResults.prefix(3) {
            print("  [\(index)] \(String(format: "%.4f", similarity)) - \(testQuotes[index])")
        }

        #expect(insanityResults[0].0 == 1,
                "Insanity query should return the Bruce Feirstein quote first")
        print("✅ Passed\n")

        // Test: "leader" should find Steve Jobs quote (index 4)
        let leaderResults = try await service.search(query: "leader", in: embeddings)

        print("=== Test: 'leader' query ===")
        print("Expected: Index 4 (leader and follower quote) should rank high")
        for (index, similarity) in leaderResults.prefix(3) {
            print("  [\(index)] \(String(format: "%.4f", similarity)) - \(testQuotes[index])")
        }

        let leaderTopIndices = leaderResults.prefix(2).map { $0.0 }
        #expect(leaderTopIndices.contains(4),
                "Leader query should return Steve Jobs quote in top 2")
        print("✅ Passed\n")
    }
}
