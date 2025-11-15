import Foundation
import Testing
@testable import NaturalLanguageEmbeddings

@Suite("Comparison Tests: BasicEmbeddingService vs EmbeddingService", .serialized)
struct ComparisonTests {
    let basicService: BasicEmbeddingService
    let optimizedService: EmbeddingService

    init() async throws {
        basicService = try await BasicEmbeddingService(specific: .script(.latin))
        optimizedService = try await EmbeddingService(specific: .script(.latin))
    }

    @Test("Both services generate identical normalized embeddings")
    func testIdenticalEmbeddings() async throws {
        let sentences = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Machine learning with Swift is amazing",
            "Natural language processing enables semantic search"
        ]

        for sentence in sentences {
            let basicEmbedding = try await basicService.generateEmbeddings(sentence, language: .english)
            let optimizedEmbedding = try await optimizedService.generateEmbeddings(sentence, language: .english)

            #expect(basicEmbedding.count == optimizedEmbedding.count,
                    "Embeddings should have same dimension")

            // Compare each dimension - they should be very close (within floating point precision)
            for i in 0..<basicEmbedding.count {
                let diff = abs(basicEmbedding[i] - optimizedEmbedding[i])
                #expect(diff < 0.0001,
                        "Embedding values should match. Diff at index \(i): \(diff)")
            }
        }

        print("✅ Both services generate identical embeddings")
    }

    @Test("Both services return identical search results")
    func testIdenticalSearchResults() async throws {
        let sentences = [
            "The cat sits on the mat.",
            "A dog is playing in the yard.",
            "Feline animals love to sleep.",
            "Python is a programming language.",
            "Swift is used for iOS development.",
            "Birds fly in the sky.",
            "Fish swim in the ocean.",
        ]

        // Generate embeddings using both services
        var basicEmbeddings: [[Double]] = []
        var optimizedEmbeddings: [[Double]] = []

        for sentence in sentences {
            let basicEmb = try await basicService.generateEmbeddings(sentence, language: .english)
            let optimizedEmb = try await optimizedService.generateEmbeddings(sentence, language: .english)

            basicEmbeddings.append(basicEmb)
            optimizedEmbeddings.append(optimizedEmb)
        }

        // Test multiple queries
        let queries = [
            "cats and kittens",
            "programming languages",
            "animals in nature"
        ]

        for query in queries {
            let basicResults = try await basicService.search(query: query, in: basicEmbeddings)
            let optimizedResults = try await optimizedService.search(query: query, in: optimizedEmbeddings)

            print("\n=== Query: '\(query)' ===")
            print("Basic service results:")
            for (index, similarity) in basicResults.prefix(3) {
                print("  [\(index)] \(String(format: "%.4f", similarity)) - \(sentences[index])")
            }

            print("\nOptimized service results:")
            for (index, similarity) in optimizedResults.prefix(3) {
                print("  [\(index)] \(String(format: "%.4f", similarity)) - \(sentences[index])")
            }

            // Verify same ranking
            #expect(basicResults.count == optimizedResults.count,
                    "Both services should return same number of results")

            for i in 0..<basicResults.count {
                let basicIndex = basicResults[i].0
                let optimizedIndex = optimizedResults[i].0

                #expect(basicIndex == optimizedIndex,
                        "Ranking should be identical at position \(i). Basic: \(basicIndex), Optimized: \(optimizedIndex)")

                // Similarities should be very close
                let basicSim = basicResults[i].1
                let optimizedSim = optimizedResults[i].1
                let diff = abs(basicSim - optimizedSim)

                #expect(diff < 0.0001,
                        "Similarities should match at position \(i). Diff: \(diff)")
            }

            print("✅ Both services return identical rankings\n")
        }
    }

    @Test("Both services handle the quotes dataset identically")
    func testQuotesDataset() async throws {
        // Test with a subset of quotes
        let testQuotes = Array(quotes.prefix(20))

        var basicEmbeddings: [[Double]] = []
        var optimizedEmbeddings: [[Double]] = []

        for quote in testQuotes {
            let basicEmb = try await basicService.generateEmbeddings(quote, language: .english)
            let optimizedEmb = try await optimizedService.generateEmbeddings(quote, language: .english)

            basicEmbeddings.append(basicEmb)
            optimizedEmbeddings.append(optimizedEmb)
        }

        let queries = ["success", "courage", "knowledge"]

        for query in queries {
            let basicResults = try await basicService.search(query: query, in: basicEmbeddings)
            let optimizedResults = try await optimizedService.search(query: query, in: optimizedEmbeddings)

            print("\n=== Query: '\(query)' ===")
            print("Top 3 results comparison:")

            for i in 0..<min(3, basicResults.count) {
                let basicIdx = basicResults[i].0
                let optimizedIdx = optimizedResults[i].0
                let basicSim = basicResults[i].1
                let optimizedSim = optimizedResults[i].1

                print("  Position \(i + 1):")
                print("    Basic:     [\(basicIdx)] \(String(format: "%.4f", basicSim))")
                print("    Optimized: [\(optimizedIdx)] \(String(format: "%.4f", optimizedSim))")

                #expect(basicIdx == optimizedIdx,
                        "Indices should match at position \(i)")

                let diff = abs(basicSim - optimizedSim)
                #expect(diff < 0.0001,
                        "Similarities should match. Diff: \(diff)")
            }

            print("✅ Passed\n")
        }
    }

    @Test("Normalization is identical in both services")
    func testNormalizationEquality() async throws {
        let sentence = "Testing normalization consistency"

        let basicEmbedding = try await basicService.generateEmbeddings(sentence, language: .english)
        let optimizedEmbedding = try await optimizedService.generateEmbeddings(sentence, language: .english)

        // Calculate norms
        let basicNorm = sqrt(basicEmbedding.reduce(0) { $0 + $1 * $1 })
        let optimizedNorm = sqrt(optimizedEmbedding.reduce(0) { $0 + $1 * $1 })

        print("\n=== Normalization Test ===")
        print("Basic service L2 norm:     \(basicNorm)")
        print("Optimized service L2 norm: \(optimizedNorm)")

        #expect(abs(basicNorm - 1.0) < 0.0001,
                "Basic service embeddings should be normalized")
        #expect(abs(optimizedNorm - 1.0) < 0.0001,
                "Optimized service embeddings should be normalized")

        let normDiff = abs(basicNorm - optimizedNorm)
        #expect(normDiff < 0.0001,
                "Norms should be identical. Diff: \(normDiff)")

        print("✅ Both services normalize identically\n")
    }
}
