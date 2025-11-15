import Foundation
import Testing
@testable import NaturalLanguageEmbeddings

@Suite("Threshold Filtering Tests", .serialized)
struct ThresholdFilteringTests {
    let service: EmbeddingService

    init() async throws {
        service = try await EmbeddingService(specific: .script(.latin))
    }

    @Test("Search without threshold returns all results")
    func testNoThreshold() async throws {
        let sentences = [
            "Machine learning and artificial intelligence",
            "Cooking pasta with tomato sauce",
            "The weather is sunny today",
            "Deep neural networks for classification"
        ]

        var embeddings: [[Double]] = []
        for sentence in sentences {
            let embedding = try await service.generateEmbeddings(sentence, language: .english)
            embeddings.append(embedding)
        }

        let results = try await service.search(
            query: "artificial intelligence",
            in: embeddings
        )

        print("\n=== Search without threshold ===")
        print("Query: 'artificial intelligence'")
        print("Total results: \(results.count)")
        for (index, similarity) in results {
            print("  [\(index)] \(String(format: "%.4f", similarity)) - \(sentences[index])")
        }

        #expect(results.count == 4, "Should return all 4 results")
        print("✅ Returned all results\n")
    }

    @Test("Search with threshold filters low-relevance results")
    func testWithThreshold() async throws {
        let sentences = [
            "Machine learning and artificial intelligence",      // 0 - highly relevant
            "Deep neural networks for classification",          // 1 - relevant
            "Cooking pasta with tomato sauce",                  // 2 - unrelated
            "The weather is sunny today",                       // 3 - unrelated
            "Natural language processing with transformers"     // 4 - related
        ]

        var embeddings: [[Double]] = []
        for sentence in sentences {
            let embedding = try await service.generateEmbeddings(sentence, language: .english)
            embeddings.append(embedding)
        }

        let query = "artificial intelligence and machine learning"

        // Without threshold
        let allResults = try await service.search(query: query, in: embeddings)

        // With threshold 0.85
        let filteredResults = try await service.search(
            query: query,
            in: embeddings,
            minimumSimilarity: 0.85
        )

        print("\n=== Threshold Filtering Comparison ===")
        print("Query: '\(query)'")

        print("\n--- All Results (no threshold) ---")
        for (index, similarity) in allResults {
            let marker = similarity >= 0.85 ? "✓" : "✗"
            print("  \(marker) [\(index)] \(String(format: "%.4f", similarity)) - \(sentences[index].prefix(50))...")
        }

        print("\n--- Filtered Results (threshold ≥ 0.85) ---")
        for (index, similarity) in filteredResults {
            print("  ✓ [\(index)] \(String(format: "%.4f", similarity)) - \(sentences[index].prefix(50))...")
        }

        print("\nFiltering summary:")
        print("  Without threshold: \(allResults.count) results")
        print("  With threshold ≥ 0.85: \(filteredResults.count) results")
        print("  Filtered out: \(allResults.count - filteredResults.count) low-relevance results")

        #expect(filteredResults.count < allResults.count,
                "Threshold should filter out some results")
        #expect(filteredResults.allSatisfy { $0.1 >= 0.85 },
                "All filtered results should meet threshold")

        print("✅ Threshold filtering works correctly\n")
    }

    @Test("Different thresholds produce different result counts")
    func testMultipleThresholds() async throws {
        let sentences = Array(quotes.prefix(50))

        var embeddings: [[Double]] = []
        for sentence in sentences {
            let embedding = try await service.generateEmbeddings(sentence, language: .english)
            embeddings.append(embedding)
        }

        let query = "success and achievement"

        let thresholds: [Double] = [0.75, 0.80, 0.85, 0.90, 0.95]
        var resultCounts: [(Double, Int)] = []

        print("\n=== Threshold Impact Analysis ===")
        print("Query: '\(query)'")
        print("Dataset: 50 quotes")
        print("\nThreshold vs Result Count:")

        for threshold in thresholds {
            let results = try await service.search(
                query: query,
                in: embeddings,
                minimumSimilarity: threshold
            )
            resultCounts.append((threshold, results.count))
            print("  ≥ \(String(format: "%.2f", threshold)): \(String(format: "%2d", results.count)) results")
        }

        // Verify counts decrease as threshold increases
        for i in 0..<(resultCounts.count - 1) {
            let (threshold1, count1) = resultCounts[i]
            let (threshold2, count2) = resultCounts[i + 1]

            #expect(count1 >= count2,
                    "Higher threshold (\(threshold2)) should have ≤ results than lower threshold (\(threshold1))")
        }

        print("\n✅ Higher thresholds correctly reduce result count\n")
    }

    @Test("Threshold filtering for voice memo use case")
    func testVoiceMemoScenario() async throws {
        // Simulate voice memo transcriptions
        let memoTexts = [
            "Meeting with the design team about the new app interface. We discussed user onboarding flow and decided to add a tutorial.",
            "Doctor appointment reminder. Need to schedule a checkup next month. Also need to refill prescriptions.",
            "Project deadline is next Friday. Team members should finish their tasks by Wednesday for review.",
            "Grocery list: milk, eggs, bread, chicken, vegetables. Don't forget to buy cat food.",
            "Ideas for blog post about machine learning and natural language processing. Focus on practical applications.",
            "Called customer support about billing issue. They said it will be resolved in 3-5 business days.",
            "Team standup notes. Sarah working on authentication. Mike debugging the payment flow. Jane designing new icons.",
            "Weekend plans: hiking trip on Saturday, family dinner on Sunday.",
            "Reminder to review the quarterly financial reports before the board meeting.",
            "New app feature ideas: dark mode, offline sync, push notifications for important updates."
        ]

        var embeddings: [[Double]] = []
        for text in memoTexts {
            let embedding = try await service.generateEmbeddings(text, language: .english)
            embeddings.append(embedding)
        }

        // User searches for work-related memos
        let query = "project work and team meetings"

        print("\n=== Voice Memo Search Scenario ===")
        print("Query: '\(query)'")
        print("Total memos: \(memoTexts.count)")

        // Show results with different thresholds
        let thresholds: [Double] = [0.80, 0.85, 0.90]

        for threshold in thresholds {
            let results = try await service.search(
                query: query,
                in: embeddings,
                minimumSimilarity: threshold
            )

            print("\n--- Results with threshold ≥ \(String(format: "%.2f", threshold)) ---")
            if results.isEmpty {
                print("  No results above threshold")
            } else {
                for (index, similarity) in results.prefix(5) {
                    print("  [\(index)] \(String(format: "%.4f", similarity))")
                    print("      \(memoTexts[index].prefix(70))...")
                }
            }
        }

        // Recommended threshold for voice memos: 0.85
        let recommendedResults = try await service.search(
            query: query,
            in: embeddings,
            minimumSimilarity: 0.85
        )

        print("\n💡 Recommendation for voice memos: Use threshold ≥ 0.85")
        print("   This filters out unrelated memos while keeping relevant ones")
        print("   Found \(recommendedResults.count) relevant memos with this threshold")

        #expect(recommendedResults.count > 0,
                "Should find at least one relevant memo")

        print("✅ Voice memo search scenario validated\n")
    }
}
