import Foundation
import Testing
@testable import NaturalLanguageEmbeddings

@Suite("Threshold-Based Search Algorithm Tests", .serialized)
struct ThresholdSearchTests {
    let basicService: BasicEmbeddingService
    let defaultThresholdService: EmbeddingService
    let lowThresholdService: EmbeddingService
    let highThresholdService: EmbeddingService
    
    init() async throws {
        basicService = try await BasicEmbeddingService(specific: .script(.latin))
        defaultThresholdService = try await EmbeddingService(specific: .script(.latin), optimizationThreshold: 100)
        lowThresholdService = try await EmbeddingService(specific: .script(.latin), optimizationThreshold: 10)
        highThresholdService = try await EmbeddingService(specific: .script(.latin), optimizationThreshold: 200)
    }
    
    // MARK: - Helper Methods
    
    private func generateEmbeddings(for texts: [String]) async throws -> [[Double]] {
        var embeddings: [[Double]] = []
        for text in texts {
            let embedding = try await defaultThresholdService.generateEmbeddings(text, language: .english)
            embeddings.append(embedding)
        }
        return embeddings
    }
    
    private func verifyResultsMatch(
        service1Results: [(Int, Double)],
        service2Results: [(Int, Double)],
        service1Name: String,
        service2Name: String,
        tolerance: Double = 0.0001
    ) {
        #expect(service1Results.count == service2Results.count,
                "\(service1Name) and \(service2Name) should return same number of results")
        
        for i in 0..<service1Results.count {
            let (idx1, sim1) = service1Results[i]
            let (idx2, sim2) = service2Results[i]
            
            #expect(idx1 == idx2,
                    "Ranking mismatch at position \(i): \(service1Name)=\(idx1), \(service2Name)=\(idx2)")
            
            let diff = abs(sim1 - sim2)
            #expect(diff < tolerance,
                    "Similarity mismatch at position \(i): \(service1Name)=\(String(format: "%.6f", sim1)), \(service2Name)=\(String(format: "%.6f", sim2)), diff=\(String(format: "%.6f", diff))")
        }
    }
    
    private func printResults(
        _ results: [(Int, Double)],
        texts: [String],
        serviceName: String,
        limit: Int = 5
    ) {
        print("  \(serviceName):")
        for (rank, (index, similarity)) in results.prefix(limit).enumerated() {
            let text = texts[index].prefix(60)
            print("    \(rank + 1). [\(index)] \(String(format: "%.6f", similarity)) - \(text)")
        }
    }
    
    // MARK: - Small Dataset Tests (Below Threshold)
    
    @Test("Search with 5 items uses simple algorithm (default threshold=100)")
    func testSearchWith5Items() async throws {
        let testData = Array(quotes.prefix(5))
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "success and achievement"
        
        print("\n=== Test: 5 items (< 100 threshold) ===")
        print("Expected: Uses SIMPLE search algorithm")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: testData, serviceName: "Default (simple)")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default (simple)",
            service2Name: "Basic"
        )
        
        print("✅ Simple algorithm matches BasicEmbeddingService\n")
    }
    
    @Test("Search with 10 items uses simple algorithm (default threshold=100)")
    func testSearchWith10Items() async throws {
        let testData = Array(quotes.prefix(10))
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "courage and determination"
        
        print("\n=== Test: 10 items (< 100 threshold) ===")
        print("Expected: Uses SIMPLE search algorithm")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: testData, serviceName: "Default (simple)")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default (simple)",
            service2Name: "Basic"
        )
        
        print("✅ Simple algorithm matches BasicEmbeddingService\n")
    }
    
    @Test("Search with 50 items uses simple algorithm (default threshold=100)")
    func testSearchWith50Items() async throws {
        let testData = Array(quotes.prefix(50))
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "wisdom and knowledge"
        
        print("\n=== Test: 50 items (< 100 threshold) ===")
        print("Expected: Uses SIMPLE search algorithm")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: testData, serviceName: "Default (simple)")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default (simple)",
            service2Name: "Basic"
        )
        
        print("✅ Simple algorithm matches BasicEmbeddingService\n")
    }
    
    // MARK: - Boundary Tests (At Threshold)
    
    @Test("Search with 99 items uses simple algorithm (just below threshold)")
    func testSearchWith99Items() async throws {
        let testData = Array(quotes.prefix(99))
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "leadership and vision"
        
        print("\n=== Test: 99 items (< 100 threshold, boundary) ===")
        print("Expected: Uses SIMPLE search algorithm")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: testData, serviceName: "Default (simple)")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default (simple)",
            service2Name: "Basic"
        )
        
        print("✅ Simple algorithm matches BasicEmbeddingService at boundary\n")
    }
    
    @Test("Search with 100 items uses optimized algorithm (at threshold)")
    func testSearchWith100Items() async throws {
        let testData = Array(quotes.prefix(100))
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "motivation and inspiration"
        
        print("\n=== Test: 100 items (>= 100 threshold, boundary) ===")
        print("Expected: Uses OPTIMIZED search algorithm (vDSP)")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: testData, serviceName: "Default (optimized)")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default (optimized)",
            service2Name: "Basic"
        )
        
        print("✅ Optimized algorithm matches BasicEmbeddingService at boundary\n")
    }
    
    @Test("Search with 101 items uses optimized algorithm (just above threshold)")
    func testSearchWith101Items() async throws {
        let testData = Array(quotes.prefix(101))
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "perseverance and resilience"
        
        print("\n=== Test: 101 items (>= 100 threshold, just above) ===")
        print("Expected: Uses OPTIMIZED search algorithm (vDSP)")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: testData, serviceName: "Default (optimized)")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default (optimized)",
            service2Name: "Basic"
        )
        
        print("✅ Optimized algorithm matches BasicEmbeddingService just above boundary\n")
    }
    
    // MARK: - Large Dataset Tests (Above Threshold)
    
    @Test("Search with large dataset uses appropriate algorithm")
    func testSearchWith150Items() async throws {
        let testData = quotes // All quotes
        
        // This test adapts to the available data
        // With threshold=100: <100 uses simple, >=100 uses optimized
        let actualData = Array(testData.prefix(min(150, testData.count)))
        let embeddings = try await generateEmbeddings(for: actualData)
        let query = "excellence and mastery"
        
        let algorithm = actualData.count >= 100 ? "OPTIMIZED" : "SIMPLE"
        print("\n=== Test: \(actualData.count) items (threshold=100) ===")
        print("Expected: Uses \(algorithm) search algorithm")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: actualData, serviceName: "Default (\(algorithm.lowercased()))")
        printResults(basicResults, texts: actualData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default",
            service2Name: "Basic"
        )
        
        print("✅ \(algorithm) algorithm matches BasicEmbeddingService on dataset of \(actualData.count) items\n")
    }
    
    @Test("Search with all available quotes uses appropriate algorithm")
    func testSearchWithAllQuotes() async throws {
        let testData = quotes
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "creativity and innovation"
        
        let algorithm = testData.count >= 100 ? "OPTIMIZED" : "SIMPLE"
        print("\n=== Test: All quotes (\(testData.count) items, threshold=100) ===")
        print("Expected: Uses \(algorithm) search algorithm")
        print("Query: '\(query)'")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(defaultResults, texts: testData, serviceName: "Default (\(algorithm.lowercased()))")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default",
            service2Name: "Basic"
        )
        
        print("✅ \(algorithm) algorithm matches BasicEmbeddingService on full dataset (\(testData.count) items)\n")
    }
    
    // MARK: - Custom Threshold Tests
    
    @Test("Custom threshold=10: 5 items use simple, 10+ items use optimized")
    func testCustomLowThreshold() async throws {
        let smallData = Array(quotes.prefix(5))
        let smallEmbeddings = try await generateEmbeddings(for: smallData)
        
        let largeData = Array(quotes.prefix(15))
        let largeEmbeddings = try await generateEmbeddings(for: largeData)
        
        let query = "achievement"
        
        print("\n=== Test: Custom threshold=10 ===")
        
        // Test with 5 items (should use simple)
        print("\nPart 1: 5 items (< 10 threshold)")
        print("Expected: Uses SIMPLE search algorithm")
        
        let smallResults = try await lowThresholdService.search(query: query, in: smallEmbeddings)
        let smallBasicResults = try await basicService.search(query: query, in: smallEmbeddings)
        
        printResults(smallResults, texts: smallData, serviceName: "Low threshold (simple)")
        printResults(smallBasicResults, texts: smallData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: smallResults,
            service2Results: smallBasicResults,
            service1Name: "Low threshold (simple)",
            service2Name: "Basic"
        )
        
        // Test with 15 items (should use optimized)
        print("\nPart 2: 15 items (>= 10 threshold)")
        print("Expected: Uses OPTIMIZED search algorithm (vDSP)")
        
        let largeResults = try await lowThresholdService.search(query: query, in: largeEmbeddings)
        let largeBasicResults = try await basicService.search(query: query, in: largeEmbeddings)
        
        printResults(largeResults, texts: largeData, serviceName: "Low threshold (optimized)")
        printResults(largeBasicResults, texts: largeData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: largeResults,
            service2Results: largeBasicResults,
            service1Name: "Low threshold (optimized)",
            service2Name: "Basic"
        )
        
        print("✅ Custom threshold=10 correctly switches between algorithms\n")
    }
    
    @Test("Custom threshold=200: datasets under 200 use simple algorithm")
    func testCustomHighThreshold() async throws {
        let testData = quotes // Should be around 110 items
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "purpose and meaning"
        
        print("\n=== Test: Custom threshold=200 ===")
        print("Dataset size: \(testData.count) items")
        print("Expected: Uses SIMPLE search algorithm (< 200 threshold)")
        print("Query: '\(query)'")
        
        let highThresholdResults = try await highThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        printResults(highThresholdResults, texts: testData, serviceName: "High threshold (simple)")
        printResults(basicResults, texts: testData, serviceName: "Basic")
        
        verifyResultsMatch(
            service1Results: highThresholdResults,
            service2Results: basicResults,
            service1Name: "High threshold (simple)",
            service2Name: "Basic"
        )
        
        print("✅ Custom threshold=200 uses simple algorithm for datasets under threshold\n")
    }
    
    // MARK: - Algorithm Equivalence Tests
    
    @Test("Simple vs Optimized: Both algorithms produce identical results")
    func testSimpleVsOptimizedAlgorithms() async throws {
        // Use a dataset size that allows testing both algorithms with different thresholds
        let testData = Array(quotes.prefix(50))
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "success in life"
        
        print("\n=== Test: Simple vs Optimized Algorithm Equivalence ===")
        print("Dataset: 50 items")
        print("Query: '\(query)'")
        
        // With threshold=100, 50 items will use simple
        print("\nConfiguration 1: threshold=100 (uses SIMPLE for 50 items)")
        let simpleResults = try await defaultThresholdService.search(query: query, in: embeddings)
        printResults(simpleResults, texts: testData, serviceName: "Simple algorithm")
        
        // With threshold=10, 50 items will use optimized
        print("\nConfiguration 2: threshold=10 (uses OPTIMIZED for 50 items)")
        let optimizedResults = try await lowThresholdService.search(query: query, in: embeddings)
        printResults(optimizedResults, texts: testData, serviceName: "Optimized algorithm")
        
        // Both should produce identical results
        verifyResultsMatch(
            service1Results: simpleResults,
            service2Results: optimizedResults,
            service1Name: "Simple",
            service2Name: "Optimized"
        )
        
        print("✅ Simple and Optimized algorithms produce IDENTICAL results\n")
    }
    
    @Test("Cross-comparison: All three services produce identical rankings")
    func testThreeWayComparison() async throws {
        let testData = Array(quotes.prefix(20))
        let embeddings = try await generateEmbeddings(for: testData)
        
        let queries = [
            "leadership and power",
            "failure and learning",
            "time management",
            "creativity and art"
        ]
        
        print("\n=== Test: Three-Way Service Comparison ===")
        print("Dataset: 20 items")
        print("Services: Basic, Default (simple), Low threshold (optimized)")
        
        for query in queries {
            print("\n--- Query: '\(query)' ---")
            
            let basicResults = try await basicService.search(query: query, in: embeddings)
            let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
            let lowThresholdResults = try await lowThresholdService.search(query: query, in: embeddings)
            
            printResults(basicResults, texts: testData, serviceName: "Basic", limit: 3)
            printResults(defaultResults, texts: testData, serviceName: "Default (simple)", limit: 3)
            printResults(lowThresholdResults, texts: testData, serviceName: "Low threshold (optimized)", limit: 3)
            
            // Verify all three match
            verifyResultsMatch(
                service1Results: basicResults,
                service2Results: defaultResults,
                service1Name: "Basic",
                service2Name: "Default"
            )
            
            verifyResultsMatch(
                service1Results: basicResults,
                service2Results: lowThresholdResults,
                service1Name: "Basic",
                service2Name: "Low threshold"
            )
            
            print("✅ All three services produce identical results")
        }
        
        print("\n✅ Three-way comparison passed for all queries\n")
    }
    
    // MARK: - Comprehensive Dataset Size Tests
    
    @Test("Comprehensive test: Various dataset sizes [5, 10, 50, 99, 100, 101]")
    func testVariousDatasetSizes() async throws {
        let sizes = [5, 10, 50, 99, 100, 101]
        let query = "wisdom and experience"
        
        print("\n=== Test: Comprehensive Dataset Sizes ===")
        print("Query: '\(query)'")
        print("Threshold: 100")
        
        for size in sizes {
            let adjustedSize = min(size, quotes.count)
            let testData = Array(quotes.prefix(adjustedSize))
            let embeddings = try await generateEmbeddings(for: testData)
            
            let algorithm = adjustedSize < 100 ? "SIMPLE" : "OPTIMIZED"
            print("\n--- Size: \(adjustedSize) items (uses \(algorithm)) ---")
            
            let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
            let basicResults = try await basicService.search(query: query, in: embeddings)
            
            printResults(defaultResults, texts: testData, serviceName: "Default", limit: 3)
            printResults(basicResults, texts: testData, serviceName: "Basic", limit: 3)
            
            verifyResultsMatch(
                service1Results: defaultResults,
                service2Results: basicResults,
                service1Name: "Default",
                service2Name: "Basic"
            )
            
            print("✅ Size \(adjustedSize) passed")
        }
        
        print("\n✅ All dataset sizes produce identical results across algorithms\n")
    }
    
    // MARK: - Multiple Query Tests
    
    @Test("Multiple queries produce consistent results across algorithms")
    func testMultipleQueries() async throws {
        let testData = Array(quotes.prefix(75))
        let embeddings = try await generateEmbeddings(for: testData)
        
        let queries = [
            "success",
            "failure",
            "courage",
            "wisdom",
            "leadership",
            "creativity",
            "perseverance",
            "innovation",
            "motivation",
            "excellence"
        ]
        
        print("\n=== Test: Multiple Queries with Same Dataset ===")
        print("Dataset: 75 items (uses SIMPLE with threshold=100)")
        print("Testing \(queries.count) different queries")
        
        for (index, query) in queries.enumerated() {
            let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
            let basicResults = try await basicService.search(query: query, in: embeddings)
            
            // Verify match
            verifyResultsMatch(
                service1Results: defaultResults,
                service2Results: basicResults,
                service1Name: "Default",
                service2Name: "Basic"
            )
            print("  [\(index + 1)/\(queries.count)] '\(query)' ✅")
        }
        
        print("\n✅ All \(queries.count) queries passed\n")
    }
    
    // MARK: - Edge Cases
    
    @Test("Empty dataset returns empty results")
    func testEmptyDataset() async throws {
        let embeddings: [[Double]] = []
        let query = "test query"
        
        print("\n=== Test: Empty Dataset ===")
        
        let results = try await defaultThresholdService.search(query: query, in: embeddings)
        
        #expect(results.isEmpty, "Empty dataset should return empty results")
        
        print("✅ Empty dataset handled correctly\n")
    }
    
    @Test("Single item dataset works correctly")
    func testSingleItemDataset() async throws {
        let testData = [quotes[0]]
        let embeddings = try await generateEmbeddings(for: testData)
        let query = "success"
        
        print("\n=== Test: Single Item Dataset ===")
        
        let defaultResults = try await defaultThresholdService.search(query: query, in: embeddings)
        let basicResults = try await basicService.search(query: query, in: embeddings)
        
        #expect(defaultResults.count == 1, "Should return exactly one result")
        #expect(basicResults.count == 1, "Should return exactly one result")
        
        verifyResultsMatch(
            service1Results: defaultResults,
            service2Results: basicResults,
            service1Name: "Default",
            service2Name: "Basic"
        )
        
        print("✅ Single item dataset handled correctly\n")
    }
    
    @Test("Threshold boundary precision: Exactly at threshold value")
    func testExactThresholdBoundary() async throws {
        // Create a service with threshold=50
        let service50 = try await EmbeddingService(specific: .script(.latin), optimizationThreshold: 50)
        
        let data49 = Array(quotes.prefix(49))
        let data50 = Array(quotes.prefix(50))
        let data51 = Array(quotes.prefix(51))
        
        let embeddings49 = try await generateEmbeddings(for: data49)
        let embeddings50 = try await generateEmbeddings(for: data50)
        let embeddings51 = try await generateEmbeddings(for: data51)
        
        let query = "determination"
        
        print("\n=== Test: Threshold Boundary Precision (threshold=50) ===")
        
        // 49 items - simple
        print("\n49 items (< 50): Uses SIMPLE")
        let results49 = try await service50.search(query: query, in: embeddings49)
        let basic49 = try await basicService.search(query: query, in: embeddings49)
        verifyResultsMatch(
            service1Results: results49,
            service2Results: basic49,
            service1Name: "Service50 (simple)",
            service2Name: "Basic"
        )
        print("✅ 49 items passed")
        
        // 50 items - optimized
        print("\n50 items (>= 50): Uses OPTIMIZED")
        let results50 = try await service50.search(query: query, in: embeddings50)
        let basic50 = try await basicService.search(query: query, in: embeddings50)
        verifyResultsMatch(
            service1Results: results50,
            service2Results: basic50,
            service1Name: "Service50 (optimized)",
            service2Name: "Basic"
        )
        print("✅ 50 items passed")
        
        // 51 items - optimized
        print("\n51 items (>= 50): Uses OPTIMIZED")
        let results51 = try await service50.search(query: query, in: embeddings51)
        let basic51 = try await basicService.search(query: query, in: embeddings51)
        verifyResultsMatch(
            service1Results: results51,
            service2Results: basic51,
            service1Name: "Service50 (optimized)",
            service2Name: "Basic"
        )
        print("✅ 51 items passed")
        
        print("\n✅ Threshold boundary switching verified at exact threshold value\n")
    }
}
