# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NaturalLanguageEmbeddings is a Swift package that provides semantic text embeddings using Apple's NLContextualEmbedding framework. It offers two implementations: a simple `BasicEmbeddingService` and an optimized `EmbeddingService` that uses Accelerate framework for better performance on large datasets.

## Common Commands

### Building and Testing
```bash
# Build the package
swift build

# Run all tests
swift test

# Build for specific platform
swift build -c release
```

### Running Individual Tests
```bash
# Run specific test class
swift test --filter BasicEmbeddingServiceTests
swift test --filter EmbeddingServiceTests
swift test --filter ComparisonTests

# Run specific test method
swift test --filter BasicEmbeddingServiceTests/testEmbeddingsAreNormalized
```

## Architecture

### Two Service Implementations

The package provides two embedding service implementations with identical APIs:

1. **BasicEmbeddingService** (`Sources/NaturalLanguageEmbeddings/BasicEmbeddingService.swift`)
   - Simple, straightforward implementation
   - Uses standard Swift operations for normalization and similarity calculations
   - Best for small datasets or when simplicity is preferred
   - Acts as reference implementation for correctness

2. **EmbeddingService** (`Sources/NaturalLanguageEmbeddings/EmbeddingService.swift`)
   - Performance-optimized implementation using Accelerate framework (vDSP)
   - Adaptive search algorithm: simple loop for <100 items, vDSP matrix multiplication for ≥100 items
   - Uses `vDSP_mmulD` for batch similarity calculations on large datasets
   - Recommended for production use with large embedding collections

### Key Design Principles

**Independence from Content**: Both services return `[(Int, Double)]` tuples (index, similarity) rather than document content. This makes them reusable across different data types - callers maintain their own document arrays and use returned indices to retrieve content.

**Mean Pooling Strategy**: Both implementations use mean pooling to aggregate token embeddings:
- Enumerate all token vectors from NLContextualEmbedding
- Accumulate vectors in-place using vDSP (EmbeddingService) or standard operations (BasicEmbeddingService)
- Divide by token count to compute mean

**L2 Normalization**: All embeddings are L2-normalized (unit vectors) so cosine similarity becomes equivalent to dot product, enabling vDSP optimization.

**Identical Results Guarantee**: The `ComparisonTests` suite verifies both implementations return identical embeddings and search results.

### Core Components

- **ModelSpecific** (`Sources/NaturalLanguageEmbeddings/ModelSpecific.swift`): Enum for specifying which NLContextualEmbedding model to load (by language, script, or model identifier)
- **EmbeddingError** (`Sources/NaturalLanguageEmbeddings/EmbeddingError.swift`): Error types for embedding operations
- **Logger** (`Sources/NaturalLanguageEmbeddings/Logger.swift`): Logging utility

### Accelerate Framework Usage

The `EmbeddingService` uses several Accelerate/vDSP functions for performance:
- `vDSP_vaddD`: Vector addition (mean pooling accumulation)
- `vDSP_vsdivD`: Vector-scalar division (mean pooling and normalization)
- `vDSP_svesqD`: Sum of vector elements squared (L2 norm calculation)
- `vDSP_dotprD`: Dot product (cosine similarity for small datasets)
- `vDSP_mmulD`: Matrix-matrix/vector multiplication (batch similarity for large datasets)

**Important**: Package.swift includes `ACCELERATE_NEW_LAPACK` compiler flag for compatibility.

### API Usage Pattern

Both services follow the same pattern:

```swift
// Initialize service
let service = try await EmbeddingService(specific: .script(.latin))

// Generate embeddings for your documents
var embeddings: [[Double]] = []
for document in documents {
    let embedding = try await service.generateEmbeddings(document)
    embeddings.append(embedding)
}

// Search (returns indices)
let results = try await service.search(query: "your query", in: embeddings)

// Use indices to retrieve original documents
for (index, similarity) in results.prefix(10) {
    print("[\(index)] \(similarity) - \(documents[index])")
}
```

## Testing Philosophy

The test suite emphasizes correctness with clear assertions:

- **ComparisonTests**: Verifies both implementations produce identical results
- **Service-specific tests**: Test normalization, similarity ordering, search ordering, and real-world semantic search scenarios
- Tests use verbose output showing actual vs expected results for debugging
- Quotes dataset (`Tests/NaturalLanguageEmbeddingsTests/Quotes.swift`) provides real-world test data

## Platform Support

Requires minimum platform versions (from Package.swift):
- iOS 17+
- macOS 14+
- tvOS 17+
- watchOS 10+

Uses Swift 6.2 with strict concurrency checking (both services are `actor` types).
