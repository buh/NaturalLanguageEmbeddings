# NaturalLanguageEmbeddings

A Swift package for semantic text search using Apple's `NLContextualEmbedding` framework. Provides on-device, privacy-first text embeddings with optimized search algorithms for both small and large datasets.

## Features

- **512-dimensional** embeddings on iOS/tvOS/watchOS
- **768-dimensional** embeddings on macOS
- **100% On-Device** - Zero network calls, complete privacy
- **Zero Bundle Size** - Uses native iOS/macOS frameworks
- **Dual Implementation** - Simple and Accelerate-optimized variants
- **Adaptive Performance** - Automatically switches between algorithms based on dataset size
- **Type-Independent** - Index-based results work with any content type
- **Threshold Filtering** - Built-in relevance filtering
- **Swift 6 Ready** - Full concurrency support with actor isolation

## Installation

### Swift Package Manager

Add `NaturalLanguageEmbeddings` to your project via Xcode or Package.swift:

```swift
dependencies: [
    .package(url: "https://github.com/buh/NaturalLanguageEmbeddings.git", from: "1.0.0")
]
```

## Requirements

- iOS 17.0+ / macOS 14.0+ / tvOS 17.0+ / watchOS 10.0+
- Swift 6.2+
- Xcode 16.0+

## Quick Start

### Basic Usage

```swift
import NaturalLanguageEmbeddings

// 1. Initialize the service
let service = try await EmbeddingService(specific: .script(.latin))

// 2. Your documents (any type you want)
let documents = [
    "Machine learning enables computers to learn from data",
    "Swift is a powerful programming language for iOS development",
    "Natural language processing helps computers understand human language",
    "The weather forecast predicts rain tomorrow"
]

// 3. Generate embeddings
var embeddings: [[Double]] = []
for document in documents {
    let embedding = try await service.generateEmbeddings(document)
    embeddings.append(embedding)
}

// 4. Search with optional threshold
let results = try await service.search(
    query: "artificial intelligence and machine learning",
    in: embeddings,
    minimumSimilarity: 0.85  // Optional: filter low-relevance results
)

// 5. Use indices to retrieve your documents
for (index, similarity) in results.prefix(3) {
    print("[\(index)] \(similarity): \(documents[index])")
}
```

### Notes Search Example

```swift
struct Note {
    let id: UUID
    let title: String
    let text: String
    let date: Date
}

// Chunk long texts for better precision
func chunkText(_ text: String, wordsPerChunk: Int = 250) -> [String] {
    let words = text.split(separator: " ")
    var chunks: [String] = []

    for i in stride(from: 0, to: words.count, by: wordsPerChunk - 50) {
        let end = min(i + wordsPerChunk, words.count)
        let chunk = words[i..<end].joined(separator: " ")
        chunks.append(chunk)
    }

    return chunks
}

// Index your notes
var noteChunks: [(noteID: UUID, chunkIndex: Int, text: String)] = []
var embeddings: [[Double]] = []

for note in notes {
    let chunks = chunkText(note.text)

    for (index, chunk) in chunks.enumerated() {
        let embedding = try await service.generateEmbeddings(chunk)
        noteChunks.append((note.id, index, chunk))
        embeddings.append(embedding)
    }
}

// Search across all notes
let results = try await service.search(
    query: "team meeting about project deadline",
    in: embeddings,
    minimumSimilarity: 0.85
)

// Group by note and show top matches
var noteMatches: [UUID: [(Int, String, Double)]] = [:]
for (resultIndex, similarity) in results {
    let (noteID, chunkIndex, text) = noteChunks[resultIndex]
    noteMatches[noteID, default: []].append((chunkIndex, text, similarity))
}

// Display results sorted by best match per note
for (noteID, chunks) in noteMatches.sorted(by: { $0.value[0].2 > $1.value[0].2 }) {
    let note = notes.first { $0.id == noteID }!
    let bestMatch = chunks.max(by: { $0.2 < $1.2 })!

    print("\(note.title) - Match: \(bestMatch.2)")
    print("  \"\(bestMatch.1.prefix(100))...\"")
}
```

## Design Philosophy: Index-Based Results

A key design decision is that search results return **indices** rather than content:

```swift
let results: [(Int, Double)] = try await service.search(query: "...", in: embeddings)
//                ^     ^
//             index  similarity
```

### Why Indices?

This makes the package **type-independent** and **maximally flexible**:

#### ✅ Works with Any Content Type

```swift
// Works with plain strings
let documents: [String] = [...]

// Works with custom types
struct Article {
    let title: String
    let body: String
    let metadata: Metadata
}
let articles: [Article] = [...]

// Works with chunks referencing parent documents
struct Chunk {
    let documentID: UUID
    let chunkIndex: Int
    let text: String
}
let chunks: [Chunk] = [...]
```

#### ✅ Separation of Concerns

- **Embeddings**: Pure vector representations (independent of content)
- **Your Data**: Store and manage however you want (database, memory, files)
- **Mapping**: You control how indices map to your content

#### ✅ Enables Advanced Patterns

```swift
// Multi-field search: combine title + body embeddings
for article in articles {
    let titleEmbedding = try await service.generateEmbeddings(article.title)
    let bodyEmbedding = try await service.generateEmbeddings(article.body)

    embeddings.append(titleEmbedding)
    embeddings.append(bodyEmbedding)

    // Track which index maps to which article/field
    indexMap.append((article.id, field: .title))
    indexMap.append((article.id, field: .body))
}
```

## API Reference

### EmbeddingService

The production-ready service with Accelerate framework optimizations.

```swift
// Initialize
let service = try await EmbeddingService(specific: .script(.latin))

// Generate embedding for text
let embedding = try await service.generateEmbeddings(
    _ text: String,
    language: NLLanguage? = nil
) -> [Double]

// Search with optional filtering
let results = try await service.search(
    query: String,
    in embeddings: [[Double]],
    minimumSimilarity: Double? = nil  // Optional threshold (0.0 - 1.0)
) -> [(Int, Double)]
```

### Model Specification

```swift
// By script (recommended for multi-language support)
EmbeddingService(specific: .script(.latin))

// By language
EmbeddingService(specific: .language(.english))

// By model identifier
EmbeddingService(specific: .modelIdentifier("com.apple.embedding.model"))
```

## Performance Characteristics

### Adaptive Algorithm Selection

The package automatically chooses the optimal search algorithm:

- **< 100 items**: Simple cosine similarity (vDSP dot product)
- **≥ 100 items**: Optimized matrix-vector multiplication (vDSP mmul)

### Benchmarks

| Dataset Size | Search Time | Algorithm |
|-------------|-------------|-----------|
| 50 items | ~1ms | Simple |
| 100 items | ~2ms | Optimized |
| 500 items | ~8ms | Optimized |
| 1000 items | ~15ms | Optimized |

*Tested on M3 MacBook Air with 768-dimensional embeddings*

## Search Quality & Best Practices

### Evaluation Results

Comprehensive quality testing shows NLContextualEmbedding performs well for most use cases:

- **Precision@1**: 1.00 (100%) - Top result is almost always relevant ✅
- **Mean Reciprocal Rank**: 0.83 (Excellent) - Relevant results in top 2 positions ✅
- **Use Cases**: Excellent for FAQ search, product discovery, document retrieval, voice note search

See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for detailed quality metrics and benchmarks.

### Recommended Similarity Thresholds

Based on comprehensive evaluation testing:

```swift
// High confidence matches (synonyms, near-duplicates)
minimumSimilarity: 0.90

// Relevant results (recommended for most use cases)
minimumSimilarity: 0.85  // ← Recommended default

// Broad search (related topics)
minimumSimilarity: 0.80

// No filtering (return all results sorted by relevance)
minimumSimilarity: nil
```

### Important Considerations

1. **Use Phrases, Not Single Words**
   - ❌ Bad: Single words show high baseline similarity (0.60-0.89 even for unrelated terms)
   - ✅ Good: Full sentences or phrases provide better discrimination

2. **Chunk Long Documents**
   - Recommended: 200-300 words per chunk (~1.5-2 minutes of speech)
   - Use 50-word overlap to avoid cutting ideas mid-thought
   - Better precision: Find exact relevant section in long content

3. **Tune Thresholds for Your Use Case**
   - Start with 0.85 as baseline
   - A/B test different thresholds with real user queries
   - Monitor precision@k metrics

4. **Consider Metadata Filtering**
   ```swift
   // Combine semantic search with metadata filters
   let recentNotes = allNotes.filter { $0.date > lastWeek }
   let results = try await service.search(query: query, in: recentNotes)
   ```

### When NLContextualEmbedding is Perfect

✅ Privacy-critical applications (medical notes, personal journals)
✅ Offline-first apps (no internet required)
✅ Bundle size constraints (zero overhead)
✅ FAQ/knowledge base search
✅ Product catalog search

### When to Consider Alternatives

⚠️ Code documentation search (may need code-aware embeddings)
⚠️ Domain-specific vocabulary (medical, legal) requiring fine-tuning
⚠️ Need for known benchmarks and published metrics

## Examples

### E-commerce Product Search

```swift
struct Product {
    let id: String
    let name: String
    let description: String
}

let products: [Product] = loadProducts()
var embeddings: [[Double]] = []

for product in products {
    // Combine name and description for better search
    let searchText = "\(product.name). \(product.description)"
    let embedding = try await service.generateEmbeddings(searchText)
    embeddings.append(embedding)
}

// Customer searches
let results = try await service.search(
    query: "running shoes for marathon training",
    in: embeddings,
    minimumSimilarity: 0.85
)

for (index, similarity) in results.prefix(10) {
    print("\(products[index].name) - Relevance: \(Int(similarity * 100))%")
}
```

### FAQ Search

```swift
struct FAQ {
    let question: String
    let answer: String
}

let faqs: [FAQ] = loadFAQs()
var embeddings: [[Double]] = []

for faq in faqs {
    let embedding = try await service.generateEmbeddings(faq.question)
    embeddings.append(embedding)
}

// User asks a question
let results = try await service.search(
    query: "How do I reset my password?",
    in: embeddings,
    minimumSimilarity: 0.85
)

if let (index, _) = results.first {
    print("Q: \(faqs[index].question)")
    print("A: \(faqs[index].answer)")
}
```

## Testing

The package includes comprehensive test coverage:

- Unit tests for both implementations
- Threshold optimization tests
- Quality evaluation tests (Precision@K, MRR)
- Real-world scenario tests (notes, e-commerce, FAQs)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built on Apple's NLContextualEmbedding framework and Accelerate vDSP for high-performance vector operations.
