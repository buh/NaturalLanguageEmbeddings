import NaturalLanguage
import CoreML
import Accelerate

/// Service for generating and searching embeddings using NLContextualEmbedding models.
/// Supports mean pooling and L2 normalization, as well as optimized search using vDSP.
public actor EmbeddingService {
    /// Indicates if the embedding model has its assets available.
    public var isModelAvailable: Bool { model.hasAvailableAssets }
    
    private let model: NLContextualEmbedding
    
    /// Threshold for switching between simple and optimized search algorithms.
    /// Datasets with fewer items use simple search, larger datasets use vDSP-optimized search.
    /// Internal for testing purposes - allows verification that both algorithms produce identical results.
    private let optimizationThreshold: Int
    
    /// Initializes the EmbeddingService with a specified model.
    /// - Parameter specific: The model specification to use.
    public init(specific: ModelSpecific = .script(.latin)) async throws {
        try await self.init(specific: specific, optimizationThreshold: 100)
    }
    
    /// Internal initializer with configurable optimization threshold for testing.
    /// - Parameters:
    ///   - specific: The model specification to use.
    ///   - optimizationThreshold: Threshold for using optimized search (internal, for testing).
    init(specific: ModelSpecific = .script(.latin), optimizationThreshold: Int) async throws {
        let model: NLContextualEmbedding?
        switch specific {
        case .language(let language):
            model = NLContextualEmbedding(language: language)
        case .script(let script):
            model = NLContextualEmbedding(script: script)
        case .modelIdentifier(let identifier):
            model = NLContextualEmbedding(modelIdentifier: identifier)
        }
        
        guard let model else {
            throw EmbeddingError.modelUnavailable
        }
        
        if !model.hasAvailableAssets {
            logger.log("Requesting assets for model: \(model.modelIdentifier, privacy: .public)")
            try await model.requestAssets()
            logger.log("Assets available for model: \(model.modelIdentifier, privacy: .public)")
        }
        
        self.model = model
        self.optimizationThreshold = optimizationThreshold
    }
    
    /// Provides information about the loaded embedding model.
    public func modelInfo() -> String {
        """
Model Identifier: \(model.modelIdentifier)
Dimension: \(model.dimension)
Available Assets: \(model.hasAvailableAssets)
Languages: \(model.languages.map { $0.rawValue })
Scripts: \(model.scripts.map { $0.rawValue })
"""
    }
}

// MARK: - Encoding

extension EmbeddingService {
    /// Generates normalized embeddings for a sentence (using mean pooling and L2 normalization)
    public func generateEmbeddings(_ sentence: String, language: NLLanguage? = nil) async throws -> [Double] {
        guard !sentence.isEmpty else {
            throw EmbeddingError.generationFailed
        }
        
        guard model.hasAvailableAssets else {
            throw EmbeddingError.missingEmbeddingResource
        }
        
        guard model.dimension > 0 else {
            throw EmbeddingError.generationFailed
        }
        
        let embedding = try model.embeddingResult(for: sentence, language: language)
        let dimension = model.dimension
        
        // Initialize accumulator for mean pooling
        var meanPooled = [Double](repeating: 0.0, count: dimension)
        var tokenCount = 0
        
        // Accumulate token vectors in-place using vDSP (more efficient than storing all vectors)
        embedding.enumerateTokenVectors(in: sentence.startIndex ..< sentence.endIndex) { (tokenVector, _) -> Bool in
            vDSP_vaddD(meanPooled, 1, tokenVector, 1, &meanPooled, 1, vDSP_Length(dimension))
            tokenCount += 1
            return true
        }
        
        // Divide by token count to get mean
        guard tokenCount > 0 else {
            throw EmbeddingError.generationFailed
        }
        
        var divisor = Double(tokenCount)
        vDSP_vsdivD(meanPooled, 1, &divisor, &meanPooled, 1, vDSP_Length(dimension))
        
        return normalize(meanPooled)
    }
    
    private func normalize(_ embeddings: [Double]) -> [Double] {
        // Compute L2 norm efficiently using vDSP
        var sumSquares: Double = 0
        let length = vDSP_Length(embeddings.count)
        vDSP_svesqD(embeddings, 1, &sumSquares, length)
        let norm = sqrt(sumSquares)
        
        guard norm > 1e-10 else {
            // If norm is too small, return as-is to avoid division by zero
            return embeddings
        }
        
        // Divide by norm to normalize
        var divisor = norm
        var normalized = [Double](repeating: 0, count: embeddings.count)
        vDSP_vsdivD(embeddings, 1, &divisor, &normalized, 1, length)
        
        return normalized
    }
}

// MARK: - Search

extension EmbeddingService {
    /// Searches for the most similar embeddings to a query string
    /// - Parameters:
    ///   - query: The query string to search for
    ///   - embeddings: Array of pre-computed embeddings to search through
    ///   - minimumSimilarity: Optional threshold to filter results. Only results with similarity >= this value are returned.
    ///                        Recommended: 0.85 for relevant results, 0.90 for high confidence matches.
    /// - Returns: Array of (index, similarity) tuples sorted by similarity (descending)
    public func search(
        query: String,
        in embeddings: [[Double]],
        minimumSimilarity: Double? = nil
    ) async throws -> [(Int, Double)] {
        guard !embeddings.isEmpty else {
            return []
        }
        
        let queryEmbedding = try await generateEmbeddings(query)
        
        guard queryEmbedding.count == embeddings.first!.count else {
            throw EmbeddingError.unsupportedNormalization
        }
        
        let dimension = queryEmbedding.count
        let count = embeddings.count
        
        // For larger datasets (>= optimizationThreshold), use vDSP matrix-vector multiplication for better performance
        return if count < optimizationThreshold {
            searchSimple(queryEmbedding: queryEmbedding, embeddings: embeddings, threshold: minimumSimilarity)
        } else {
            searchOptimized(
                queryEmbedding: queryEmbedding,
                embeddings: embeddings,
                dimension: dimension,
                count: count,
                threshold: minimumSimilarity
            )
        }
    }
}

// MARK: - Cosine Search Implementations

private extension EmbeddingService {
    /// Simple search using basic cosine similarity (good for small datasets)
    func searchSimple(queryEmbedding: [Double], embeddings: [[Double]], threshold: Double?) -> [(Int, Double)] {
        var results: [(Int, Double)] = []
        
        for (index, embedding) in embeddings.enumerated() {
            let similarity = cosineSimilarity(queryEmbedding, embedding)
            
            // Filter during collection if threshold is specified
            if let minSim = threshold {
                if similarity >= minSim {
                    results.append((index, similarity))
                }
            } else {
                results.append((index, similarity))
            }
        }
        
        return results.sorted { $0.1 > $1.1 }
    }
    
    /// Computes cosine similarity between two L2-normalized vectors.
    /// Since generateEmbeddings always returns unit vectors, dot product equals cosine similarity.
    func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        precondition(a.count == b.count, "Vectors must have same dimensions")
        var dotProduct: Double = 0
        vDSP_dotprD(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        return min(max(dotProduct, -1.0), 1.0)
    }
}

// MARK: - Optimized Search

private extension EmbeddingService {
    /// Optimized search using vDSP matrix-vector multiplication (good for large datasets)
    func searchOptimized(queryEmbedding: [Double], embeddings: [[Double]], dimension: Int, count: Int, threshold: Double?) -> [(Int, Double)] {
        // Build matrix of embeddings (row-major: each row is an embedding)
        var matrix = [Double]()
        matrix.reserveCapacity(count * dimension)
        
        for embedding in embeddings {
            matrix.append(contentsOf: embedding)
        }
        
        // Compute all similarities at once using vDSP matrix-vector multiplication
        // Since embeddings are normalized, dot product = cosine similarity
        var similarities = [Double](repeating: 0, count: count)
        
        // Matrix-vector multiplication: similarities = matrix × queryEmbedding
        // A is count×dimension (embeddings), B is dimension×1 (query), C is count×1 (result)
        vDSP_mmulD(
            matrix,                      // Matrix A (count × dimension)
            1,                           // Stride for A
            queryEmbedding,              // Vector B (dimension × 1)
            1,                           // Stride for B
            &similarities,               // Result C (count × 1)
            1,                           // Stride for C
            vDSP_Length(count),          // M: number of rows in A
            vDSP_Length(1),              // N: number of columns in B (1 for vector)
            vDSP_Length(dimension)       // P: number of columns in A / rows in B
        )
        
        // Pair indices with similarities, filtering if threshold is specified
        var results: [(Int, Double)] = []
        for (index, similarity) in similarities.enumerated() {
            // Filter during collection if threshold is specified
            if let minSim = threshold {
                if similarity >= minSim {
                    results.append((index, similarity))
                }
            } else {
                results.append((index, similarity))
            }
        }
        
        return results.sorted { $0.1 > $1.1 }
    }
}
