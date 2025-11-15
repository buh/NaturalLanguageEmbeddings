import Foundation
import NaturalLanguage

actor BasicEmbeddingService {
    var isModelAvailable: Bool { model.hasAvailableAssets }
    
    private let model: NLContextualEmbedding

    init(specific: ModelSpecific = .script(.latin)) async throws {
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
        
        self.model = model
    }
}

// MARK: - Encoding

extension BasicEmbeddingService {
    func generateEmbeddings(_ sentence: String, language: NLLanguage? = nil) async throws -> [Double] {
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
        
        // Accumulate token vectors in-place (more efficient than storing all vectors)
        embedding.enumerateTokenVectors(in: sentence.startIndex ..< sentence.endIndex) { (tokenVector, _) -> Bool in
            for i in 0..<dimension {
                meanPooled[i] += tokenVector[i]
            }
            tokenCount += 1
            return true
        }
        
        // Divide by token count to get mean
        guard tokenCount > 0 else {
            throw EmbeddingError.generationFailed
        }
        
        let divisor = Double(tokenCount)
        for i in 0..<dimension {
            meanPooled[i] /= divisor
        }
        
        return normalize(meanPooled)
    }
    
    private func normalize(_ vector: [Double]) -> [Double] {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        return vector.map { $0 / norm }
    }
    
    func search(query: String, in embeddings: [[Double]]) async throws -> [(Int, Double)] {
        let queryEmbedding = try await generateEmbeddings(query)
        var results: [(Int, Double)] = []
        
        for (index, embedding) in embeddings.enumerated() {
            let similarity = cosineSimilarity(queryEmbedding, embedding)
            results.append((index, similarity))
        }
        
        return results.sorted { $0.1 > $1.1 }
    }
}

private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
    precondition(a.count == b.count)
    let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    let magA = sqrt(a.reduce(0) { $0 + $1 * $1 })
    let magB = sqrt(b.reduce(0) { $0 + $1 * $1 })
    return dot / (magA * magB)
}
