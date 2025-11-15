import Foundation

enum EmbeddingError: Error {
    case modelUnavailable
    case missingEmbeddingResource
    case generationFailed
    case unsupportedNormalization
}
