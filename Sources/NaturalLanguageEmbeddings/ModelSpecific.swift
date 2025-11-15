import NaturalLanguage

/// Model specificity for initializing NLContextualEmbedding models.
public enum ModelSpecific: Sendable {
    case language(NLLanguage)
    case script(NLScript)
    case modelIdentifier(String)
}
