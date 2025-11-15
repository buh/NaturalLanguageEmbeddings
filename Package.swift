// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "NaturalLanguageEmbeddings",
    platforms: [.iOS(.v17), .macOS(.v14), .tvOS(.v17), .watchOS(.v10)],
    products: [
        .library(name: "NaturalLanguageEmbeddings", targets: ["NaturalLanguageEmbeddings"]),
    ],
    targets: [
        .target(
            name: "NaturalLanguageEmbeddings",
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK")
            ]
        ),
        .testTarget(
            name: "NaturalLanguageEmbeddingsTests",
            dependencies: ["NaturalLanguageEmbeddings"]
        ),
    ]
)
