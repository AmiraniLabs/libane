// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ANE",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "ANE", targets: ["ANE"]),
    ],
    targets: [
        // C library target (wraps the static libane)
        .systemLibrary(
            name: "CLibane",
            path: "Sources/CLibane",
            pkgConfig: "ane",
            providers: [.brew(["amiranilabs/tap/libane"])]
        ),

        // Swift wrapper
        .target(
            name: "ANE",
            dependencies: ["CLibane"],
            path: "Sources/ANE",
            swiftSettings: [
                .unsafeFlags(["-enable-experimental-feature", "StrictConcurrency"],
                             .when(configuration: .debug))
            ]
        ),

        // Tests
        .testTarget(
            name: "ANETests",
            dependencies: ["ANE"],
            path: "Tests/ANETests"
        ),
    ]
)
