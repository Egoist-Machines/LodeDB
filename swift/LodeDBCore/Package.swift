// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "LodeDBCore",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(name: "LodeDBCore", targets: ["LodeDBCore"])
    ],
    targets: [
        .target(name: "CLodeDBCoreBridge", publicHeadersPath: "include"),
        .target(name: "LodeDBCore", dependencies: ["CLodeDBCoreBridge"]),
        .testTarget(name: "LodeDBCoreTests", dependencies: ["LodeDBCore"])
    ]
)
