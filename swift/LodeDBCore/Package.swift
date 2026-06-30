// swift-tools-version: 5.9

import PackageDescription
import Foundation

// The native core ships as a prebuilt XCFramework (static `liblodedb_ffi` slices
// for macOS, iOS device, and iOS simulator, plus the C header as a Clang module).
//
// By default the package links the framework from a local Artifacts/ directory,
// produced by `scripts/package_xcframework.sh`. A fresh checkout must run that
// script once before building (the script ends by invoking `swift build`).
//
// For tagged releases, set LODEDB_FFI_BINARY_URL + LODEDB_FFI_BINARY_CHECKSUM to
// point at the hosted XCFramework zip (the value `swift package compute-checksum`
// prints for the release asset); the manifest then resolves the remote artifact
// instead of the local path. This is what consumers depending on a release tag get.
let nativeCore: Target = {
    let env = ProcessInfo.processInfo.environment
    if let urlString = env["LODEDB_FFI_BINARY_URL"],
       let checksum = env["LODEDB_FFI_BINARY_CHECKSUM"],
       !urlString.isEmpty,
       !checksum.isEmpty {
        return .binaryTarget(name: "LodeDBCoreFFI", url: urlString, checksum: checksum)
    }
    return .binaryTarget(name: "LodeDBCoreFFI", path: "Artifacts/LodeDBCoreFFI.xcframework")
}()

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
        nativeCore,
        .target(
            name: "LodeDBCore",
            dependencies: ["LodeDBCoreFFI"],
            linkerSettings: [
                // The static core calls Accelerate (BLAS) for the rotation-matrix build.
                .linkedFramework("Accelerate")
            ]
        ),
        .testTarget(name: "LodeDBCoreTests", dependencies: ["LodeDBCore"])
    ]
)
