// swift-tools-version: 5.9

import PackageDescription
import Foundation

// The native core ships as a prebuilt XCFramework (static `liblodedb_ffi` slices
// for macOS arm64, iOS device arm64, and iOS simulator arm64, plus the C header as a
// Clang module). Apple platforms are arm64-only here; an Intel macOS host or x86_64
// simulator would need extra slices added to scripts/package_xcframework.sh.
//
// Resolution:
//   - Default (no env vars): the local `Artifacts/LodeDBCoreFFI.xcframework`, built by
//     `scripts/package_xcframework.sh`. This is the in-repo dev/CI path and the only
//     supported way to build from a checkout; run that script once first (it ends by
//     invoking `swift build`).
//   - When LODEDB_FFI_BINARY_URL + LODEDB_FFI_BINARY_CHECKSUM are set, the manifest
//     resolves that hosted XCFramework zip instead. The release workflow uploads the
//     zip and records its `swift package compute-checksum` value, so consuming a
//     released build via SwiftPM means setting those two vars (or pinning a manifest
//     that hard-codes them). Turnkey tag consumption with no env vars is not wired
//     yet; a fresh external checkout that sets neither will not find the local path.
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
