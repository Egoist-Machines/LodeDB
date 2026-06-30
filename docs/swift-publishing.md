# Publishing the Swift package

The Swift binding is distributed as a standalone SwiftPM package at
`Egoist-Machines/swift-lodedb`: a thin repo holding the Swift sources and a
`Package.swift` whose binary target is pinned to a released
`LodeDBCoreFFI.xcframework` (URL + checksum). Keeping it separate from this Python/Rust
monorepo gives consumers a clean, root-level SwiftPM package without cloning the whole
repository.

## Consuming a release

```swift
dependencies: [
    .package(url: "https://github.com/Egoist-Machines/swift-lodedb", from: "1.1.0"),
],
targets: [
    .target(name: "App", dependencies: [
        .product(name: "LodeDBCore", package: "swift-lodedb"),
    ]),
]
```

Each `swift-lodedb` version `vX.Y.Z` corresponds to LodeDB release `vX.Y.Z` and resolves
the matching `LodeDBCoreFFI.xcframework` from this repo's GitHub Release (macOS arm64,
iOS device arm64, iOS simulator arm64).

## How a release publishes it

On a `vX.Y.Z` tag, `release.yml`:

1. `swift-xcframework` builds the three-slice `LodeDBCoreFFI.xcframework`, runs
   `swift package compute-checksum`, and uploads the zip + checksum.
2. `publish` attaches `LodeDBCoreFFI.xcframework.zip` (and its `.checksum`) to the
   GitHub Release, giving it a stable public URL:
   `https://github.com/Egoist-Machines/LodeDB/releases/download/vX.Y.Z/LodeDBCoreFFI.xcframework.zip`.
3. `swift-package-publish` runs `swift/LodeDBCore/scripts/publish_swift_package.sh` to
   assemble the package (copy `swift/LodeDBCore/Sources/`, generate a `Package.swift`
   with that release's binary-target URL + checksum), then commits, tags `vX.Y.Z`, and
   pushes it to `swift-lodedb`.

The tests stay in this repo (run by the `swift-binding` CI job); the published package
ships only the library.

## One-time setup

`swift-package-publish` skips cleanly until both of these exist, so it never blocks a
release:

1. Create the package repo `Egoist-Machines/swift-lodedb` (it can start empty; the
   first release populates it).
2. Add a repository secret `SWIFT_PACKAGE_DEPLOY_TOKEN` to the LodeDB repo: a token with
   push (`contents: write`) access to `swift-lodedb` (a fine-grained PAT scoped to that
   repo, or a deploy key). The job authenticates the cross-repo push with it.

To target a different repo name, change `PACKAGE_REPO` in the `swift-package-publish`
job and the URLs in `scripts/publish_swift_package.sh`.

## Local / pre-release consumption

To build against the binding without a published release (development, or pinning a
specific build), build the XCFramework locally and use the in-repo package, or point a
`binaryTarget` at any hosted zip via `LODEDB_FFI_BINARY_URL` + `LODEDB_FFI_BINARY_CHECKSUM`.
See [swift/LodeDBCore/README.md](../swift/LodeDBCore/README.md).
