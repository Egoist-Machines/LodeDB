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
3. `swift-package-publish` runs `swift/LodeDBCore/scripts/release_swift_package.sh`.
   The script assembles the package through `publish_swift_package.sh`, force-pushes a
   `release/vX.Y.Z` branch, opens or reuses a pull request, squash-merges it to `main`,
   and tags the merged commit as `vX.Y.Z`. The pull request is required by the mirror's
   `main` branch ruleset; tag refs are not covered. Reruns reuse matching content, pull
   requests, and tags. For an empty mirror, the script pushes the initial `main` directly
   because there is no base branch for a pull request.

The tests stay in this repo (run by the `swift-binding` CI job); the published package
ships only the library.

## One-time setup

`swift-package-publish` skips cleanly until both of these exist, so it never blocks a
release:

1. Create the package repo `Egoist-Machines/swift-lodedb` (it can start empty; the
   first release populates it).
2. Add a repository secret `SWIFT_PACKAGE_DEPLOY_TOKEN` to the LodeDB repo: a token with
   `contents: write` and pull-request read/write access to `swift-lodedb`, such as a
   fine-grained PAT scoped to that repo. The job uses it for cross-repo Git and GitHub
   CLI operations.

To target a different repo name, change `PACKAGE_REPO` in the `swift-package-publish`
job and the URLs in `scripts/publish_swift_package.sh`.

## Local / pre-release consumption

To build against the binding without a published release (development, or pinning a
specific build), build the XCFramework locally and use the in-repo package, or point a
`binaryTarget` at any hosted zip via `LODEDB_FFI_BINARY_URL` + `LODEDB_FFI_BINARY_CHECKSUM`.
See [swift/LodeDBCore/README.md](../swift/LodeDBCore/README.md).
