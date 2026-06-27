import Foundation
import CLodeDBCoreBridge

#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

final class NativeCoreLibrary {
    private typealias ABIVersionFn = @convention(c) () -> UInt32
    private typealias EngineNewInMemoryFn = @convention(c) (
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
        UnsafeMutablePointer<UnsafeMutablePointer<LodeError>?>?
    ) -> UInt32
    private typealias EngineFreeFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias ErrorFreeFn = @convention(c) (UnsafeMutablePointer<LodeError>?) -> Void
    private typealias OwnedStringFreeFn = @convention(c) (UnsafeMutablePointer<LodeOwnedString>?) -> Void
    private typealias EngineCreateIndexFn = @convention(c) (
        UnsafeMutableRawPointer?,
        LodeStringView,
        UInt,
        UInt,
        UnsafeMutablePointer<UnsafeMutablePointer<LodeError>?>?
    ) -> UInt32
    private typealias PrepareTextUpsertJSONFn = @convention(c) (
        UnsafeMutableRawPointer?,
        LodeStringView,
        LodeStringView,
        UInt8,
        UInt8,
        UInt,
        UnsafeMutablePointer<UnsafeMutablePointer<LodeOwnedString>?>?,
        UnsafeMutablePointer<UnsafeMutablePointer<LodeError>?>?
    ) -> UInt32
    private typealias ApplyTextUpsertJSONFn = @convention(c) (
        UnsafeMutableRawPointer?,
        LodeStringView,
        LodeStringView,
        Double,
        UnsafeMutablePointer<UnsafeMutablePointer<LodeOwnedString>?>?,
        UnsafeMutablePointer<UnsafeMutablePointer<LodeError>?>?
    ) -> UInt32

    private let handle: UnsafeMutableRawPointer
    private let abiVersionFn: ABIVersionFn
    private let engineNewInMemoryFn: EngineNewInMemoryFn
    private let engineFreeFn: EngineFreeFn
    private let errorFreeFn: ErrorFreeFn
    private let ownedStringFreeFn: OwnedStringFreeFn
    private let engineCreateIndexFn: EngineCreateIndexFn
    private let prepareTextUpsertJSONFn: PrepareTextUpsertJSONFn
    private let applyTextUpsertJSONFn: ApplyTextUpsertJSONFn

    init(path: String) throws {
        guard let opened = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            throw LodeDBError.invalidArgument("failed to load native core: \(Self.dynamicLoaderError())")
        }
        handle = opened
        do {
            abiVersionFn = try Self.load("lodedb_abi_version", from: opened)
            engineNewInMemoryFn = try Self.load("lodedb_engine_new_in_memory", from: opened)
            engineFreeFn = try Self.load("lodedb_engine_free", from: opened)
            errorFreeFn = try Self.load("lodedb_error_free", from: opened)
            ownedStringFreeFn = try Self.load("lodedb_owned_string_free", from: opened)
            engineCreateIndexFn = try Self.load("lodedb_engine_create_index", from: opened)
            prepareTextUpsertJSONFn = try Self.load("lodedb_engine_prepare_text_upsert_json", from: opened)
            applyTextUpsertJSONFn = try Self.load("lodedb_engine_apply_text_upsert_json", from: opened)
        } catch {
            dlclose(opened)
            throw error
        }
    }

    deinit {
        dlclose(handle)
    }

    func abiVersion() -> UInt32 {
        abiVersionFn()
    }

    func newInMemoryEngine() throws -> UnsafeMutableRawPointer {
        var engine: UnsafeMutableRawPointer?
        var error: UnsafeMutablePointer<LodeError>?
        let status = engineNewInMemoryFn(&engine, &error)
        try check(status, error: error)
        guard let engine else {
            throw LodeDBError.invalidArgument("native core did not return an engine")
        }
        return engine
    }

    func freeEngine(_ engine: UnsafeMutableRawPointer?) {
        engineFreeFn(engine)
    }

    func createIndex(engine: UnsafeMutableRawPointer, indexID: String, vectorDimension: Int) throws {
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            engineCreateIndexFn(engine, indexView, UInt(vectorDimension), 4, &error)
        }
        try check(status, error: error)
    }

    func prepareTextUpsertJSON(
        engine: UnsafeMutableRawPointer,
        indexID: String,
        documentsJSON: String,
        storeText: Bool,
        indexText: Bool,
        chunkCharacterLimit: Int
    ) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            withStringView(documentsJSON) { documentsView in
                prepareTextUpsertJSONFn(
                    engine,
                    indexView,
                    documentsView,
                    storeText ? 1 : 0,
                    indexText ? 1 : 0,
                    UInt(chunkCharacterLimit),
                    &out,
                    &error
                )
            }
        }
        try check(status, error: error)
        return try copyOwnedString(out)
    }

    func applyTextUpsertJSON(
        engine: UnsafeMutableRawPointer,
        planJSON: String,
        embeddingsJSON: String,
        embeddingTimeMS: Double
    ) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(planJSON) { planView in
            withStringView(embeddingsJSON) { embeddingsView in
                applyTextUpsertJSONFn(engine, planView, embeddingsView, embeddingTimeMS, &out, &error)
            }
        }
        try check(status, error: error)
        return try copyOwnedString(out)
    }

    private func check(_ status: UInt32, error: UnsafeMutablePointer<LodeError>?) throws {
        guard status != 0 else {
            return
        }
        defer { errorFreeFn(error) }
        let message = error?.pointee.message.map { String(cString: $0) } ?? "native core call failed"
        switch status {
        case 2:
            throw LodeDBError.notFound(message)
        case 3:
            throw LodeDBError.corruptStore(message)
        default:
            throw LodeDBError.invalidArgument(message)
        }
    }

    private func copyOwnedString(_ out: UnsafeMutablePointer<LodeOwnedString>?) throws -> String {
        guard let out else {
            throw LodeDBError.invalidArgument("native core did not return JSON")
        }
        defer { ownedStringFreeFn(out) }
        let owned = out.pointee
        guard let data = owned.data else {
            if owned.len == 0 {
                return ""
            }
            throw LodeDBError.invalidArgument("native core returned null string data")
        }
        let bytes = Data(bytes: data, count: Int(owned.len))
        guard let text = String(data: bytes, encoding: .utf8) else {
            throw LodeDBError.invalidArgument("native core returned invalid UTF-8")
        }
        return text
    }

    private static func load<T>(_ name: String, from handle: UnsafeMutableRawPointer) throws -> T {
        guard let symbol = dlsym(handle, name) else {
            throw LodeDBError.invalidArgument("missing native core symbol \(name): \(dynamicLoaderError())")
        }
        return unsafeBitCast(symbol, to: T.self)
    }

    private static func dynamicLoaderError() -> String {
        guard let message = dlerror() else {
            return "unknown dynamic loader error"
        }
        return String(cString: message)
    }
}

final class NativeTextCore {
    private let library: NativeCoreLibrary
    private let engine: UnsafeMutableRawPointer
    private let indexID: String

    init(library: NativeCoreLibrary, indexID: String = "default", vectorDimension: Int) throws {
        self.library = library
        self.indexID = indexID
        engine = try library.newInMemoryEngine()
        try library.createIndex(engine: engine, indexID: indexID, vectorDimension: vectorDimension)
    }

    deinit {
        library.freeEngine(engine)
    }

    func prepareTextUpsertJSON(
        _ documentsJSON: String,
        storeText: Bool,
        indexText: Bool,
        chunkCharacterLimit: Int
    ) throws -> String {
        try library.prepareTextUpsertJSON(
            engine: engine,
            indexID: indexID,
            documentsJSON: documentsJSON,
            storeText: storeText,
            indexText: indexText,
            chunkCharacterLimit: chunkCharacterLimit
        )
    }

    func applyTextUpsertJSON(
        planJSON: String,
        embeddingsJSON: String,
        embeddingTimeMS: Double
    ) throws -> String {
        try library.applyTextUpsertJSON(
            engine: engine,
            planJSON: planJSON,
            embeddingsJSON: embeddingsJSON,
            embeddingTimeMS: embeddingTimeMS
        )
    }
}

private func withStringView<T>(_ string: String, _ body: (LodeStringView) throws -> T) rethrows -> T {
    try string.withCString { pointer in
        let view = LodeStringView(
            size: UInt32(MemoryLayout<LodeStringView>.size),
            version: 1,
            data: pointer,
            len: UInt(string.utf8.count)
        )
        return try body(view)
    }
}
