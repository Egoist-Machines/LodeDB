import Foundation
import LodeDBCoreFFI

/// The C ABI version this binding is built against. Checked at engine creation so a
/// mismatched XCFramework fails loudly instead of corrupting memory.
let lodeNativeExpectedABIVersion: UInt32 = 1

/// Owning wrapper around a native `LodeEngine *`, statically linked from the
/// `LodeDBCoreFFI` XCFramework (no `dlopen`). Not thread-safe for concurrent
/// writers; callers serialize access (see `LodeDB`).
final class NativeEngine {
    private let handle: OpaquePointer
    let indexID: String

    static func abiVersion() -> UInt32 { lodedb_abi_version() }

    init(indexID: String = "default", vectorDimension: Int, bitWidth: Int = 4) throws {
        let abi = lodedb_abi_version()
        guard abi == lodeNativeExpectedABIVersion else {
            throw LodeDBError.corruptStore(
                "native core ABI \(abi) does not match expected \(lodeNativeExpectedABIVersion)")
        }
        var engine: OpaquePointer?
        var error: UnsafeMutablePointer<LodeError>?
        try NativeEngine.check(lodedb_engine_new_in_memory(&engine, &error), error: error)
        guard let engine else {
            throw LodeDBError.internalError("native core did not return an engine")
        }
        self.handle = engine
        self.indexID = indexID

        var createError: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            lodedb_engine_create_index(handle, indexView, UInt(vectorDimension), UInt(bitWidth), &createError)
        }
        do {
            try NativeEngine.check(status, error: createError)
        } catch {
            lodedb_engine_free(handle)
            throw error
        }
    }

    deinit {
        lodedb_engine_free(handle)
    }

    func upsertVectorsJSON(_ documentsJSON: String) throws {
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            withStringView(documentsJSON) { documentsView in
                lodedb_engine_upsert_vectors_json(handle, indexView, documentsView, &error)
            }
        }
        try NativeEngine.check(status, error: error)
    }

    func queryVectorJSON(_ vector: [Float], k: Int, filterJSON: String?) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            vector.withUnsafeBufferPointer { queryBuffer in
                withStringView(filterJSON ?? "") { filterView in
                    lodedb_engine_query_vector_json(
                        handle,
                        indexView,
                        queryBuffer.baseAddress,
                        UInt(vector.count),
                        UInt(k),
                        filterView,
                        filterJSON == nil ? 0 : 1,
                        &out,
                        &error
                    )
                }
            }
        }
        try NativeEngine.check(status, error: error)
        return try NativeEngine.copyOwnedString(out)
    }

    func prepareTextUpsertJSON(
        _ documentsJSON: String,
        storeText: Bool,
        indexText: Bool,
        chunkCharacterLimit: Int
    ) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            withStringView(documentsJSON) { documentsView in
                lodedb_engine_prepare_text_upsert_json(
                    handle,
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
        try NativeEngine.check(status, error: error)
        return try NativeEngine.copyOwnedString(out)
    }

    func applyTextUpsertJSON(
        planJSON: String,
        embeddingsJSON: String,
        embeddingTimeMS: Double
    ) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(planJSON) { planView in
            withStringView(embeddingsJSON) { embeddingsView in
                lodedb_engine_apply_text_upsert_json(handle, planView, embeddingsView, embeddingTimeMS, &out, &error)
            }
        }
        try NativeEngine.check(status, error: error)
        return try NativeEngine.copyOwnedString(out)
    }

    func prepareQueryTextJSON(_ query: String, mode: String) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(query) { queryView in
            withStringView(mode) { modeView in
                lodedb_engine_prepare_query_text_json(handle, queryView, modeView, &out, &error)
            }
        }
        try NativeEngine.check(status, error: error)
        return try NativeEngine.copyOwnedString(out)
    }

    func searchEmbeddedTextJSON(
        queryPlanJSON: String,
        queryEmbeddingJSON: String?,
        k: Int,
        filterJSON: String?
    ) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            withStringView(queryPlanJSON) { queryPlanView in
                withStringView(queryEmbeddingJSON ?? "") { embeddingView in
                    withStringView(filterJSON ?? "") { filterView in
                        lodedb_engine_search_embedded_text_json(
                            handle,
                            indexView,
                            queryPlanView,
                            embeddingView,
                            queryEmbeddingJSON == nil ? 0 : 1,
                            UInt(k),
                            filterView,
                            filterJSON == nil ? 0 : 1,
                            &out,
                            &error
                        )
                    }
                }
            }
        }
        try NativeEngine.check(status, error: error)
        return try NativeEngine.copyOwnedString(out)
    }

    private static func check(_ status: UInt32, error: UnsafeMutablePointer<LodeError>?) throws {
        guard status != 0 else { return }
        defer { lodedb_error_free(error) }
        let message = error?.pointee.message.map { String(cString: $0) } ?? "native core call failed"
        switch status {
        case 1: throw LodeDBError.invalidArgument(message)
        case 2: throw LodeDBError.notFound(message)
        case 3: throw LodeDBError.corruptStore(message)
        case 4: throw LodeDBError.planStale(message)
        case 5: throw LodeDBError.unsupported(message)
        default: throw LodeDBError.internalError(message)
        }
    }

    private static func copyOwnedString(_ out: UnsafeMutablePointer<LodeOwnedString>?) throws -> String {
        guard let out else {
            throw LodeDBError.internalError("native core did not return JSON")
        }
        defer { lodedb_owned_string_free(out) }
        let owned = out.pointee
        guard let data = owned.data else {
            if owned.len == 0 {
                return ""
            }
            throw LodeDBError.internalError("native core returned null string data")
        }
        let bytes = Data(bytes: data, count: Int(owned.len))
        guard let text = String(data: bytes, encoding: .utf8) else {
            throw LodeDBError.internalError("native core returned invalid UTF-8")
        }
        return text
    }
}

func withStringView<T>(_ string: String, _ body: (LodeStringView) throws -> T) rethrows -> T {
    try string.withCString { pointer in
        let view = LodeStringView(
            size: UInt32(MemoryLayout<LodeStringView>.size),
            version: lodeNativeExpectedABIVersion,
            data: pointer,
            len: UInt(string.utf8.count)
        )
        return try body(view)
    }
}
