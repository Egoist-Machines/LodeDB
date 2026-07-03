import Foundation

/// A running single-checkpointer over a persisted store's single index.
///
/// It folds the write-ahead log that concurrent `LodeAppender` processes log into
/// fresh committed generations, continuously, without an application re-opening a
/// writable `LodeDB`. One process holds a crash-reclaimable lease and drives
/// `checkpoint()` on a loop or timer; appended records then become durable and
/// visible to a read-only handle's `refresh()` shortly after they are logged. It is
/// the counterpart to the exclusive writer that used to be the only thing that could
/// fold the WAL.
///
/// Unlike a writable `LodeDB`, it does not hold the writer lock for its lifetime: it
/// holds only the lease and takes the exclusive writer lock for the brief window of
/// each fold, so appenders keep logging between folds. The store must hold exactly
/// one index and be operated in WAL commit mode (generation mode keeps no WAL to
/// fold, so `open` rejects it). Like `LodeDB`, a single instance is not thread-safe;
/// serialize calls to it.
///
/// One process at a time holds the lease: a second `open` on the same store waits for
/// the first to close, then fails. A dead holder's lease is reclaimable (the OS
/// releases it on death), so a fresh checkpointer can take over after a crash.
public final class LodeCheckpointer {
    private let native: NativeCheckpointer

    private init(native: NativeCheckpointer) {
        self.native = native
    }

    /// Opens a checkpointer over the store at `path`, acquiring the lease.
    ///
    /// The checkpointer always uses WAL commit mode. `durability` defaults to `.fsync`
    /// (each folded generation is fsynced before returning, matching `LodeDB`); pass
    /// `.buffered` to trade power-loss durability for fold throughput.
    ///
    /// `storeText`/`indexText`/`chunkCharacterLimit` mirror the store's writer exactly
    /// as for a `LodeAppender`: the fold retains a document's text only under
    /// `storeText` and its lexical tokens only under `indexText`, and re-tokenizes at
    /// `chunkCharacterLimit`. Open the checkpointer with the same retention the store's
    /// writer uses, or the fold rewrites the store to the checkpointer's policy
    /// (dropping retained text/tokens on a mismatch). They default to `false`
    /// (privacy), so a store whose writer retains text must pass `storeText: true`.
    public static func open(
        at path: URL,
        durability: Durability = .fsync,
        storeText: Bool = false,
        indexText: Bool = false,
        chunkCharacterLimit: Int = 8192
    ) throws -> LodeCheckpointer {
        let options = LodeStoreOptions(
            durability: durability,
            commitMode: .wal,
            storeText: storeText,
            indexText: indexText,
            chunkCharacterLimit: chunkCharacterLimit,
            // The checkpointer owns the writer lock manually (per fold) via its lease;
            // the core forces the warm engine's lifetime lock off regardless.
            acquireWriterLock: false
        )
        let optionsJSON = try options.coreOpenOptionsJSON(path: path.path, readOnly: false)
        return LodeCheckpointer(native: try NativeCheckpointer.open(optionsJSON: optionsJSON))
    }

    /// Folds the appended WAL tail into a fresh committed generation and returns the
    /// number of records folded (`0` when nothing new was appended).
    ///
    /// Takes the exclusive writer lock only for this fold, so appenders run freely
    /// between calls. Drive it on a loop or timer to keep the store continuously
    /// current. If a concurrent writer advanced the committed base since the last
    /// fold, the warm state is reloaded first, so a fold never targets a stale base.
    @discardableResult
    public func checkpoint() throws -> UInt64 {
        try native.checkpoint()
    }
}
