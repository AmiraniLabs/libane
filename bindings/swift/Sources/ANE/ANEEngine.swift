/**
 * ANEEngine — Swift wrapper for libane.
 *
 * Primary audience: iOS/macOS app developers who want ANE without CoreML.
 * ObjC++ core is directly bridgeable via CLibane module.
 *
 * Usage:
 *   import ANE
 *
 *   let engine = ANEEngine()
 *   guard engine.isAvailable else { / * fallback * / }
 *
 *   // fp16 matmul
 *   let result = try engine.matmul(a: tensorA, b: tensorB, M: 512, K: 512, N: 512)
 *
 *   // fp32 matmul
 *   let result = try engine.matmulF32(a: arrayA, b: arrayB, M: 64, K: 64, N: 64)
 */
import Foundation
import CLibane

// MARK: - Error types

/// Errors thrown by ANEEngine operations.
public enum ANEError: LocalizedError {
    case unavailable(String)
    case invalidShape(String)
    case compileFailed(String)
    case executeFailed(String)
    case invalidArgument(String)

    public var errorDescription: String? {
        switch self {
        case .unavailable(let s):    return "ANE unavailable: \(s)"
        case .invalidShape(let s):   return "Invalid shape: \(s)"
        case .compileFailed(let s):  return "Compile failed: \(s)"
        case .executeFailed(let s):  return "Execute failed: \(s)"
        case .invalidArgument(let s):return "Invalid argument: \(s)"
        }
    }

    static func from(status: libane_status_t, fallback: String = "") -> ANEError {
        let last = String(cString: libane_last_error())
        let msg  = last.isEmpty ? fallback : last
        switch status {
        case LIBANE_ERR_UNAVAILABLE:    return .unavailable(msg)
        case LIBANE_ERR_INVALID_SHAPE:  return .invalidShape(msg)
        case LIBANE_ERR_COMPILE_FAILED: return .compileFailed(msg)
        case LIBANE_ERR_EXECUTE_FAILED: return .executeFailed(msg)
        case LIBANE_ERR_INVALID_ARG:    return .invalidArgument(msg)
        default:                         return .executeFailed(msg)
        }
    }
}

// MARK: - Log level

public enum ANELogLevel: Int32 {
    case silent = 0
    case error  = 1
    case warn   = 2
    case info   = 3
    case debug  = 4
}

// MARK: - CompiledProgram

/// An opaque compiled ANE program handle.
/// Must be released by calling `engine.release(_:)` or going out of scope.
public final class ANEProgram {
    let handle: libane_handle_t

    init(_ h: libane_handle_t) {
        self.handle = h
    }

    deinit {
        libane_release(handle)
    }
}

// MARK: - ANEEngine

/// Main entry point for libane operations.
///
/// Thread-safe: all public methods may be called from any thread.
/// The compile cache is shared across instances.
public final class ANEEngine: @unchecked Sendable {

    // MARK: - Properties

    /// True if the Apple Neural Engine is accessible on this device.
    public var isAvailable: Bool {
        libane_available() != 0
    }

    /// libane version string.
    public var version: String {
        String(cString: libane_version())
    }

    /// Last error string from the ANE runtime.
    public var lastError: String {
        String(cString: libane_last_error())
    }

    // MARK: - Init

    public init(logLevel: ANELogLevel = .error) {
        libane_set_log_level(libane_log_level_t(rawValue: logLevel.rawValue))
    }

    // MARK: - Backend

    /// Force a specific backend.
    /// - Parameter backend: `.ane` to require ANE, `.cpu` to force CPU, `nil` for auto.
    public enum Backend {
        case ane, cpu
    }

    public func setBackend(_ backend: Backend?) {
        switch backend {
        case .ane:  libane_set_backend("ane")
        case .cpu:  libane_set_backend("cpu")
        case nil:   libane_set_backend(nil)
        }
    }

    // MARK: - Compile / Release

    /// Compile a program for a given operation and shape.
    ///
    /// Compilation is cached — subsequent calls with the same (op, shape, weightHash)
    /// return in sub-millisecond time.
    ///
    /// - Parameters:
    ///   - op: The operation to compile.
    ///   - shape: The tensor shape (must satisfy ANE constraints).
    ///   - weights: Optional weight data (fp16 bytes).
    /// - Throws: `ANEError` on failure.
    public func compile(op: libane_op_t,
                        shape: libane_shape_t,
                        weights: Data? = nil) throws -> ANEProgram {
        let h: libane_handle_t?
        if let w = weights {
            h = w.withUnsafeBytes { ptr in
                libane_compile(op, shape, ptr.baseAddress, ptr.count)
            }
        } else {
            h = libane_compile(op, shape, nil, 0)
        }
        guard let handle = h else {
            throw ANEError.compileFailed(lastError)
        }
        return ANEProgram(handle)
    }

    /// Execute a compiled program.
    ///
    /// - Parameters:
    ///   - program: Handle from `compile(...)`.
    ///   - input:   fp16 input bytes.
    ///   - output:  fp16 output buffer (caller-allocated, same element count as input).
    ///   - shape:   Runtime shape (must match compiled shape).
    /// - Throws: `ANEError` on failure.
    public func execute(_ program: ANEProgram,
                        input: UnsafeRawPointer,
                        output: UnsafeMutableRawPointer,
                        shape: libane_shape_t) throws {
        let st = libane_execute(program.handle, input, output, shape)
        if st != LIBANE_OK {
            throw ANEError.from(status: st)
        }
    }

    // MARK: - Convenience: Matmul

    /// fp16 matrix multiplication: C = A × B
    ///
    /// A: [M, K], B: [K, N], C: [M, N] — all fp16 row-major.
    /// Falls back to Accelerate if ANE is unavailable or shapes are unsupported.
    ///
    /// - Returns: Result array [M × N] of fp16 values (as UInt16 raw bytes).
    /// - Throws: `ANEError` on failure.
    public func matmul(a: [UInt16], b: [UInt16], M: Int, K: Int, N: Int) throws -> [UInt16] {
        guard a.count == M * K else {
            throw ANEError.invalidArgument("a.count (\(a.count)) != M*K (\(M*K))")
        }
        guard b.count == K * N else {
            throw ANEError.invalidArgument("b.count (\(b.count)) != K*N (\(K*N))")
        }

        var result = [UInt16](repeating: 0, count: M * N)
        let st = a.withUnsafeBufferPointer { aptr in
            b.withUnsafeBufferPointer { bptr in
                result.withUnsafeMutableBufferPointer { cptr in
                    libane_matmul_f16(
                        UnsafePointer<libane_f16_t>(OpaquePointer(aptr.baseAddress!)),
                        UnsafePointer<libane_f16_t>(OpaquePointer(bptr.baseAddress!)),
                        UnsafeMutablePointer<libane_f16_t>(OpaquePointer(cptr.baseAddress!)),
                        Int32(M), Int32(K), Int32(N)
                    )
                }
            }
        }
        if st != LIBANE_OK {
            throw ANEError.from(status: st)
        }
        return result
    }

    /// fp32 matrix multiplication: C = A × B
    ///
    /// A: [M, K], B: [K, N], C: [M, N] — float32 row-major.
    /// Internally casts fp32→fp16 on the ANE path; returns fp32.
    ///
    /// - Throws: `ANEError` on failure.
    public func matmulF32(a: [Float], b: [Float], M: Int, K: Int, N: Int) throws -> [Float] {
        guard a.count == M * K else {
            throw ANEError.invalidArgument("a.count (\(a.count)) != M*K (\(M*K))")
        }
        guard b.count == K * N else {
            throw ANEError.invalidArgument("b.count (\(b.count)) != K*N (\(K*N))")
        }

        var result = [Float](repeating: 0, count: M * N)
        let st = a.withUnsafeBufferPointer { aptr in
            b.withUnsafeBufferPointer { bptr in
                result.withUnsafeMutableBufferPointer { cptr in
                    libane_matmul_f32(aptr.baseAddress, bptr.baseAddress,
                                      cptr.baseAddress, Int32(M), Int32(K), Int32(N))
                }
            }
        }
        if st != LIBANE_OK {
            throw ANEError.from(status: st)
        }
        return result
    }

    // MARK: - Cache

    /// Flush the compile cache and release all cached programs.
    public func cacheFlush() {
        libane_cache_flush()
    }

    /// Current compile cache usage in bytes.
    public var cacheSizeBytes: Int {
        Int(libane_cache_size_bytes())
    }
}

// MARK: - Shape construction helpers

extension libane_shape_t {
    /// Construct a 4D ANE shape [batch=1, C, H=1, S].
    public static func ane(channels C: Int32, seq S: Int32) -> libane_shape_t {
        libane_shape_t(dims: (1, C, 1, S), ndim: 4)
    }

    /// Validate the shape satisfies ANE constraints.
    /// Returns nil if valid, or an error string if invalid.
    public func validate() -> String? {
        if dims.0 != 1    { return "batch must be 1" }
        if dims.2 != 1    { return "height must be 1" }
        if dims.3 % 8 != 0 { return "S must be multiple of 8" }
        if dims.3 > 65536  { return "S must be ≤ 65536" }
        if dims.1 > 16384  { return "C must be ≤ 16384" }
        return nil
    }
}
