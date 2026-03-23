import XCTest
@testable import ANE

final class ANEEngineTests: XCTestCase {

    var engine: ANEEngine!

    override func setUp() {
        engine = ANEEngine(logLevel: .silent)
        engine.setBackend(.cpu) // Force CPU for deterministic test results
    }

    override func tearDown() {
        engine.setBackend(nil)
        engine = nil
    }

    // MARK: - Basic properties

    func testVersionString() {
        let v = engine.version
        XCTAssertFalse(v.isEmpty)
        XCTAssertTrue(v.hasPrefix("0."), "version should start with '0.'")
    }

    func testAvailableIsBoolean() {
        // Just check it returns without crashing
        let _ = engine.isAvailable
    }

    // MARK: - Matmul fp32

    func testMatmulF32IdentityMatrix() throws {
        let N = 8
        // Identity matrix
        var I = [Float](repeating: 0, count: N * N)
        for i in 0..<N { I[i * N + i] = 1.0 }

        // A = [0, 1, ..., 63]
        let A = (0..<N*N).map { Float($0) }

        let C = try engine.matmulF32(a: I, b: A, M: N, K: N, N: N)
        XCTAssertEqual(C.count, N * N)

        for i in 0..<N*N {
            XCTAssertEqual(C[i], A[i], accuracy: 0.01)
        }
    }

    func testMatmulF32KnownResult() throws {
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        let A: [Float] = [1,2,3, 4,5,6]
        let B: [Float] = [7,8, 9,10, 11,12]
        let C = try engine.matmulF32(a: A, b: B, M: 2, K: 3, N: 2)

        XCTAssertEqual(C[0], 58.0,  accuracy: 0.1)
        XCTAssertEqual(C[1], 64.0,  accuracy: 0.1)
        XCTAssertEqual(C[2], 139.0, accuracy: 0.1)
        XCTAssertEqual(C[3], 154.0, accuracy: 0.1)
    }

    func testMatmulF32ShapeMismatchThrows() {
        let A: [Float] = [1, 2, 3, 4]  // 2×2
        let B: [Float] = [1, 2]        // should be 2×1 but we declare 3×1 → mismatch

        XCTAssertThrowsError(try engine.matmulF32(a: A, b: B, M: 2, K: 3, N: 1))
    }

    // MARK: - Matmul fp16

    func testMatmulF16UniformOnes() throws {
        let M = 8, K = 8, N = 8
        // Both A and B are all-ones in fp16 (0x3C00)
        let ones = [UInt16](repeating: 0x3C00, count: M * K)
        let C = try engine.matmul(a: ones, b: ones, M: M, K: K, N: N)

        XCTAssertEqual(C.count, M * N)
        // Each element should be K * 1.0 * 1.0 = 8.0
        // fp16(8.0) = 0x4800
        for v in C {
            let f = fp16ToFloat(v)
            XCTAssertEqual(f, Float(K), accuracy: 0.5)
        }
    }

    // MARK: - Compile and execute

    func testCompileSoftmax() throws {
        var shape = libane_shape_t.ane(channels: 1, seq: 8)
        let prog = try engine.compile(op: LIBANE_OP_SOFTMAX, shape: shape)
        XCTAssertNotNil(prog)
    }

    func testCompileGelu() throws {
        let shape = libane_shape_t.ane(channels: 1, seq: 8)
        let prog = try engine.compile(op: LIBANE_OP_GELU, shape: shape)
        XCTAssertNotNil(prog)
    }

    func testExecuteSoftmaxSumsToOne() throws {
        let S = 8
        let shape = libane_shape_t.ane(channels: 1, seq: Int32(S))
        let prog = try engine.compile(op: LIBANE_OP_SOFTMAX, shape: shape)

        // Input: [1, 2, 3, 4, 5, 6, 7, 8] as fp16
        var input  = [UInt16](repeating: 0, count: S)
        var output = [UInt16](repeating: 0, count: S)
        for i in 0..<S { input[i] = floatToFp16(Float(i + 1)) }

        try input.withUnsafeBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try engine.execute(prog,
                                   input:  UnsafeRawPointer(inPtr.baseAddress!),
                                   output: UnsafeMutableRawPointer(outPtr.baseAddress!),
                                   shape:  shape)
            }
        }

        let sum = output.reduce(0.0) { $0 + fp16ToFloat($1) }
        XCTAssertEqual(sum, 1.0, accuracy: 0.05) // fp16 precision
    }

    // MARK: - Shape validation

    func testShapeValidationPasses() {
        let s = libane_shape_t.ane(channels: 64, seq: 512)
        XCTAssertNil(s.validate())
    }

    func testShapeValidationFailsNotMultipleOf8() {
        let s = libane_shape_t.ane(channels: 64, seq: 513)
        XCTAssertNotNil(s.validate())
    }

    func testShapeValidationFailsLargeSeq() {
        let s = libane_shape_t.ane(channels: 64, seq: 65537)
        XCTAssertNotNil(s.validate())
    }

    // MARK: - Cache

    func testCacheFlush() {
        let shape = libane_shape_t.ane(channels: 8, seq: 64)
        let _ = try? engine.compile(op: LIBANE_OP_GELU, shape: shape)
        XCTAssertGreaterThan(engine.cacheSizeBytes, 0)
        engine.cacheFlush()
        XCTAssertEqual(engine.cacheSizeBytes, 0)
    }

    // MARK: - Helpers

    private func fp16ToFloat(_ h: UInt16) -> Float {
        let sign     = UInt32(h & 0x8000) << 16
        let exp      = UInt32((h >> 10) & 0x1F)
        let mantissa = UInt32(h & 0x3FF)
        var fb: UInt32
        if exp == 0 {       fb = sign | (mantissa << 13) }
        else if exp == 31 { fb = sign | 0x7F800000 | (mantissa << 13) }
        else {              fb = sign | ((exp + 112) << 23) | (mantissa << 13) }
        var f: Float = 0
        withUnsafeMutableBytes(of: &f) { $0.storeBytes(of: fb, as: UInt32.self) }
        return f
    }

    private func floatToFp16(_ f: Float) -> UInt16 {
        var fb: UInt32 = 0
        withUnsafeBytes(of: f) { fb = $0.load(as: UInt32.self) }
        let sign = UInt16((fb >> 16) & 0x8000)
        let exp  = Int32((fb >> 23) & 0xFF) - 127 + 15
        let mant = UInt16((fb >> 13) & 0x3FF)
        if exp <= 0        { return sign }
        else if exp >= 31  { return sign | 0x7C00 }
        else               { return sign | UInt16(exp << 10) | mant }
    }
}
