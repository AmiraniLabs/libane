/**
 * IOSurface Buffer Manager — manages ANE-compatible tensor buffers.
 *
 * ANE tensor I/O requires IOSurface-backed memory in [1, C, 1, S] fp16 layout.
 * Handles:
 *  - Allocation of IOSurface-backed buffers in the correct layout
 *  - Zero-copy views into MLX unified memory arrays where layout matches
 *  - Layout conversion for fp32 inputs (copy + cast)
 *  - Buffer pool to amortize IOSurface allocation cost
 *
 * On non-ANE builds (x86 or fallback mode) this module degrades to
 * plain malloc-backed buffers with identical public interface.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#endif

namespace libane {

/* ── Buffer descriptor ───────────────────────────────────────────────────── */

/**
 * A tensor buffer compatible with ANE I/O.
 *
 * On ANE-capable hardware: backed by an IOSurface with the correct
 * pixel format (kCVPixelFormatType_OneComponent16Half) and stride.
 *
 * In fallback mode: plain aligned heap allocation.
 */
class AneBuffer {
public:
    ~AneBuffer();

    /* Non-copyable, movable. */
    AneBuffer(const AneBuffer&) = delete;
    AneBuffer& operator=(const AneBuffer&) = delete;
    AneBuffer(AneBuffer&&) noexcept;
    AneBuffer& operator=(AneBuffer&&) noexcept;

    /** Returns the base pointer to fp16 data. Always non-null after construction. */
    void*       data()       { return ptr_; }
    const void* data() const { return ptr_; }

    /** Size in bytes. */
    size_t bytes() const { return bytes_; }

    /** Number of fp16 elements. */
    size_t numel() const { return bytes_ / 2; }

    /** True if backed by IOSurface (ANE path). */
    bool is_iosurface() const { return iosurface_ != nullptr; }

#ifdef __APPLE__
    /** IOSurface ref (or nullptr in fallback mode). */
    IOSurfaceRef iosurface() const { return iosurface_; }
#endif

    /**
     * Lock the IOSurface for CPU access.
     * Must be called before reading/writing data() on the CPU side.
     * No-op in fallback mode.
     */
    void lock_for_cpu();
    void unlock_for_cpu();

    /**
     * Copy fp16 data into this buffer.
     * Handles IOSurface locking internally.
     */
    void copy_from(const void* src, size_t bytes);

    /**
     * Copy fp32 data into this buffer, casting to fp16.
     * count = number of float elements.
     */
    void copy_from_f32(const float* src, size_t count);

    /**
     * Copy fp16 data out of this buffer.
     */
    void copy_to(void* dst, size_t bytes) const;

    /**
     * Copy fp16 data out of this buffer, casting each element to fp32.
     */
    void copy_to_f32(float* dst, size_t count) const;

private:
    friend class BufferPool;

    AneBuffer() = default;

    void* ptr_      = nullptr;
    size_t bytes_   = 0;
    bool   owned_   = true;    // if false, ptr_ is external (zero-copy view)

#ifdef __APPLE__
    IOSurfaceRef iosurface_ = nullptr;
#else
    void* iosurface_ = nullptr;
#endif
};

/* ── Buffer pool ─────────────────────────────────────────────────────────── */

/**
 * Pool that amortizes IOSurface allocation cost.
 *
 * Buffers are partitioned by size. The pool keeps at most max_idle_per_size
 * idle buffers of each size. All pool operations are thread-safe.
 */
class BufferPool {
public:
    static constexpr size_t kDefaultMaxIdle = 4;

    explicit BufferPool(bool use_iosurface = true,
                        size_t max_idle_per_size = kDefaultMaxIdle);
    ~BufferPool();

    /**
     * Acquire a buffer of the given byte size.
     * May return a recycled buffer from the pool (contents undefined).
     */
    std::unique_ptr<AneBuffer> acquire(size_t bytes);

    /**
     * Acquire a buffer and copy fp16 data in.
     */
    std::unique_ptr<AneBuffer> acquire_with_data(const void* fp16_data, size_t bytes);

    /**
     * Acquire a buffer and copy-cast fp32 data in.
     */
    std::unique_ptr<AneBuffer> acquire_from_f32(const float* fp32_data, size_t count);

    /**
     * Return a buffer to the pool.
     * If the pool is full for this size, the buffer is destroyed.
     */
    void release(std::unique_ptr<AneBuffer> buf);

    /** Destroy all pooled buffers and free IOSurface resources. */
    void flush();

    /** Number of buffers currently in the pool. */
    size_t pool_size() const;

    /** Whether this pool uses IOSurface (ANE mode). */
    bool uses_iosurface() const { return use_iosurface_; }

private:
    std::unique_ptr<AneBuffer> allocate(size_t bytes);

    bool   use_iosurface_;
    size_t max_idle_per_size_;

    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<std::unique_ptr<AneBuffer>>> idle_;
};

/* ── Global singleton ────────────────────────────────────────────────────── */

/** Returns the process-wide BufferPool (initialized on first call). */
BufferPool& global_buffer_pool();

} // namespace libane
