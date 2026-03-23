#include "buffer_manager.hpp"
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <cstdlib>

#include "../fallback/fallback.hpp"
using fp16_t = libane::fallback::fp16_t;

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#  include <CoreFoundation/CoreFoundation.h>
#endif

namespace libane {

/* ── AneBuffer ───────────────────────────────────────────────────────────── */

AneBuffer::~AneBuffer() {
    if (!owned_) return; // external data, don't free

#ifdef __APPLE__
    if (iosurface_) {
        // Unmap and release
        IOSurfaceUnlock(iosurface_, kIOSurfaceLockReadOnly, nullptr);
        CFRelease(iosurface_);
        iosurface_ = nullptr;
        ptr_ = nullptr;
        return;
    }
#endif
    if (ptr_) {
        std::free(ptr_);
        ptr_ = nullptr;
    }
}

AneBuffer::AneBuffer(AneBuffer&& o) noexcept
    : ptr_(o.ptr_), bytes_(o.bytes_), owned_(o.owned_)
#ifdef __APPLE__
    , iosurface_(o.iosurface_)
#endif
{
    o.ptr_ = nullptr;
    o.bytes_ = 0;
#ifdef __APPLE__
    o.iosurface_ = nullptr;
#endif
}

AneBuffer& AneBuffer::operator=(AneBuffer&& o) noexcept {
    if (this != &o) {
        this->~AneBuffer();
        new(this) AneBuffer(std::move(o));
    }
    return *this;
}

void AneBuffer::lock_for_cpu() {
#ifdef __APPLE__
    if (iosurface_) {
        IOSurfaceLock(iosurface_, 0 /*read-write*/, nullptr);
        ptr_ = IOSurfaceGetBaseAddress(iosurface_);
    }
#endif
}

void AneBuffer::unlock_for_cpu() {
#ifdef __APPLE__
    if (iosurface_) {
        IOSurfaceUnlock(iosurface_, 0, nullptr);
    }
#endif
}

void AneBuffer::copy_from(const void* src, size_t nbytes) {
    assert(nbytes <= bytes_);
    lock_for_cpu();
    std::memcpy(ptr_, src, nbytes);
    unlock_for_cpu();
}

void AneBuffer::copy_from_f32(const float* src, size_t count) {
    assert(count * 2 <= bytes_);
    lock_for_cpu();
    auto* dst = static_cast<fp16_t*>(ptr_);
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
    for (size_t i = 0; i < count; ++i)
        dst[i] = static_cast<fp16_t>(src[i]);
#else
    // Software fp32->fp16
    auto* dst16 = static_cast<uint16_t*>(ptr_);
    for (size_t i = 0; i < count; ++i) {
        uint32_t fb; std::memcpy(&fb, &src[i], 4);
        uint32_t sign     = (fb >> 16) & 0x8000;
        int32_t  exp      = ((fb >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (fb >> 13) & 0x3FF;
        uint16_t h;
        if (exp <= 0)       h = static_cast<uint16_t>(sign);
        else if (exp >= 31) h = static_cast<uint16_t>(sign | 0x7C00);
        else                h = static_cast<uint16_t>(sign | (exp << 10) | mantissa);
        dst16[i] = h;
    }
#endif
    unlock_for_cpu();
}

void AneBuffer::copy_to(void* dst, size_t nbytes) const {
#ifdef __APPLE__
    if (iosurface_) {
        // Lock read-only for the copy
        IOSurfaceLock(iosurface_, kIOSurfaceLockReadOnly, nullptr);
        std::memcpy(dst, IOSurfaceGetBaseAddress(iosurface_), nbytes);
        IOSurfaceUnlock(iosurface_, kIOSurfaceLockReadOnly, nullptr);
        return;
    }
#endif
    std::memcpy(dst, ptr_, nbytes);
}

void AneBuffer::copy_to_f32(float* dst, size_t count) const {
    assert(count * 2 <= bytes_);
#ifdef __APPLE__
    const fp16_t* src;
    if (iosurface_) {
        IOSurfaceLock(iosurface_, kIOSurfaceLockReadOnly, nullptr);
        src = static_cast<const fp16_t*>(IOSurfaceGetBaseAddress(iosurface_));
    } else {
        src = static_cast<const fp16_t*>(ptr_);
    }
#else
    const fp16_t* src = static_cast<const fp16_t*>(ptr_);
#endif

#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
    for (size_t i = 0; i < count; ++i)
        dst[i] = static_cast<float>(src[i]);
#else
    const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < count; ++i) {
        uint16_t h = src16[i];
        uint32_t sign     = (h & 0x8000) << 16;
        uint32_t exp      = ((h >> 10) & 0x1F);
        uint32_t mantissa = (h & 0x3FF);
        uint32_t fb;
        if (exp == 0)       fb = sign | (mantissa << 13);
        else if (exp == 31) fb = sign | 0x7F800000 | (mantissa << 13);
        else                fb = sign | ((exp + 112) << 23) | (mantissa << 13);
        std::memcpy(&dst[i], &fb, 4);
    }
#endif

#ifdef __APPLE__
    if (iosurface_)
        IOSurfaceUnlock(iosurface_, kIOSurfaceLockReadOnly, nullptr);
#endif
}

/* ── BufferPool ──────────────────────────────────────────────────────────── */

BufferPool::BufferPool(bool use_iosurface, size_t max_idle_per_size)
    : use_iosurface_(use_iosurface)
    , max_idle_per_size_(max_idle_per_size)
{}

BufferPool::~BufferPool() {
    flush();
}

std::unique_ptr<AneBuffer> BufferPool::allocate(size_t nbytes) {
    auto buf = std::unique_ptr<AneBuffer>(new AneBuffer());
    buf->bytes_ = nbytes;

#ifdef __APPLE__
    if (use_iosurface_) {
        // ANE IOSurface layout (matching Orion's orion_tensor_create exactly):
        //   kIOSurfaceWidth           = surface_bytes  (padded to 49 KB minimum)
        //   kIOSurfaceHeight          = 1
        //   kIOSurfaceBytesPerElement = 1
        //   kIOSurfaceBytesPerRow     = surface_bytes
        //   kIOSurfaceAllocSize       = surface_bytes
        //   kIOSurfacePixelFormat     = 0
        //
        // Constraint #4: ANE requires IOSurface allocation >= 49 KB for eval to succeed.
        // Pad the surface size but keep buf->bytes_ as the real data size for
        // CPU copy operations. All IOSurface dimensions must be padded consistently
        // because the ANE derives channel stride from the total allocation size.
        // Tensors smaller than 49 KB are not supported for ANE execution.
        static constexpr size_t kMinIOSurface = 49152;
        size_t surface_bytes = std::max(nbytes, kMinIOSurface);

        CFMutableDictionaryRef props = CFDictionaryCreateMutable(
            kCFAllocatorDefault, 0,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks);

        int64_t width     = static_cast<int64_t>(surface_bytes);
        int64_t height    = 1;
        int64_t bpe       = 1;
        int64_t bpr       = static_cast<int64_t>(surface_bytes);
        int64_t allocsize = static_cast<int64_t>(surface_bytes);
        int64_t fmt       = 0;

        CFNumberRef width_ref  = CFNumberCreate(nullptr, kCFNumberSInt64Type, &width);
        CFNumberRef height_ref = CFNumberCreate(nullptr, kCFNumberSInt64Type, &height);
        CFNumberRef bpe_ref    = CFNumberCreate(nullptr, kCFNumberSInt64Type, &bpe);
        CFNumberRef bpr_ref    = CFNumberCreate(nullptr, kCFNumberSInt64Type, &bpr);
        CFNumberRef alloc_ref  = CFNumberCreate(nullptr, kCFNumberSInt64Type, &allocsize);
        CFNumberRef fmt_ref    = CFNumberCreate(nullptr, kCFNumberSInt64Type, &fmt);

        CFDictionarySetValue(props, kIOSurfaceWidth,           width_ref);
        CFDictionarySetValue(props, kIOSurfaceHeight,          height_ref);
        CFDictionarySetValue(props, kIOSurfaceBytesPerElement, bpe_ref);
        CFDictionarySetValue(props, kIOSurfaceBytesPerRow,     bpr_ref);
        CFDictionarySetValue(props, kIOSurfaceAllocSize,       alloc_ref);
        CFDictionarySetValue(props, kIOSurfacePixelFormat,     fmt_ref);

        IOSurfaceRef surface = IOSurfaceCreate(props);
        CFRelease(width_ref); CFRelease(height_ref);
        CFRelease(bpe_ref); CFRelease(bpr_ref); CFRelease(alloc_ref); CFRelease(fmt_ref);
        CFRelease(props);

        if (surface) {
            buf->iosurface_ = surface;
            IOSurfaceLock(surface, 0, nullptr);
            buf->ptr_ = IOSurfaceGetBaseAddress(surface);
            IOSurfaceUnlock(surface, 0, nullptr);
            return buf;
        }
        // Fall through to malloc if IOSurface fails
    }
#endif

    // Plain aligned allocation
    void* p = nullptr;
    if (posix_memalign(&p, 64, nbytes) != 0 || p == nullptr)
        throw std::bad_alloc();
    buf->ptr_ = p;
    return buf;
}

std::unique_ptr<AneBuffer> BufferPool::acquire(size_t nbytes) {
    std::lock_guard lock(mutex_);
    auto it = idle_.find(nbytes);
    if (it != idle_.end() && !it->second.empty()) {
        auto buf = std::move(it->second.back());
        it->second.pop_back();
        return buf;
    }
    return allocate(nbytes);
}

std::unique_ptr<AneBuffer> BufferPool::acquire_with_data(const void* fp16_data,
                                                           size_t nbytes) {
    auto buf = acquire(nbytes);
    buf->copy_from(fp16_data, nbytes);
    return buf;
}

std::unique_ptr<AneBuffer> BufferPool::acquire_from_f32(const float* fp32_data,
                                                          size_t count) {
    auto buf = acquire(count * 2);
    buf->copy_from_f32(fp32_data, count);
    return buf;
}

void BufferPool::release(std::unique_ptr<AneBuffer> buf) {
    if (!buf) return;
    size_t sz = buf->bytes_;
    std::lock_guard lock(mutex_);
    auto& pool = idle_[sz];
    if (pool.size() < max_idle_per_size_) {
        pool.push_back(std::move(buf));
    }
    // else: let the unique_ptr destructor run
}

void BufferPool::flush() {
    std::lock_guard lock(mutex_);
    idle_.clear();
}

size_t BufferPool::pool_size() const {
    std::lock_guard lock(mutex_);
    size_t n = 0;
    for (auto& [sz, v] : idle_) n += v.size();
    return n;
}

/* ── Global pool ─────────────────────────────────────────────────────────── */

BufferPool& global_buffer_pool() {
#ifdef __APPLE__
    static BufferPool pool(/*use_iosurface=*/true);
#else
    static BufferPool pool(/*use_iosurface=*/false);
#endif
    return pool;
}

} // namespace libane
