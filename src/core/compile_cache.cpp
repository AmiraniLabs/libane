#include "compile_cache.hpp"
#include <cstring>
#include <functional>
#include <filesystem>
#include <mutex>

namespace libane {

/* ── CacheKey ────────────────────────────────────────────────────────────── */

bool CacheKey::operator==(const CacheKey& o) const noexcept {
    return op == o.op &&
           weight_hash == o.weight_hash &&
           std::memcmp(&shape, &o.shape, sizeof(shape)) == 0;
}

static size_t hash_combine(size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

size_t CacheKeyHash::operator()(const CacheKey& k) const noexcept {
    size_t h = std::hash<int>{}(static_cast<int>(k.op));
    h = hash_combine(h, std::hash<uint64_t>{}(k.weight_hash));
    h = hash_combine(h, std::hash<int32_t>{}(k.shape.dims[0]));
    h = hash_combine(h, std::hash<int32_t>{}(k.shape.dims[1]));
    h = hash_combine(h, std::hash<int32_t>{}(k.shape.dims[2]));
    h = hash_combine(h, std::hash<int32_t>{}(k.shape.dims[3]));
    return h;
}

/* ── CompileCache ────────────────────────────────────────────────────────── */

CompileCache::CompileCache(size_t max_bytes)
    : max_bytes_(max_bytes) {}

std::shared_ptr<CacheEntry> CompileCache::get(const CacheKey& key) {
    std::shared_lock lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    hits_.fetch_add(1, std::memory_order_relaxed);

    // Promote to front of LRU (needs write access — upgrade lock)
    lock.unlock();
    std::unique_lock wlock(mutex_);
    auto it2 = map_.find(key);
    if (it2 == map_.end()) return nullptr; // evicted between unlock/relock
    lru_.splice(lru_.begin(), lru_, it2->second.lru_pos);
    return it2->second.entry;
}

std::shared_ptr<CacheEntry> CompileCache::put(std::unique_ptr<CacheEntry> entry) {
    if (!entry) return nullptr;

    std::unique_lock lock(mutex_);

    // Remove existing entry with same key
    auto it = map_.find(entry->key);
    if (it != map_.end()) {
        current_bytes_ -= it->second.entry->size_bytes;
        lru_.erase(it->second.lru_pos);
        map_.erase(it);
    }

    // Evict until there is room
    size_t entry_size = entry->size_bytes;
    while (current_bytes_ + entry_size > max_bytes_ && !lru_.empty()) {
        evict_lru_locked();
    }

    entry->inserted_at = std::chrono::steady_clock::now();
    auto shared = std::shared_ptr<CacheEntry>(std::move(entry));
    lru_.push_front(shared);
    map_[shared->key] = { lru_.begin(), shared };
    current_bytes_ += entry_size;

    return shared;
}

bool CompileCache::erase(const CacheKey& key) {
    std::unique_lock lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    current_bytes_ -= it->second.entry->size_bytes;
    lru_.erase(it->second.lru_pos);
    map_.erase(it);
    return true;
}

void CompileCache::evict_lru_locked() {
    if (lru_.empty()) return;
    auto& victim = lru_.back();
    current_bytes_ -= victim->size_bytes;
    map_.erase(victim->key);
    lru_.pop_back();
    evictions_.fetch_add(1, std::memory_order_relaxed);
}

void CompileCache::flush() {
    std::unique_lock lock(mutex_);
    lru_.clear();
    map_.clear();
    current_bytes_ = 0;
}

size_t CompileCache::size_bytes() const {
    std::shared_lock lock(mutex_);
    return current_bytes_;
}

size_t CompileCache::size() const {
    std::shared_lock lock(mutex_);
    return map_.size();
}

CompileCache::Stats CompileCache::stats() const {
    Stats snapshot;
    snapshot.hits = hits_.load(std::memory_order_relaxed);
    snapshot.misses = misses_.load(std::memory_order_relaxed);
    snapshot.evictions = evictions_.load(std::memory_order_relaxed);
    return snapshot;
}

void CompileCache::enable_disk_cache(const std::string& directory) {
    std::unique_lock lock(mutex_);
    std::filesystem::create_directories(directory);
    disk_cache_dir_ = directory;
}

void CompileCache::disable_disk_cache() {
    std::unique_lock lock(mutex_);
    disk_cache_dir_ = std::nullopt;
}

} // namespace libane
