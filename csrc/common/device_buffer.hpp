#pragma once

// Owning wrapper around cudaMalloc / cudaFree for an array of T in GPU memory. The memory is
// freed when the object goes out of scope, including when an exception unwinds the stack.

#include "check.hpp"

#include <cstddef>
#include <cstdio>

template <class T>
class DeviceBuffer {
public:
    // Allocates room for `count` elements. The contents are uninitialized.
    explicit DeviceBuffer(std::size_t count) : count_(count)
    {
        if (count_ > 0) {
            void* raw = nullptr;
            CUDA_CHECK(cudaMalloc(&raw, bytes()));
            ptr_ = static_cast<T*>(raw);
        }
    }

    ~DeviceBuffer()
    {
        // A destructor must not throw, so a failed free is reported instead.
        const cudaError_t err = cudaFree(ptr_);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "warning: cudaFree failed: %s\n", cudaGetErrorString(err));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    // Host-to-device copy of `count` elements into the start of the buffer. Blocks until done.
    void upload(const T* src, std::size_t count)
    {
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    // Host-to-device copy of `count` elements to elements [offset, offset + count).
    void upload_at(std::size_t offset, const T* src, std::size_t count)
    {
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(ptr_ + offset, src, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    // Device-to-host copy of `count` elements from the start of the buffer. Blocks until done,
    // and waits for kernels already launched on the default stream.
    void download(T* dst, std::size_t count) const
    {
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    // Sets every byte to zero.
    void zero()
    {
        if (count_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
        }
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};
