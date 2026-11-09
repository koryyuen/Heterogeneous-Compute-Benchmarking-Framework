#include "heterobench/metal_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace heterobench {
namespace {

std::string error_description(NSError* error) {
    if (error == nil) return "No additional error information";
    const char* description = [[error localizedDescription] UTF8String];
    return description != nullptr ? description : "Unknown Metal error";
}

} // namespace

struct MetalBackend::Impl {
    MetalKernel kernel = MetalKernel::Naive;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLBuffer> buffer_a = nil;
    id<MTLBuffer> buffer_b = nil;
    id<MTLBuffer> buffer_c = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    MPSMatrix* mps_a = nil;
    MPSMatrix* mps_b = nil;
    MPSMatrix* mps_c = nil;
    MPSMatrixMultiplication* mps_multiply = nil;

    MatrixData* matrix_data = nullptr;
    std::size_t element_count = 0;
    std::size_t row_bytes = 0;
    std::uint32_t dimension = 0;
    std::optional<double> gpu_ms;

    ~Impl() {
        // Destruction waits for resource safety; only synchronize() returns output.
        if (command_buffer != nil) [command_buffer waitUntilCompleted];
    }
};

bool MetalBackend::is_available() {
    @autoreleasepool { return MTLCreateSystemDefaultDevice() != nil; }
}

MetalBackend::MetalBackend(MetalKernel kernel) {
    impl_ = std::make_unique<Impl>();
    impl_->kernel = kernel;
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (impl_->device == nil) throw std::runtime_error("No Metal device is available");
        impl_->queue = [impl_->device newCommandQueue];
        if (impl_->queue == nil) throw std::runtime_error("Failed to create Metal command queue");
    }
}

MetalBackend::~MetalBackend() = default;

std::string MetalBackend::device_name() const {
    @autoreleasepool {
        const char* name = [[impl_->device name] UTF8String];
        return name != nullptr ? name : "Unknown Metal device";
    }
}

std::optional<double> MetalBackend::last_gpu_time_ms() const { return impl_->gpu_ms; }

void MetalBackend::prepare(const WorkloadConfig& config, MatrixData& data) {
    if (impl_->command_buffer != nil) {
        throw std::logic_error("Synchronize pending Metal execution before preparing again");
    }
    const std::size_t dimension = config.matrix_size;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (dimension == 0 || dimension > maximum / dimension) {
        throw std::invalid_argument("Matrix dimension is zero or its square overflows");
    }
    const std::size_t count = dimension * dimension;
    if (count > maximum / sizeof(float)) {
        throw std::invalid_argument("Matrix byte count overflows std::size_t");
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Matrix exceeds the supported 32-bit shader indexing range");
    }
    if (data.a.size() != count || data.b.size() != count || data.c.size() != count) {
        throw std::invalid_argument("Each matrix must contain dimension * dimension elements");
    }

    @autoreleasepool {
        // MPS recommends padded row strides. Plain kernels use contiguous rows.
        std::size_t row_bytes = dimension * sizeof(float);
        if (impl_->kernel == MetalKernel::Mps) {
            if (!MPSSupportsMTLDevice(impl_->device)) throw std::runtime_error("MPS is unavailable on this device");
            row_bytes = [MPSMatrixDescriptor rowBytesFromColumns:dimension dataType:MPSDataTypeFloat32];
        }
        if (row_bytes < dimension * sizeof(float) || row_bytes > maximum / dimension) {
            throw std::invalid_argument("Matrix row stride overflows");
        }
        const std::size_t bytes = row_bytes * dimension;
        if (bytes > [impl_->device maxBufferLength]) {
            throw std::invalid_argument("Matrix exceeds Metal's maximum buffer length");
        }
        id<MTLBuffer> buffer_a = [impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_b = [impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_c = [impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buffer_a == nil || buffer_b == nil || buffer_c == nil) {
            throw std::runtime_error("Failed to allocate Metal matrix buffers");
        }
        for (std::size_t row = 0; row < dimension; ++row) {
            auto* a_row = static_cast<char*>([buffer_a contents]) + row * row_bytes;
            auto* b_row = static_cast<char*>([buffer_b contents]) + row * row_bytes;
            std::memcpy(a_row, data.a.data() + row * dimension, dimension * sizeof(float));
            std::memcpy(b_row, data.b.data() + row * dimension, dimension * sizeof(float));
        }

        id<MTLComputePipelineState> pipeline = nil;
        MPSMatrix* mps_a = nil;
        MPSMatrix* mps_b = nil;
        MPSMatrix* mps_c = nil;
        MPSMatrixMultiplication* mps_multiply = nil;
        if (impl_->kernel == MetalKernel::Mps) {
            MPSMatrixDescriptor* descriptor =
                [MPSMatrixDescriptor matrixDescriptorWithRows:dimension columns:dimension
                                                     rowBytes:row_bytes dataType:MPSDataTypeFloat32];
            mps_a = [[MPSMatrix alloc] initWithBuffer:buffer_a descriptor:descriptor];
            mps_b = [[MPSMatrix alloc] initWithBuffer:buffer_b descriptor:descriptor];
            mps_c = [[MPSMatrix alloc] initWithBuffer:buffer_c descriptor:descriptor];
            mps_multiply = [[MPSMatrixMultiplication alloc]
                initWithDevice:impl_->device transposeLeft:NO transposeRight:NO
                resultRows:dimension resultColumns:dimension interiorColumns:dimension alpha:1.0 beta:0.0];
            if (mps_a == nil || mps_b == nil || mps_c == nil || mps_multiply == nil) {
                throw std::runtime_error("Failed to prepare MPS matrix multiplication");
            }
        } else {
            NSString* path = [NSString stringWithUTF8String:HETEROBENCH_METALLIB_PATH];
            NSError* error = nil;
            id<MTLLibrary> library = [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            if (library == nil) throw std::runtime_error("Failed to load Metal library: " + error_description(error));
            NSString* name = impl_->kernel == MetalKernel::Tiled ? @"matmul_tiled" : @"matmul";
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (function == nil) throw std::runtime_error("Metal library is missing the selected kernel");
            error = nil;
            pipeline = [impl_->device newComputePipelineStateWithFunction:function error:&error];
            if (pipeline == nil) throw std::runtime_error("Failed to create pipeline: " + error_description(error));
            if (impl_->kernel == MetalKernel::Tiled && [pipeline maxTotalThreadsPerThreadgroup] < 256) {
                throw std::runtime_error("Device cannot run the fixed 16 x 16 tiled kernel");
            }
        }

        // Failed preparation leaves the preceding valid state usable.
        impl_->buffer_a = buffer_a;
        impl_->buffer_b = buffer_b;
        impl_->buffer_c = buffer_c;
        impl_->pipeline = pipeline;
        impl_->mps_a = mps_a;
        impl_->mps_b = mps_b;
        impl_->mps_c = mps_c;
        impl_->mps_multiply = mps_multiply;
        impl_->matrix_data = &data;
        impl_->element_count = count;
        impl_->row_bytes = row_bytes;
        impl_->dimension = static_cast<std::uint32_t>(dimension);
        impl_->gpu_ms.reset();
    }
}

void MetalBackend::execute() {
    if (impl_->matrix_data == nullptr) throw std::logic_error("Metal execution requires prepare");
    if (impl_->command_buffer != nil) throw std::logic_error("Synchronize before submitting again");
    if (impl_->matrix_data->a.size() != impl_->element_count ||
        impl_->matrix_data->b.size() != impl_->element_count ||
        impl_->matrix_data->c.size() != impl_->element_count) {
        throw std::logic_error("Matrix sizes changed after preparation");
    }
    impl_->gpu_ms.reset();
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
        if (command_buffer == nil) throw std::runtime_error("Failed to create command buffer");
        [command_buffer setLabel:impl_->kernel == MetalKernel::Tiled ? @"HeteroBench tiled" :
                                impl_->kernel == MetalKernel::Mps ? @"HeteroBench MPS" : @"HeteroBench naive"];
        if (impl_->kernel == MetalKernel::Mps) {
            [impl_->mps_multiply encodeToCommandBuffer:command_buffer leftMatrix:impl_->mps_a
                                          rightMatrix:impl_->mps_b resultMatrix:impl_->mps_c];
        } else {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) throw std::runtime_error("Failed to create compute encoder");
            [encoder setComputePipelineState:impl_->pipeline];
            [encoder setBuffer:impl_->buffer_a offset:0 atIndex:0];
            [encoder setBuffer:impl_->buffer_b offset:0 atIndex:1];
            [encoder setBuffer:impl_->buffer_c offset:0 atIndex:2];
            const std::uint32_t dimension = impl_->dimension;
            [encoder setBytes:&dimension length:sizeof(dimension) atIndex:3];
            const NSUInteger maximum = [impl_->pipeline maxTotalThreadsPerThreadgroup];
            const NSUInteger width = std::min<NSUInteger>(16, maximum);
            const NSUInteger height = impl_->kernel == MetalKernel::Tiled ? 16 : std::min<NSUInteger>(16, maximum / width);
            const MTLSize threads = MTLSizeMake(width, height, 1);
            const MTLSize groups = MTLSizeMake((dimension + width - 1) / width,
                                               (dimension + height - 1) / height, 1);
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
            [encoder endEncoding];
        }
        impl_->command_buffer = command_buffer;
        [command_buffer commit];
    }
}

void MetalBackend::synchronize() {
    if (impl_->command_buffer == nil) throw std::logic_error("No pending Metal execution");
    @autoreleasepool {
        id<MTLCommandBuffer> completed = impl_->command_buffer;
        [completed waitUntilCompleted];
        impl_->command_buffer = nil;
        if ([completed status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Metal execution failed: " + error_description([completed error]));
        }
        const double start = [completed GPUStartTime];
        const double end = [completed GPUEndTime];
        if (std::isfinite(start) && std::isfinite(end) && start > 0.0 && end > start) {
            impl_->gpu_ms = (end - start) * 1000.0;
        }
        if (impl_->matrix_data->c.size() != impl_->element_count) {
            throw std::logic_error("Output size changed before readback");
        }
        const auto* source = static_cast<const char*>([impl_->buffer_c contents]);
        for (std::size_t row = 0; row < impl_->dimension; ++row) {
            std::memcpy(impl_->matrix_data->c.data() + row * impl_->dimension,
                        source + row * impl_->row_bytes, impl_->dimension * sizeof(float));
        }
    }
}

} // namespace heterobench
