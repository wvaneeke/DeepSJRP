# device.jl - CPU/GPU device dispatch utilities

using Random
using LinearAlgebra

# Global device configuration
const DEVICE_CONFIG = Ref{Symbol}(:cpu)

"""
    check_cuda_available()

Check if CUDA is functional and available.
"""
function check_cuda_available()
    return CUDA.functional()
end


"""
    set_device!(device::Symbol)

Set the compute device. Valid options: :cpu, :gpu
"""
function set_device!(device::Symbol)
    if device == :gpu
        if check_cuda_available()
            CUDA.allowscalar(false)
            DEVICE_CONFIG[] = :gpu
            println("Device: CUDA GPU")
            CUDA.versioninfo()
        else
            DEVICE_CONFIG[] = :cpu
            println("WARNING: CUDA not available, falling back to CPU")
        end
    else
        DEVICE_CONFIG[] = :cpu
        println("Device: CPU")
    end
    println("BLAS threads: ", BLAS.get_num_threads())
    println("Julia threads: ", Threads.nthreads())
    flush(stdout)
    return DEVICE_CONFIG[]
end


"""
    get_device()

Get the current compute device.
"""
get_device() = DEVICE_CONFIG[]


"""
    is_gpu()

Check if currently using GPU.
"""
is_gpu() = DEVICE_CONFIG[] == :gpu


"""
    to_device(x)

Move data to the current device.
"""
function to_device(x)
    if is_gpu()
        return cu(x)
    else
        return x
    end
end


"""
    to_cpu(x)

Move data from GPU to CPU if necessary.
"""
function to_cpu(x)
    if is_gpu()
        return Array(x)
    else
        return x
    end
end


"""
    randn_device(T::Type, dims...)

Generate random normal samples on the current device.
"""
function randn_device(T::Type, dims...)
    if is_gpu()
        return CUDA.randn(T, dims...)
    else
        return randn(T, dims...)
    end
end


"""
    rand_device(T::Type, dims...)

Generate uniform random samples on the current device.
"""
function rand_device(T::Type, dims...)
    if is_gpu()
        return CUDA.rand(T, dims...)
    else
        return rand(T, dims...)
    end
end


"""
    rand_device(dims...)

Generate uniform random Float32 samples on the current device.
"""
function rand_device(dims...)
    return rand_device(Float32, dims...)
end


"""
    zeros_device(T::Type, dims...)

Create zero array on the current device.
"""
function zeros_device(T::Type, dims...)
    if is_gpu()
        return CUDA.zeros(T, dims...)
    else
        return zeros(T, dims...)
    end
end


"""
    ones_device(T::Type, dims...)

Create ones array on the current device.
"""
function ones_device(T::Type, dims...)
    if is_gpu()
        return CUDA.ones(T, dims...)
    else
        return ones(T, dims...)
    end
end


"""
    set_random_seed!(seed::Int)

Set random seed for reproducibility on the current device.
"""
function set_random_seed!(seed::Int)
    Random.seed!(seed)
    if is_gpu()
        CUDA.seed!(seed)
    end
end


"""
    device_of(x)

Return a string indicating the array type, element type, and shape.
"""
function device_of(x::AbstractArray)
    return "$(nameof(typeof(x))){$(eltype(x))} $(size(x))"
end
device_of(x) = "$(typeof(x))"


const _DEVICE_VERIFIED = Ref(false)

"""
    verify_devices(; kwargs...)

Print device location of named arrays (runs once, then silences itself).
Call `reset_device_check!()` to re-enable.

# Example
    verify_devices(X=X_sample, dB=dB_sample, μ=μ)
"""
function verify_devices(; kwargs...)
    _DEVICE_VERIFIED[] && return
    _DEVICE_VERIFIED[] = true
    println("─── Device check ───")
    for (name, val) in pairs(kwargs)
        println("  $(name): $(device_of(val))")
    end
    println("────────────────────")
    flush(stdout)
end

"""
    reset_device_check!()

Re-enable `verify_devices` for the next call.
"""
reset_device_check!() = (_DEVICE_VERIFIED[] = false; nothing)


"""
    get_flux_device()

Get the Flux device function for the current device.
"""
function get_flux_device()
    if is_gpu()
        return Flux.gpu
    else
        return Flux.cpu
    end
end
