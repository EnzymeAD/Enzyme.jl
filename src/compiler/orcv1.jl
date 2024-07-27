module JIT

using LLVM
import LLVM: TargetMachine

import GPUCompiler: CompilerJob, JuliaContext
import ..Compiler
import ..Compiler: API, cpu_name, cpu_features

export get_trampoline

# We have one global JIT and TM
const jit = Ref{OrcJIT}()
const tm  = Ref{TargetMachine}()

get_tm() = tm[]

function __init__()
    opt_level = Base.JLOptions().opt_level
    if opt_level < 2
        optlevel = LLVM.API.LLVMCodeGenLevelNone
    elseif opt_level == 2
        optlevel = LLVM.API.LLVMCodeGenLevelDefault
    else
        optlevel = LLVM.API.LLVMCodeGenLevelAggressive
    end

    tm[] = LLVM.JITTargetMachine(LLVM.triple(), cpu_name(), cpu_features(); optlevel)
    LLVM.asm_verbosity!(tm[], true)

    jit[] = OrcJIT(tm[]) # takes ownership of tm

    if haskey(ENV, "ENABLE_GDBLISTENER")
        LLVM.register!(jit[], LLVM.GDBRegistrationListener())
    end
    atexit() do
        dispose(jit[])
    end
end

mutable struct CallbackContext
    job::CompilerJob
    stub::Symbol
    l_job::ReentrantLock
    addr::Ptr{Cvoid}
    CallbackContext(job, stub, l_job) = new(job, stub, l_job, C_NULL)
end

const l_outstanding = Base.ReentrantLock()
const outstanding = Base.IdSet{CallbackContext}()

# Setup the lazy callback for creating a module
function callback(orc_ref::LLVM.API.LLVMOrcJITStackRef, callback_ctx::Ptr{Cvoid})
    JuliaContext() do ctx
        orc = OrcJIT(orc_ref)
        cc = Base.unsafe_pointer_to_objref(callback_ctx)::CallbackContext

        # 1. Lock job
        lock(cc.l_job)

        # 2. lookup if we are the first
        lock(l_outstanding)
        if in(cc, outstanding)
            delete!(outstanding, cc)
        else
            unlock(l_outstanding)
            unlock(cc.l_job)

            # 3. We are the second callback to run, but we raced the other one
            #    thus we return the addr from them.
            @assert cc.addr != C_NULL
            return UInt64(reinterpret(UInt, cc.addr))
        end
        unlock(l_outstanding)

        try
            thunk = Compiler._link(cc.job, Compiler._thunk(cc.job))
            mode = cc.job.config.params.mode
            use_primal = mode == API.DEM_ReverseModePrimal
            cc.addr = use_primal ? thunk.primal : thunk.adjoint

            # 4. Update the stub pointer to point to the recently compiled module
            set_stub!(orc, string(cc.stub), cc.addr)
        finally
            unlock(cc.l_job)
        end

        # 5. Return the address of the implementation, since we are going to call it now
        @assert cc.addr != C_NULL
        return UInt64(reinterpret(UInt, cc.addr))
	end
end

function get_trampoline(job)
    l_job = Base.ReentrantLock()

    cc = CallbackContext(job, gensym(:func), l_job)
    lock(l_outstanding)
    push!(outstanding, cc)
    unlock(l_outstanding)

    c_callback = @cfunction(callback, UInt64, (LLVM.API.LLVMOrcJITStackRef, Ptr{Cvoid}))

    orc = jit[]
    addr_adjoint = callback!(orc, c_callback, pointer_from_objref(cc))
    create_stub!(orc, string(cc.stub), addr_adjoint)

    return address(orc, string(cc.stub))
end


function resolver(name, ctx)
    name = unsafe_string(name)
    ptr = try
        ## Step 0: Should have already resolved it iff it was in the
        ##         same module
        ## Step 1: See if it's something known to the execution enging
        # TODO: Do we need to do this?
        # address(jit[], name)

        ## Step 2: Search the program symbols
        #
        # SearchForAddressOfSymbol expects an unmangled 'C' symbol name.
        # Iff we are on Darwin, strip the leading '_' off.
        @static if Sys.isapple()
            if name[1] == '_'
                name = name[2:end]
            end
        end

        for (k, v) in Compiler.JuliaGlobalNameMap
            if "ejl_"*k == name
                return unsafe_load(Base.reinterpret(Ptr{Ptr{Cvoid}}, Libdl.dlsym(hnd, k)))
            end
        end

        for (k, v) in Compiler.JuliaEnzymeNameMap
            if "ejl_"*k == name
                return Compiler.unsafe_to_ptr(v)
            end
        end

        LLVM.API.LLVMSearchForAddressOfSymbol(name)
        ## Step 4: Lookup in libatomic
        # TODO: Do we need to do this?
    catch ex
        @error "Enzyme: Lookup failed" jl_name exception=(ex, Base.catch_backtrace())
        C_NULL
    end
    if ptr === C_NULL
        @show name
        error("Enzyme: Symbol lookup failed. Aborting!")
    end

    return UInt64(reinterpret(UInt, ptr))
end

function add!(mod)
    return compile!(jit[], mod, @cfunction(resolver, UInt64, (Cstring, Ptr{Cvoid})))
end

function lookup(jitted_mod, name)
    return LLVM.addressin(jit[], jitted_mod, name)
end

end
