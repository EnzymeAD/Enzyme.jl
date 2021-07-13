module JIT

using LLVM
import LLVM: TargetMachine

import GPUCompiler: CompilerJob
import ..Compiler

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

    tm[] = LLVM.JITTargetMachine(optlevel=optlevel)
    LLVM.asm_verbosity!(tm[], true)

    jit[] = OrcJIT(tm[]) # takes ownership of tm
    atexit() do
        dispose(jit[])
    end
end

mutable struct CallbackContext
    tag::Symbol
    job::CompilerJob
    stub::Symbol
    l_job::ReentrantLock
    addr::Ptr{Cvoid}
    CallbackContext(tag, job, stub, l_job) = new(tag, job, stub, l_job, C_NULL)
end

const l_outstanding = Base.ReentrantLock()
const outstanding = Dict{Symbol, Tuple{CallbackContext, CallbackContext}}()

# Setup the lazy callback for creating a module
function callback(orc_ref::LLVM.API.LLVMOrcJITStackRef, callback_ctx::Ptr{Cvoid})
    orc = OrcJIT(orc_ref)
    cc = Base.unsafe_pointer_to_objref(callback_ctx)::CallbackContext

    # 1. Lock job
    lock(cc.l_job)

    # 2. lookup if we are the first
    lock(l_outstanding)
    if haskey(outstanding, cc.tag)
        ccs = outstanding[cc.tag]
        delete!(outstanding, cc.tag)
    else
        ccs = nothing
    end
    unlock(l_outstanding)

    # 3. We are the second callback to run, but we raced the other one
    #    thus we return the addr from them.
    if ccs === nothing
        unlock(cc.l_job)
        @assert cc.addr != C_NULL
        return UInt64(reinterpret(UInt, cc.addr))
    end

    cc_adjoint, cc_primal = ccs
    try
        thunk = Compiler._link(cc.job, Compiler._thunk(cc.job))::Compiler.Thunk
        cc_adjoint.addr = thunk.adjoint
        cc_primal.addr  = thunk.primal

        # 4. Update the stub pointer to point to the recently compiled module
        set_stub!(orc, string(cc_adjoint.stub), thunk.adjoint)
        set_stub!(orc, string(cc_primal.stub),  thunk.primal)
    finally
        unlock(cc.l_job)
    end

    # 5. Return the address of the implementation, since we are going to call it now
    @assert cc.addr != C_NULL
    return UInt64(reinterpret(UInt, cc.addr))
end

function get_trampoline(job)
    tag = gensym(:tag)
    l_job = Base.ReentrantLock()
    cc_adjoint = CallbackContext(tag, job, gensym(:adjoint), l_job)
    cc_primal = CallbackContext(tag, job, gensym(:primal), l_job)
    lock(l_outstanding) do
        outstanding[tag] = (cc_adjoint, cc_primal)
    end

    c_callback = @cfunction(callback, UInt64, (LLVM.API.LLVMOrcJITStackRef, Ptr{Cvoid}))

    orc = jit[]
    addr_adjoint = callback!(orc, c_callback, pointer_from_objref(cc_adjoint))
    create_stub!(orc, string(cc_adjoint.stub), addr_adjoint)

    addr_primal = callback!(orc, c_callback, pointer_from_objref(cc_primal))
    create_stub!(orc, string(cc_primal.stub), addr_primal)

    return address(orc, string(cc_adjoint.stub))#, address(orc, string(cc_primal.stub))
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
