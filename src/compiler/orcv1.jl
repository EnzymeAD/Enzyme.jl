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

    if haskey(ENV, "ENABLE_GDBLISTENER")
        LLVM.register!(jit[], LLVM.GDBRegistrationListener())
    end
    atexit() do
        dispose(jit[])
    end
end

mutable struct CallbackContext
    job::CompilerJob
    stub::String
    compiled::Bool
end

const outstanding = IdDict{CallbackContext, Nothing}()

# Setup the lazy callback for creating a module
function callback(orc_ref::LLVM.API.LLVMOrcJITStackRef, callback_ctx::Ptr{Cvoid})
    orc = OrcJIT(orc_ref)
    cc = Base.unsafe_pointer_to_objref(callback_ctx)::CallbackContext

    @assert !cc.compiled
    job = cc.job

    thunk = Compiler._link(job, Compiler._thunk(job))
    cc.compiled = true
    delete!(outstanding, cc)

    # 4. Update the stub pointer to point to the recently compiled module
    set_stub!(orc, cc.stub, thunk.adjoint)

    # 5. Return the address of tie implementation, since we are going to call it now
    ptr = thunk.adjoint
    return UInt64(reinterpret(UInt, ptr))
end

function get_trampoline(job)
    cc = CallbackContext(job, String(gensym(:trampoline)), false)
    outstanding[cc] = nothing

    c_callback = @cfunction(callback, UInt64, (LLVM.API.LLVMOrcJITStackRef, Ptr{Cvoid}))

    orc = jit[]
    initial_addr = callback!(orc, c_callback, pointer_from_objref(cc))
    create_stub!(orc, cc.stub, initial_addr)
    return address(orc, cc.stub)
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
