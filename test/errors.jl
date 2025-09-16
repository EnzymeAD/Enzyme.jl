using Enzyme, Test

Enzyme.API.printall!(true)
Enzyme.Compiler.DumpPreCheck[] = true
Enzyme.Compiler.DumpPreEnzyme[] = true
Enzyme.Compiler.DumpPostOpt[] = true

function array_square(x)
	bc = Base.Broadcast.Broadcasted(*, (2, x))
    dest = similar(x)
    if undefined1(bc.args) && undefined2(bc.args)
        for I in 1:length(x)
        	fx = Base.Fix2(undefined3, I)
            bc.f(fx(2), fx(x))
        end
    else
        @inbounds Base.copyto!(dest, bc)
    end
    return dest 
end

Enzyme.gradient(Reverse, array_square, [2.0])
