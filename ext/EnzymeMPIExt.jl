module EnzymeMPIExt

using MPI
using Enzyme

import Enzyme.EnzymeCore: EnzymeRules

function EnzymeRules.forward(config, ::Const{typeof(MPI.Allreduce!)}, rt, v, op::Const, comm::Const)
    op = op.val
    comm = comm.val

    if !(op == MPI.SUM || op == +)
        error("Forward mode MPI.Allreduce! is only implemented for MPI.SUM.")
    end

    if EnzymeRules.needs_primal(config)
        MPI.Allreduce!(v.val, op, comm)
    end

    if EnzymeRules.width(config) == 1
        MPI.Allreduce!(v.dval, op, comm)
    else
        # would be nice to use MPI non-blocking collectives
        foreach(v.dval) do dval
            MPI.Allreduce!(dval, op, comm)
        end
    end

    if EnzymeRules.needs_primal(config)
        return v
    else
        return v.dval
    end
end


end
