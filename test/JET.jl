using Enzyme

# basic
# -----

union_result(cond, x) = cond ? x : 0
let result = report_enzyme(Active, true, Active(1.0)) do cond, x
        union_result(cond, x) * x
    end
    reports = Enzyme.get_reports(result)
    @test length(reports) == 1
    report = reports[1]
    @test report isa Enzyme.UnionResultReport
    @test report.tt === Tuple{typeof(union_result),Bool,Float64}
    @test report.rt === Union{Int,Float64}
end
let result = @report_enzyme autodiff(Active, true, Active(1.0)) do cond, x
        union_result(cond, x) * x
    end
    reports = Enzyme.get_reports(result)
    @test length(reports) == 1
    report = reports[1]
    @test report isa Enzyme.UnionResultReport
    @test report.tt === Tuple{typeof(union_result),Bool,Float64}
    @test report.rt === Union{Int,Float64}
end

# activity analysis
# -----------------

let result = report_enzyme(Active, true, Active(1.0)) do cond, x
        union_result(cond, x) # trivially unused
        return x * x
    end
    @test isempty(Enzyme.get_reports(result))
end
let result = report_enzyme(Active, true, Active(1.0)) do cond, x
        r = Ref{Any}(union_result(cond, x)) # will be DCE-ed
        return x * x
    end
    @test_broken isempty(Enzyme.get_reports(result))
end

# invalidation
# ------------

union_result(cond, x) = cond ? x : zero(x)
let result = report_enzyme(Active, true, Active(1.0)) do cond, x
        union_result(cond, x) * x
    end
    @test isempty(Enzyme.get_reports(result))
end

# Test.jl integration

test_enzyme(Active, true, Active(1.0)) do cond, x
    union_result(cond, x) * x
end
@test_enzyme autodiff(Active, true, Active(1.0)) do cond, x
    union_result(cond, x) * x
end
