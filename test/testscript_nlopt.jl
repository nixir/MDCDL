using JuMP
using NLopt

m = Model(solver=NLoptSolver(algorithm=:LD_MMA))

nVars = 10;

x = rand(nVars)

@variable(m, v[1:nVars])

A = rand(nVars,nVars)
r = rand(nVars)
function myfunc(vc...)
    v = collect(vc)
    vecnorm(r - A*v)
end
JuMP.register(m, :myfunc, nVars, myfunc, autodiff=true)

args = rstrip(mapfoldl(x -> "v[$x],", *, 1:nVars),',')
objFcnExp = parse("@NLobjective(m, Min, myfunc($args))")
eval(objFcnExp)

setvalue(v, ones(nVars)/sqrt(nVars))

status = solve(m)

println("got ", getobjectivevalue(m), " at ", [getvalue(v[1]),getvalue(v[2]),getvalue(v[3])])
