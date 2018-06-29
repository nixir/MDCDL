
# iterative shrinkage/thresholding algorithm
# solve a regularized convex optimization problem e.x. f(x) + g(x)
function ista(gradOfLossFcn::Function, prox::Function, x0; maxIterations::Integer=20, stepSize::Real=1.0, absTol::Real=1e-10,viewStatus::Bool=false)
    x = x0
    errx = Inf
    len = length(x0)
    for nItr = 1:maxIterations
        xprev = x
        x = prox(x - stepSize*gradOfLossFcn(x), stepSize)

        errx = vecnorm(x-xprev)^2/len

        if viewStatus
            println("\#Iterations $nItr: err = $errx ")
        end

        if errx <= absTol
            break
        end
    end
    (x, errx)
end

function fista(gradOfLossFcn::Function, prox::Function, x0; maxIterations::Integer=20, stepSize::Real=1.0, absTol::Real=1e-10,viewStatus::Bool=false)
    x = x0
    errx = Inf
    len = length(x0)
    y = x0
    t = 1
    for nItr = 1:maxIterations
        xprev = x
        tprev = t

        x = prox(y - stepSize*gradOfLossFcn(y), stepSize)
        t = (1 + sqrt(1+4*t^2)) / 2
        y = x + (tprev-1)/t * (x - xprev)

        errx = vecnorm(x-xprev)^2/len

        if viewStatus
            println("\#Iterations $nItr: err = $errx ")
        end

        if errx <= absTol
            break
        end
    end
    (x, errx)
end

function mlista(mlcsc::MDCDL.MultiLayerCsc, x, lambdas::Vector{T}; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false) where T <: Real
    const L = mlcsc.nLayers
    opD  = (l, v) -> stepSynthesisBank(mlcsc.dictionaries[l], v; inputMode="augumented")
    opDt = (l, v) -> stepAnalysisBank(mlcsc.dictionaries[l], v; outputMode="augumented")

    gamma = Vector(L+1)
    gamma[1] = x
    for l = 2:L+1
        gamma[l] = opDt(l-1, gamma[l-1])
    end


    for k = 1:maxIterations
        hgamma = Vector(L+1)
        hgamma[L+1] = gamma[L+1]
        for l = L:-1:2
            hgamma[l] = opD(l, gamma[l+1])
        end
        hgamma[1] = x

        for l = 1:L
            gamma[l+1] = softshrink(hgamma[l+1] - opDt(l, opD(l,hgamma[l+1]) - gamma[l]),lambdas[l])
            println("$k, $l")
        end
    end
    gamma[2:end]
end

function mlfista(mlcsc::MDCDL.MultiLayerCsc, x, lambdas::Vector{T}; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false) where T <: Real
    const L = mlcsc.nLayers
    opD  = (l, v) -> stepSynthesisBank(mlcsc.dictionaries[l], v; inputMode="augumented")
    opDt = (l, v) -> stepAnalysisBank(mlcsc.dictionaries[l], v; outputMode="augumented")

    gamma = Vector(L+1)
    gamma[1] = x
    for l = 2:L+1
        gamma[l] = opDt(l-1, gamma[l-1])
    end

    tk = 1.0
    z = gamma[L+1]
    for k = 1:maxIterations
        hgamma = Vector(L+1)
        hgamma[L+1] = z
        for l = L:-1:2
            hgamma[l] = opD(l, gamma[l+1])
        end
        hgamma[1] = x

        glp = gamma[L+1]
        for l = 1:L
            gamma[l+1] = softshrink(hgamma[l+1] - opDt(l, opD(l,hgamma[l+1]) - gamma[l]),lambdas[l])
            println("$k, $l")
        end
        tkprev = tk
        tk = (1 + sqrt(1+4*tkprev)) / 2

        z = gamma[L+1] + (tkprev - 1)/tk * (gamma[L+1] - glp)
    end
    gamma[2:end]
end

# FB-based primal-dual splitting algorithm
# N. Komodakis and J. Pesquet, "Playing with Duality," IEEE Signal Processing Magazine, vol. 32, no. 6, pp. 31--54, 2015.
# Algorithm 3
# argmin_{x} f(x) + g(Lx) + h(x)
# f() and g(): proxiable function
# h(x): smooth convex function having a Lipschitzian gradient
# L: linear operator
function pds(gradOfLossFcn::Function, proxF::Function, proxG::Function, linearOperator::Function, adjOfLinearOperator::Function, x0, v0=linearOperator(x0); maxIterations::Integer=20, tau::Real=1.0, sigma::Real=1.0, absTol::Real=1e-10)
    x = x0
    v = v0
    cproxG = (x, s) -> (x - proxG(x,s))
    for nItr = 1:maxIterations
        xprev = x
        p = proxF(x - tau*(gradOfLossFcn(x) + adjOfLinearOperator(v)), tau)
        q = cproxG(v + linearOperator(2.0*p - xprev), sigma^-1)

        # (x, v) <- (x, v) + λ((p,q) - (x,v)) for　λ == 1
        x = p
        v = q
    end
end

# TODO:勾配の計算を関数の内部で完結させる
function iht(gradOfLossFcn::Function, x0, K::Integer; maxIterations::Integer=20, absTol::Real=1e-10,viewStatus::Bool=false)
    x = x0
    errx = Inf
    len = length(x0)
    for nItr = 1:maxIterations
        xprev = x
        x = hardshrink(x - gradOfLossFcn(x), K)

        errx = vecnorm(x-xprev)^2/len

        if viewStatus
            println("\#Iterations $nItr: err = $errx ")
        end

        if errx <= absTol
            break
        end
    end
    (x, errx)
end

function iht(mlcsc::MDCDL.MultiLayerCsc, x, y0, K::Integer; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false)
    const len = length(y0)
    y = y0
    errx = Inf
    erry = Inf

    recx = synthesize(mlcsc, y)
    for itr = 1:maxIterations
        yprev = y
        y = hardshrink(y + analyze(mlcsc, x - recx), K)
        recx = synthesize(mlcsc, y)

        errx = vecnorm(x - recx)^2/len
        erry = vecnorm(y - yprev)^2

        if viewStatus
            println("\#Iterations $itr: err = $errx ")
        end

        if errx <= absTol
            break
        end
    end
    y
end

# prox of l2-norm
function proxOfL2(x, lambda::Real)
    max(1.0 - lambda/vecnorm(x), 0) * x
end

# prox. of mixed l1- and l2- norm
function groupshrink(x, lambda::Real)
    proxOfL2.(x, lambda)
end

# prox. of l1-norm
function softshrink(x, lambda::Real)
    @. max(1.0 - lambda / abs(x),0) * x
end

# prox. of nuclear norm.
function shrinkSingularValue(A::Matrix, lambda::Real)
    U, S, V = svd(A,thin=true)
    U * diagm(softshrink(S,lambda)) * V'
end

function boxProj(x, lowerBound::Real, upperBound::Real)
    max.(min.(x,upperBound),lowerBound)
end

function l2ballProj(x::T, radius::Real, centerVec::T) where T
    dist = vecnorm(x - centerVec)

    if dist <= radius
        x
    else
        centerVec + lambda/dist*(x - centerVec)
    end
end

function hardshrink(x::AbstractArray, k::Integer)
    szx = size(x)
    vx = vec(x)
    nzids = sortperm(abs.(vx), rev=true)[1:k]
    output = zeros(vx)
    foreach(nzids) do idx
        output[idx] = vx[idx]
    end
    reshape(output,szx...)
end
