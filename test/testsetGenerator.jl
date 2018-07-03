# 1-D CNSOLT

function cnsoltConfigSet1D(seed = 610763893)
      rng = srand(seed)

      cnsoltConfigs1D = vec([((df,), nch, (ord,)) for ord in 0:8, nch in 2:16, df in 1:4])
      cnsoltConfigs1D = randsubseq(cnsoltConfigs1D, 0.12)
      # cnsoltConfigs1D = vcat([((n,), prod(nch), 0) for nch in 2:8, n in 1:4]..., cnsoltConfigs1D...)
      cnsoltConfigs1D = filter(cnsoltConfigs1D) do cfg
            df, nch, ord = cfg

            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end

      cnsoltConfigs1D
end

# cnsoltConfigs1DTypeI  = filter(x -> iseven(x[2]), cnsoltConfigs1D)
# cnsoltConfigs1DTypeII = filter(x ->  isodd(x[2]), cnsoltConfigs1D)
#
# count(x->sum(x[3]) == 0, cnsoltConfigs1DTypeI)
# count(x->sum(x[3]) == 0, cnsoltConfigs1DTypeII)

# 2-D CNSOLT
function cnsoltConfigSet2D(seed = 625122358)
      rng = srand(seed)

      cnsoltConfigs2D = vec([(crdf.I, nch, crord.I .- 1) for crord in CartesianRange((6,6) .+ 1), nch in 2:20, crdf in CartesianRange((4,4))])
      cnsoltConfigs2D = randsubseq(cnsoltConfigs2D, 0.007)
      # cnsoltConfigs2D = vcat([((n,n), nch, (0,0)) for nch in 2:10, n in 1:4]..., cnsoltConfigs2D...)
      cnsoltConfigs2D = filter(cnsoltConfigs2D) do cfg
            df, nch, ord = cfg

            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end

      cnsoltConfigs2D
end

# cnsoltConfigs2DTypeI  = filter(x -> iseven(x[2]), cnsoltConfigs2D)
# cnsoltConfigs2DTypeII = filter(x ->  isodd(x[2]), cnsoltConfigs2D)
#
# count(x->sum(x[3]) == 0, cnsoltConfigs2DTypeI)
# count(x->sum(x[3]) == 0, cnsoltConfigs2DTypeII)

# 3-D CNSOLT
function cnsoltConfigSet3D(seed = 229053284)
      rng = srand(seed)

      cnsoltConfigs3D = vec([(crdf.I, nch, crord.I .- 1) for crord in CartesianRange((4,4,4) .+ 1), nch in 2:20, crdf in CartesianRange((2,2,2))])
      cnsoltConfigs3D = randsubseq(cnsoltConfigs3D, 0.005)
      cnsoltConfigs3D = filter(cnsoltConfigs3D) do cfg
            df, nch, ord = cfg

            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
            flag = flag && prod(ord) <= 16
      end

      cnsoltConfigs3D
end

# cnsoltConfigs3DTypeI  = filter(x -> iseven(x[2]), cnsoltConfigs3D)
# cnsoltConfigs3DTypeII = filter(x ->  isodd(x[2]), cnsoltConfigs3D)
#
# count(x->sum(x[3]) == 0, cnsoltConfigs3DTypeI)
# count(x->sum(x[3]) == 0, cnsoltConfigs3DTypeII)

# 1-D RNSOLT

function rnsoltConfigSet1D(seed = 75389923)
      rng = srand(seed)

      rnsoltConfigs1D = vec([((df,), crnch.I, (ord,)) for ord in 0:8, crnch in CartesianRange((8,8)), df in 1:4])
      rnsoltConfigs1D = randsubseq(rnsoltConfigs1D, 0.01)
      # rnsoltConfigs1D = vcat([((n,), prod(nch), 0) for nch in 2:8, n in 1:4]..., rnsoltConfigs1D...)
      rnsoltConfigs1D = filter(rnsoltConfigs1D) do cfg
            df, nch, ord = cfg

            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end

      rnsoltConfigs1D
end

# rnsoltConfigs1DTypeI  = filter(x -> iseven(x[2]), rnsoltConfigs1D)
# rnsoltConfigs1DTypeII = filter(x ->  isodd(x[2]), rnsoltConfigs1D)
#
# count(x->sum(x[3]) == 0, rnsoltConfigs1DTypeI)
# count(x->sum(x[3]) == 0, rnsoltConfigs1DTypeII)

# 2-D RNSOLT
function rnsoltConfigSet2D(seed = 623122358)
      rng = srand(seed)

      rnsoltConfigs2D = vec([(crdf.I, crnch.I, crord.I .- 1) for crord in CartesianRange((6,6) .+ 1), crnch in CartesianRange((10,10)), crdf in CartesianRange((4,4))])
      rnsoltConfigs2D = randsubseq(rnsoltConfigs2D, 0.001)
      # rnsoltConfigs2D = vcat([((n,n), nch, (0,0)) for nch in 2:10, n in 1:4]..., rnsoltConfigs2D...)
      rnsoltConfigs2D = filter(rnsoltConfigs2D) do cfg
            df, nch, ord = cfg

            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end

      rnsoltConfigs2D
end

# rnsoltConfigs2DTypeI  = filter(x -> iseven(x[2]), rnsoltConfigs2D)
# rnsoltConfigs2DTypeII = filter(x ->  isodd(x[2]), rnsoltConfigs2D)
#
# count(x->sum(x[3]) == 0, rnsoltConfigs2DTypeI)
# count(x->sum(x[3]) == 0, rnsoltConfigs2DTypeII)

# 3-D CNSOLT
function rnsoltConfigSet3D(seed = 385792992)
      rng = srand(seed)

      rnsoltConfigs3D = vec([(crdf.I, crnch.I, crord.I .- 1) for crord in CartesianRange((4,4,4) .+ 1), crnch in CartesianRange((10,10)), crdf in CartesianRange((2,2,2))])
      rnsoltConfigs3D = randsubseq(rnsoltConfigs3D, 0.0005)
      rnsoltConfigs3D = filter(rnsoltConfigs3D) do cfg
            df, nch, ord = cfg

            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
            flag = flag && prod(ord) <= 16
      end

      rnsoltConfigs3D
end

# rnsoltConfigs3DTypeI  = filter(x -> iseven(x[2]), rnsoltConfigs3D)
# rnsoltConfigs3DTypeII = filter(x ->  isodd(x[2]), rnsoltConfigs3D)
#
# count(x->sum(x[3]) == 0, rnsoltConfigs3DTypeI)
# count(x->sum(x[3]) == 0, rnsoltConfigs3DTypeII)

nothing
