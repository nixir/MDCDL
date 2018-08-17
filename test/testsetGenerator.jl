# 1-D CNSOLT
using Random

function cnsoltValidConfigSet1D(seed = 610763893)
      rng = Random.seed!(seed)

      allcfgs = vec([((df,), (ord,), nch) for nch in 2:16, ord in 0:8, df in 1:4])
      cfgsTypeI  = filter(x ->  iseven(x[3]), allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  15 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> !iseven(x[3]), allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 20 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, ord, nch = cfg
            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end
end

# 2-D CNSOLT
function cnsoltValidConfigSet2D(seed = 625122358)
      rng = Random.seed!(seed)

      allcfgs = vec([(crdf.I, crord.I .- 1, nch) for nch in 2:20, crord in CartesianIndices((4,4) .+ 1), crdf in CartesianIndices((4,4))])

      cfgsTypeI  = filter(x ->  iseven(x[3]), allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  30 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> !iseven(x[3]), allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 40 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, ord, nch = cfg
            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end
end

# 3-D CNSOLT
function cnsoltValidConfigSet3D(seed = 229053284)
      rng = Random.seed!(seed)

      allcfgs = vec([(crdf.I, crord.I .- 1, nch) for nch in 2:10, crord in CartesianIndices((2,2,2) .+ 1), crdf in CartesianIndices((2,2,2))])

      cfgsTypeI  = filter(x ->  iseven(x[3]), allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  20 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> !iseven(x[3]), allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 40 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, ord, nch = cfg
            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end
end

# 1-D RNSOLT
function rnsoltValidConfigSet1D(seed = 75389923)
      rng = Random.seed!(seed)

      allcfgs = vec([((df,), (ord,), crnch.I) for crnch in CartesianIndices((10,10)), ord in 0:6, df in 1:4])

      cfgsTypeI  = filter(x -> x[3][1] == x[3][2], allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  17 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> x[3][1] != x[3][2], allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 20 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, ord, nch = cfg
            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end
end

# 2-D RNSOLT
function rnsoltValidConfigSet2D(seed = 623122358)
      rng = Random.seed!(seed)

      allcfgs = vec([(crdf.I, crord.I .- 1, crnch.I) for crnch in CartesianIndices((10,10)), crord in CartesianIndices((4,4) .+ 1), crdf in CartesianIndices((4,4))])

      cfgsTypeI  = filter(x -> x[3][1] == x[3][2], allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  20 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> x[3][1] != x[3][2], allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 60 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, ord, nch = cfg
            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end
end

# 3-D CNSOLT
function rnsoltValidConfigSet3D(seed = 385792992)
      rng = Random.seed!(seed)

      allcfgs = vec([(crdf.I, crord.I .- 1, crnch.I) for crnch in CartesianIndices((6,6)), crord in CartesianIndices((2,2,2) .+ 1), crdf in CartesianIndices((2,2,2))])

      cfgsTypeI  = filter(x -> x[3][1] == x[3][2], allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  20 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> x[3][1] != x[3][2], allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 70 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, ord, nch = cfg
            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end
end

nothing
