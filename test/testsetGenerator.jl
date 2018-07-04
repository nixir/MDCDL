# 1-D CNSOLT

function cnsoltValidConfigSet1D(seed = 610763893)
      rng = srand(seed)

      allcfgs = vec([((df,), nch, (ord,)) for ord in 0:8, nch in 2:16, df in 1:4])
      cfgsTypeI  = filter(x ->  iseven(x[2]), allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  40 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> !iseven(x[2]), allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 40 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, nch, ord = cfg
            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end
end

# 2-D CNSOLT
function cnsoltValidConfigSet2D(seed = 625122358)
      rng = srand(seed)

      allcfgs = vec([(crdf.I, nch, crord.I .- 1) for crord in CartesianRange((4,4) .+ 1), nch in 2:20, crdf in CartesianRange((4,4))])

      cfgsTypeI  = filter(x ->  iseven(x[2]), allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  60 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> !iseven(x[2]), allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 60 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, nch, ord = cfg
            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end
end

# 3-D CNSOLT
function cnsoltValidConfigSet3D(seed = 229053284)
      rng = srand(seed)

      allcfgs = vec([(crdf.I, nch, crord.I .- 1) for crord in CartesianRange((2,2,2) .+ 1), nch in 2:10, crdf in CartesianRange((2,2,2))])

      cfgsTypeI  = filter(x ->  iseven(x[2]), allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  40 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> !iseven(x[2]), allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 40 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, nch, ord = cfg
            flag = true
            flag = flag && prod(df) <= nch
            flag = flag && !(isodd(nch) && any(isodd.(ord)))
      end
end

# 1-D RNSOLT
function rnsoltValidConfigSet1D(seed = 75389923)
      rng = srand(seed)

      allcfgs = vec([((df,), crnch.I, (ord,)) for ord in 0:6, crnch in CartesianRange((10,10)), df in 1:4])

      cfgsTypeI  = filter(x -> x[2][1] == x[2][2], allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  30 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> x[2][1] != x[2][2], allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 30 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, nch, ord = cfg
            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end
end

# 2-D RNSOLT
function rnsoltValidConfigSet2D(seed = 623122358)
      rng = srand(seed)

      allcfgs = vec([(crdf.I, crnch.I, crord.I .- 1) for crord in CartesianRange((4,4) .+ 1), crnch in CartesianRange((10,10)), crdf in CartesianRange((4,4))])

      cfgsTypeI  = filter(x -> x[2][1] == x[2][2], allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  50 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> x[2][1] != x[2][2], allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 50 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, nch, ord = cfg
            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end
end

# 3-D CNSOLT
function rnsoltValidConfigSet3D(seed = 385792992)
      rng = srand(seed)

      allcfgs = vec([(crdf.I, crnch.I, crord.I .- 1) for crord in CartesianRange((2,2,2) .+ 1), crnch in CartesianRange((6,6)), crdf in CartesianRange((2,2,2))])

      cfgsTypeI  = filter(x -> x[2][1] == x[2][2], allcfgs)
      cfgsTypeI  = randsubseq(cfgsTypeI,  50 / length( cfgsTypeI))
      cfgsTypeII = filter(x -> x[2][1] != x[2][2], allcfgs)
      cfgsTypeII = randsubseq(cfgsTypeII, 50 / length(cfgsTypeII))

      cfgs = vcat(cfgsTypeI..., cfgsTypeII...)

      filter(cfgs) do cfg
            df, nch, ord = cfg
            flag = true
            flag = flag && prod(df) <= sum(nch)
            flag = flag && ((nch[1] == nch[2]) || all(iseven.(ord)))
            flag = flag && (cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) && (fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
      end
end

nothing
