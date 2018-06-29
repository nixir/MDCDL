# module OrthonormalMatrixSystem

function mat2rotations(mtx::Matrix{T}) where T <: Real
    sz = size(mtx)
    P = sz[1]

    # res = Array{LinAlg.Givens}(round(Integer,P*(P-1)/2))
    res = Array{T}(fld(P*(P-1),2))

    nr = 1
    # for idx1 = 1:P-1
    #     for idx2 = idx1+1:P
    for idx1 = 1:P-1, idx2 = (idx1+1):P
        a = givens(mtx, idx1, idx2, idx1)
        g = a[1]

        res[nr] = atan2(g.s,g.c)
        nr += 1

        rtm = eye(T,P)
        rtm[g.i1, g.i1] =  g.c
        rtm[g.i1, g.i2] =  g.s
        rtm[g.i2, g.i1] = -g.s
        rtm[g.i2, g.i2] =  g.c
        mtx = rtm*mtx
        # end
    end
    (res, round.(diag(mtx)))
end

function rotations2mat(angs::Array{T}, sig::Array{T}) where T <: Real
    L = length(angs)
    P = round(Integer, (1 + sqrt(1+8*L)) / 2)
    mtx = eye(T,P)

    nr = 1
    # for idx1 = 1:P-1
    #     for idx2 = idx1+1:P
    for idx1 = 1:P-1, idx2 = (idx1+1):P
            c = cos(angs[nr])
            s = sin(angs[nr])

            rtm = eye(T,P)
            rtm[idx1, idx1] =  c
            rtm[idx1, idx2] = -s
            rtm[idx2, idx1] =  s
            rtm[idx2, idx2] =  c
            mtx = mtx*rtm

            nr += 1
    #     end
    end
    mtx * diagm(sig)
end

# end
