function laggedvariance(X, Y, lag)
    n = length(X)
    if lag > 0
        σγ = cov(X[(1+lag):n], Y[1:(n-lag)])
    else
        σγ = cov(X[1:(n+lag)], Y[(1-lag):n])
    end
    return σγ
end

function FinkelsteinandSims2001(X, Y, m=200)
    n = length(X)
    σF = (1/(n^(1/2))) * (
        sum([laggedvariance(X, X, i) for i in range(-m, m)] .* 
        [laggedvariance(Y, Y, i) for i in range(-m, m)]) +
        sum([laggedvariance(X, Y, i) for i in range(-m, m)] .*
        [laggedvariance(Y, X, i) for i in range(-m, m)]))^(1/2)
    return σF
end