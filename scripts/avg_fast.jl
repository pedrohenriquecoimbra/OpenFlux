using CSV, DataFrames, Dates, Statistics, StringEncodings, Printf
include("uncertainties.jl")
include("load_files.jl")

function eddycov_average(corrected=[nothing, nothing, Dict()], uncorrected=[nothing, nothing, nothing], round_min=30, output_path=nothing)
    list_paths, list_times, csvargs = corrected
    list_paths2, list_times2, csvargs2 = uncorrected
    csvargs2 = (isnothing(csvargs2) ? csvargs : csvargs2)
    list_paths2 = (isnothing(list_paths2) ? list_paths : list_paths2)
    list_times2 = (isnothing(list_times2) ? list_times : list_times2)

    df = load_files(list_paths, list_times, csvargs);
    
    #if !("co2" in names(df)) #∉
    #    return nothing
    #end

    dfu = load_files(list_paths2, list_times2, csvargs2);
    select!(dfu, ["TIMESTAMP", "u", "v"])
    rename!(dfu, Dict(:u => :u_unc, :v => :v_unc))
    df = leftjoin(df, dfu, on = :TIMESTAMP)
    
    df.TIMESTAMP = ceil.(df.TIMESTAMP, Minute(round_min));
    #df.ts = df.t;

    ustar(u, v, w) = (cov(u, w)^2 + cov(v, w)^2)^(0.25);
    ustar(cov_uw, cov_vw) = (cov_uw^2 + cov_vw^2)^(0.25);
    wd(u, v, offset=-110) = mod((180 - (atan(v, u) * 180 / pi) + offset + 360), 360);
    #ol(Ustar, w, ts) = - mean(ts) * (Ustar^3) / (0.41 * 9.81 * cov(w, ts));
    ol(u, v, w, ts) = - mean(ts) * (ustar(u, v, w)^3) / (0.41 * 9.81 * cov(w, ts));
    
    df = groupby(df, :TIMESTAMP, sort=true);
    df = combine(df, 
    [:u, :v, :w, :co2, :h2o, :t_sonic] .=> mean∘skipmissing,
    [:co2, :w] => cov => :cov_wco2,
    [:h2o, :w] => cov => :cov_wh2o,
    [:w, :t_sonic] => cov => :cov_wts,
    [:u, :w] => cov => :cov_uw,
    [:v, :w] => cov => :cov_vw,
    [:u, :v, :w] => ustar => :us,
    [:u, :v, :w, :co2, :h2o, :t_sonic] .=> std .=> [:sigmau, :sigmav, :sigmaw, :sigmaco2, :sigmah2o, :sigmats], 
    [:u, :v, :w, :t_sonic] => ol => :ol,
    [:u_unc, :v_unc] .=> mean∘skipmissing,
    renamecols=false);
    
    df.wd = wd.(df.u_unc, df.v_unc)
    #[:u_unc, :v_unc, :w_unc] .=> mean∘skipmissing .=> [:u_unrotated, :v_unrotated, :w_unrotated],
    #df.wd = wd(df.u_unrotated, df.v_unrotated)
        
    df = sort!(df, :TIMESTAMP)

    if !isnothing(output_path)
        CSV.write(output_path, df);
    end

    return df
end


function randomuncertainty(files_par, round_min=30, output_path=nothing)
    list_paths, list_times, csvargs = files_par
    df = load_files(list_paths, list_times, csvargs);
    
    if !("co2" in names(df)) #∉
        return nothing
    end
    
    df.TIMESTAMP = ceil.(df.TIMESTAMP, Minute(round_min));
    
    df = groupby(df, :TIMESTAMP, sort=true);
    df = combine(df, 
    [:w, :co2] => FinkelsteinandSims2001 => :wco2_random_uncertainty,
    renamecols=false);
        
    df = sort!(df, :TIMESTAMP)

    if !isnothing(output_path)
        CSV.write(output_path, df);
    end

    return df
end

function biomet_average(list_paths, list_times=nothing, csvargs=Dict(), round_min=30, output_path=nothing)
    df = load_files(list_paths, list_times, csvargs)

    df.TIMESTAMP = ceil.(df.TIMESTAMP, Minute(round_min))

    df = groupby(df, :TIMESTAMP, sort=true)
    df = combine(df, valuecols(df) .=> mean ∘ skipmissing, renamecols=false)

    kp_cols = [
        if !occursin(r"_IU_", n)
            n
        end for n in names(df)
    ]
    kp_cols = kp_cols[kp_cols.!=nothing]
    df = select(df, kp_cols)

    df = sort!(df, :TIMESTAMP)

    if !isnothing(output_path)
        CSV.write(output_path, df)
    end

    return df
end


function biomet_all(list_paths, list_times=nothing, csvargs=nothing, round_min=30, output_path=nothing)
    df = DataFrame()
    #if isnothing(header)
    #    header = repeat([1], outer=length(list_paths))
    #end
    for i in eachindex(list_paths)
        try
            df_ = biomet_average(list_paths[i], (isnothing(list_times) ? list_times : list_times[i]), (eltype(csvargs) <: Pair ? csvargs : csvargs[i]), round_min, nothing)
            if isempty(df)
                df = df_
            else
                df = outerjoin(df, df_, on=:TIMESTAMP, makeunique=true)
            end
            df_ = nothing
        catch e
            open("C:/Users/phherigcoimb/Desktop/INRAE_longfiles/ICOS/FR-Gri/output/BM/logy.log", "a+") do io
                write(io, string(i) * ", error, " * string(e) * "\n")
            end
        end
    end

    df = sort!(df, :TIMESTAMP)
    
    #average_similar_columns(df, rsplit(output_path, ".", limit=2)[1] * ".avg.csv")
    
    if !isnothing(output_path)
        CSV.write(output_path, df)
    end
    return df
end

function average_similar_columns(data, output_path=nothing)
    if eltype(data) <: Char
        df = CSV.read(data, DataFrame;)
    else
        df = data
    end;

    df = sort!(df, :TIMESTAMP)

    dfa = DataFrame("TIMESTAMP" => df.TIMESTAMP)
    tt = hcat(join.(filter.(x -> isnothing(tryparse(Float64, string(x))), split.(names(df), "_")), "_"), names(df))
    kk = unique(tt[:, 1])
    filter!(!=("TIMESTAMP"), kk)
    for k in kk
        dfa[!, k] = passmissing.(x -> mean(x)).(eachrow(df[!, tt[in([k]).(tt[:, 1]), 2]]))
    end
    
    if !isnothing(output_path)
        CSV.write(output_path, dfa)
    end
    return dfa
end


function agg_files(list_paths, fn=nothing, output_path=nothing, csvkwargs=nothing)
    df = DataFrame
    fc = [getfield(Main, Symbol(f)) for f ∈ split(fn, "∘")]
    ls = list_paths
    
    for f in fc
        dfs = []
        for i in eachindex(ls)
            if i == 1
                df_ = CSV.read(ls[i], DataFrame; stringtype=String);
            else
                df_ = f(df_, CSV.read(ls[i], DataFrame; stringtype=String));
            end;
            push!(dfs, df_);
        end;
        ls = dfs;
    end;

    if !isnothing(output_path)
        CSV.write(output_path, df);
    end
    return df
end
