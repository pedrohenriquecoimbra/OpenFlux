using CSV, DataFrames, Dates, Statistics, StringEncodings, Printf

function numtodate(x)
    try
        x = Float64(x)
        y = ceil(Int, log10(x))
        dateformatraw = "yyyymmddHHMMSSss"
        dateformatfit = first(dateformatraw, y) * "." * last(dateformatraw, length(dateformatraw) - y)
        xstr = string(@sprintf "%.2f" Float64(x))
        return DateTime(xstr, DateFormat(dateformatfit))
    catch e
        return missing
    end
end

function load_each_file(path, t0="19700101000000", csvargs=Dict())::DataFrame # header, t0=nothing, acqrt=nothing)
    # if no TIMESTAMP, add it using t0 (given beforehand from filename) and Hz (or period saved in file)
    #t0 = ("t0" in keys(csvargs) ? csvargs["t0"] : nothing)
    acqrt = ("acqrt" in keys(csvargs) ? csvargs["acqrt"] : 20)
    header = ("header" in keys(csvargs) ? csvargs["header"] : 1)
    delim = ("delim" in keys(csvargs) ? csvargs["delim"] : nothing)
    dateform = ("dateform" in keys(csvargs) ? DateFormat(csvargs["dateform"]) : dateformat"yyyy-mm-ddTHH:MM:SS.ss")

    if eltype(header) <: Number
        skipto = header + 1
    else
        skipto = 1
    end

    if !isnothing(delim) && delim == "\t"
        path = IOBuffer(encode(replace(read(path, String), r" +" => ","), "UTF-8"))
        #path = IOBuffer(replace(read(path), r"\s*" => UInt8(','), UInt8('2') => UInt8('5')))
        delim = ","
    end

    for i in skipto:skipto+15
        df_ = CSV.read(path, DataFrame; skipto=i, limit=1, stringtype=String) # header=header
        if isempty(df_) || count([isa.(x, String) || isnothing(x) || ismissing(x) for x in df_[1, :]]) >= length(df_[1, :]) * 0.5
            skipto = i + 1
        else
            break
        end
    end

    dft = CSV.File(path, header=header, delim=delim, skipto=skipto, typemap=Dict(Int => Float64, Missing => Float64), stringtype=String, dateformat=dateform) |> DataFrame

    if isempty(dft)
        return dft
    end;

    #if !isnothing(delim) && delim == "\t"
    #    dft = eachline.(dft).replace(' ', '')
    #end

    # DROP IF ALL IS MISSING
    dft = select(dft, eltype.(eachcol(dft)) .!= Missing)

    if (!("TIMESTAMP" in names(dft)) || eltype(:TIMESTAMP) == Missing) && !isnothing(t0) && !isnothing(acqrt)
        N = length(dft[!, 1])
        dft.TIMESTAMP = [DateTime(t0, dateform) - Millisecond((N - i) * 1000 / acqrt) for i in eachindex(dft[!, 1])]
    end

    if eltype(dft.TIMESTAMP) <: String
        dft.TIMESTAMP = map(x->(v = tryparse(Float64, x); isnull(v) ? 0.0 : get(v)), dft.TIMESTAMP)
    end;
    
    dft = filter(:TIMESTAMP => t -> !ismissing(t), dft)
    
    if eltype(dft.TIMESTAMP) <: Number
        dft.TIMESTAMP = numtodate.(dft.TIMESTAMP)
        #df.TIMESTAMP = [DateTime(string(@sprintf "%.2f" d), dateformat"yyyymmdd.HHMMSSss") for d ∈ df.TIMESTAMP]
    end;

    return dft
end

function load_files(list_paths, list_times=nothing, csvargs=Dict(), output_path=nothing)::DataFrame
    if !isnothing(output_path)
        log = rsplit(output_path, ".", limit=2)[1] * ".log"
    else
        log = pop!(csvargs, "log", nothing) #"C:/Users/phherigcoimb/Desktop/INRAE_longfiles/ICOS/FR-Gri/output/BM/log.log"
    end;
    if !isnothing(log)
        open(log, "a+") do io
            write(io, "\n" * Dates.format(now(), "dd.mm.yyyy HH:MM") * "\n")
        end;
    end;

    df = DataFrame()
    #dateform = ("dateform" in keys(csvargs) ? DateFormat(csvargs["dateform"]) : dateformat"yyyymmddHHMMSS")

    for i in eachindex(list_paths)
        try
            if isempty(df)
                df = load_each_file(list_paths[i], (isnothing(list_times) ? list_times : list_times[i]), csvargs)
            else
                append!(df, load_each_file(list_paths[i], (isnothing(list_times) ? list_times : list_times[i]), csvargs), cols=:union)
            end
            if !isnothing(log)
                open(log, "a+") do io
                    write(io, list_paths[i] * ", success\n")
                end;
            end;
        catch e
            if !isnothing(log)
                open(log, "a+") do io
                    write(io, list_paths[i] * ", error, " * string(e) * "\n")
                end;
            end;
        end;
    end;

    for i in 1:ncol(df)
        #if eltype(df[!, i]) <: Union{Any}
        #    df[!, i] = string.(df[!, i])
        #end
        if eltype(df[!, i]) <: Union{String}
            df[!, i] = map(d -> (v = tryparse(Float64, d); isnothing(v) ? missing : v), df[!, i])
        end
    end

    # DROP IF ALL IS MISSING
    #df = select(df, [!all(isnothing.(c)) for c in eachcol(df)])

    if isempty(df)
        return df
    end

    #df = select(df, eltype.(eachcol(df)) .!= Missing)

    #df = filter(:TIMESTAMP => t -> !ismissing(t), df);

    #=
    if eltype(df.TIMESTAMP) <: Number
        df.TIMESTAMP = numtodate.(df.TIMESTAMP)
        #df.TIMESTAMP = [DateTime(string(@sprintf "%.2f" d), dateformat"yyyymmdd.HHMMSSss") for d ∈ df.TIMESTAMP]
    end
    if eltype(df.TIMESTAMP) <: Number && all(x -> x > 1970e10, df.TIMESTAMP) && any(x -> x < 2070e10, df.TIMESTAMP)
        df.TIMESTAMP = [DateTime(string(@sprintf "%.2f" d), dateformat"yyyymmddHHMMSS.ss") for d ∈ df.TIMESTAMP]
        #df.TIMESTAMP = DateTime.(string.(floor.(df.TIMESTAMP)), dateformat"yyyymmddHHMMSS")
        #_year, Δ = divrem.(df.TIMESTAMP, 1)
        #df.TIMESTAMP = [DateTime(string(d), dateform) for d ∈ _year] #+ microsecond.(10*Δ)
    end
    if eltype(df.TIMESTAMP) <: Number && all(x -> x > 1970e8, df.TIMESTAMP) && any(x -> x < 2070e8, df.TIMESTAMP)
        df.TIMESTAMP = [DateTime(string(@sprintf "%.2f" d), dateformat"yyyymmddHHMM.SS") for d ∈ df.TIMESTAMP]
    end
    =#
    #df.TIMESTAMP = string.(Int64.(df.TIMESTAMP));
    #df.TIMESTAMP = DateTime.(df.TIMESTAMP, dateformat"yyyymmddHHMMSS");

    if !isnothing(output_path)
        CSV.write(output_path, df)
    end

    return df
end