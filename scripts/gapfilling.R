library(REddyProc)

silenceR <- function(){
  sink(file("./.r2py_out_sink.log", "w"), type="message")
}

sink.reset <- function(){
    for(i in seq_len(sink.number())){
        sink(NULL)
    }
}


REddyProcsMDSGapFill = function(EddyData, climcols=c('Tair', 'VPD'), fluxcols=c('NEE', 'LE'),
                                Lat=43, Lon=47.5, verbosity=0, isCleaned=True){
  cols = c(climcols, fluxcols)

  #EddyData <- EddyData %>% filterLongRuns("NEE")
  
  EddyData$DateTime <- as.POSIXct(as.character(EddyData$TIMESTAMP), format=c('%Y%m%d%H%M'), tz='GMT')
  #print(head(EddyData %>% dplyr::select(DateTime, TIMESTAMP, NEE, Ustar)))
  if (verbosity>1) print(EddyData %>% dplyr::select(DateTime, TIMESTAMP, NEE, Ustar) %>% summary())
  #EddyData <- EddyData %>% 
  #  dplyr::mutate(DateTime  = as.POSIXct(as.character(TIMESTAMP_END), format=c('%Y%m%d%H%M'), tz='GMT'),#need to be carfeful with tz but needed to avoid bugs
  #           Year = lubridate::year(DateTime),
  #           DoY = lubridate::yday(DateTime),  
  #           time = strftime(DateTime, format="%H:%M"),
  #           Hour = lubridate::hour(DateTime) + lubridate::minute(DateTime)/60,
  #           Hour = ifelse(Hour == 0, 24, Hour),
  #           DoY = ifelse(Hour == 24, DoY-1, DoY),
  #           monthnb = lubridate::month(DateTime))

  #+++ Initalize R5 reference class sEddyProc for processing of eddy data
  #+++ with all variables needed for processing later
  EProc <- sEddyProc$new(
    "FluxSite", EddyData, union(c('Rg', 'Ustar'), unlist(cols)))
  
  EProc$sSetLocationInfo(LatDeg = Lat, LongDeg = Lon, TimeZoneHour = 1)
  #add Ustar threshold in dataset
  #(uStarTh <- EProc$sEstUstarThold()$uStarTh)
  
  EProc$sEstimateUstarScenarios(
    nSample = 100L, 
    probs = c(0.05, 0.5, 0.95))
  EProc$sGetUstarScenarios()
  (uStarThAgg <- EProc$sGetEstimatedUstarThresholdDistribution())
  EProc$useSeaonsalUStarThresholds()
  
  # inspect the changed thresholds to be used
  #uStarSuffixes <- colnames(EProc$sGetUstarScenarios())[-1]
  #print(uStarSuffixes)
  
  #Gap-filling of Tair, VPD, NEE and LE
  for (c in climcols){
    EProc$sMDSGapFill(c, FillAll=FALSE, isVerbose=F)
  }
  for (c in fluxcols){
    #EProc$sMDSGapFill(c, FillAll=FALSE, isVerbose=F)
    #EProc$sMDSGapFillAfterUstar(c)
    EProc$sMDSGapFillUStarScens(c)#, FillAll=FALSE, isVerbose=F)
  }
  #EProc$sFillVPDFromDew()  # fill longer gaps still present in VPD_f (important for day-time partition)
  
  #EProc$sMRFluxPartition()
  EProc$sMRFluxPartitionUStarScens()

  #++ Export gap filled and partitioned data to standard data frame
  FilledEddyData <- EProc$sExportResults()
  FilledEddyData$TIMESTAMP <- EddyData$DateTime
  #FilledEddyData$uStarThAgg <- uStarThAgg
  print(uStarThAgg)
  #print(uStarTh)
  #FilledEddyData <- cbind(EddyData, FilledEddyData)
  #FilledEddyData$TIMESTAMP <- lubridate::ymd_hms(FilledEddyData$TIMESTAMP)
  #FilledEddyData <- dplyr::select(FilledEddyData, -c(DateTime))
  
  return(FilledEddyData)
}
