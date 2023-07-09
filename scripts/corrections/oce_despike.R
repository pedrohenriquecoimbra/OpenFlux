#taken from oce 
#https://github.com/cran/oce/blob/cee564f9a15f1572909c799df7e5f301f093df9c/R/misc.R
# to avoid package problems 
#' Remove spikes from a time series
#'
#' The method identifies spikes with respect to a "reference" time-series, and
#' replaces these spikes with the reference value, or with `NA` according
#' to the value of `action`; see \dQuote{Details}.
#'
#' Three modes of operation are permitted, depending on the value of
#' `reference`.
#'
#' 1. For `reference="median"`, the first step is to linearly interpolate
#' across any gaps (spots where `x==NA`), using [approx()] with
#' `rule=2`. The second step is to pass this through
#' [runmed()] to get a running median spanning `k`
#' elements. The result of these two steps is the "reference" time-series.
#' Then, the standard deviation of the difference between `x`
#' and the reference is calculated.  Any `x` values that differ from
#' the reference by more than `n` times this standard deviation are considered
#' to be spikes.  If `replace="reference"`, the spike values are
#' replaced with the reference, and the resultant time series is
#' returned.  If `replace="NA"`, the spikes are replaced with `NA`,
#' and that result is returned.
#'
#' 2. For `reference="smooth"`, the processing is the same as for
#' `"median"`, except that [smooth()] is used to calculate the
#' reference time series.
#'
#' 3. For `reference="trim"`, the reference time series is constructed by
#' linear interpolation across any regions in which `x<min` or
#' `x>max`.  (Again, this is done with [approx()] with
#' `rule=2`.) In this case, the value of `n` is ignored, and the
#' return value is the same as `x`, except that spikes are replaced
#' with the reference series (if `replace="reference"` or with
#' `NA`, if `replace="NA"`.
#'
#' @param x a vector of (time-series) values, a list of vectors, a data frame,
#' or an [oce-class] object.
#'
#' @param reference indication of the type of reference time series to be used
#' in the detection of spikes; see \sQuote{Details}.
#'
#' @param n an indication of the limit to differences between `x` and the
#' reference time series, used for `reference="median"` or
#' `reference="smooth"`; see \sQuote{Details.}
#'
#' @param k length of running median used with `reference="median"`, and
#' ignored for other values of `reference`.
#'
#' @param min minimum non-spike value of `x`, used with
#' `reference="trim"`.
#'
#' @param max maximum non-spike value of `x`, used with
#' `reference="trim"`.
#'
#' @param replace an indication of what to do with spike values, with
#' `"reference"` indicating to replace them with the reference time
#' series, and `"NA"` indicating to replace them with `NA`.
#'
#' @param skip optional vector naming columns to be skipped. This is ignored if
#' `x` is a simple vector. Any items named in `skip` will be passed
#' through to the return value without modification.  In some cases,
#' `despike` will set up reasonable defaults for `skip`, e.g. for a
#' `ctd` object, `skip` will be set to \code{c("time", "scan",
#' "pressure")} if it is not supplied as an argument.
#'
#' @return A new vector in which spikes are replaced as described above.
#'
#' @author Dan Kelley
#'
#' @examples
#' n <- 50
#' x <- 1:n
#' y <- rnorm(n=n)
#' y[n/2] <- 10                    # 10 standard deviations
#' plot(x, y, type='l')
#' lines(x, despike(y), col='red')
#' lines(x, despike(y, reference="smooth"), col='darkgreen')
#' lines(x, despike(y, reference="trim", min=-3, max=3), col='blue')
#' legend("topright", lwd=1, col=c("black", "red", "darkgreen", "blue"),
#'        legend=c("raw", "median", "smooth", "trim"))
#'
#' # add a spike to a CTD object
#' data(ctd)
#' plot(ctd)
#' T <- ctd[["temperature"]]
#' T[10] <- T[10] + 10
#' ctd[["temperature"]] <- T
#' CTD <- despike(ctd)
#' plot(CTD)
despike <- function(x, reference=c("median", "smooth", "trim"), n=4, k=7, min=NA, max=NA,
                    replace=c("reference", "NA"), skip)
{
  if (is.vector(x)) {
    x <- despikeColumn(x, reference=reference, n=n, k=k, min=min, max=max, replace=replace)
  } else {
    if (missing(skip)) {
      if (inherits(x, "ctd"))
        skip <- c("time", "scan", "pressure")
      else
        skip <- NULL
    }
    if (inherits(x, "oce")) {
      columns <- names(x@data)
      for (column in columns) {
        if (!(column %in% skip)) {
          ## check for NA column
          if (all(is.na(x[[column]]))) {
            warning(paste("Column", column, "contains only NAs. Skipping"))
          } else {
            x[[column]] <- despikeColumn(x[[column]],
                                         reference=reference, n=n, k=k, min=min, max=max, replace=replace)
          }
        }
      }
      x@processingLog <- processingLogAppend(x@processingLog, paste(deparse(match.call()), sep="", collapse=""))
    } else {
      columns <- names(x)
      for (column in columns) {
        if (!(column %in% skip)) {
          if (all(is.na(x[[column]]))) {
            warning(paste("Column", column, "contains only NAs. Skipping"))
          } else {
            x[[column]] <- despikeColumn(x[[column]],
                                         reference=reference, n=n, k=k, min=min, max=max, replace=replace)
          }
        }
      }
    }
  }
  x
}

despikeColumn <- function(x, reference=c("median", "smooth", "trim"), n=4, k=7, min=NA, max=NA,
                          replace=c("reference", "NA"))
{
  reference <- match.arg(reference)
  replace <- match.arg(replace)
  gave.min <- !is.na(min)
  gave.max <- !is.na(max)
  nx <- length(x)
  ## degap
  na <- is.na(x)
  if (sum(na) > 0) {
    i <- 1:nx
    x.gapless <- approx(i[!na], x[!na], i, rule=2)$y
  } else {
    x.gapless <- x
  }
  if (reference == "median" || reference == "smooth") {
    if (reference == "median")
      x.reference <- runmed(x.gapless, k=k)
    else
      x.reference <- as.numeric(smooth(x.gapless))
    distance <- abs(x.reference - x.gapless)
    stddev <- sqrt(var(distance))
    bad <- distance > n * stddev
    nbad <- sum(bad)
    if (nbad > 0) {
      if (replace == "reference")
        x[bad] <- x.reference[bad]
      else
        x[bad] <- rep(NA, nbad)
    }
  } else if (reference == "trim") {
    if (!gave.min || !gave.max)
      stop("must give min and max")
    bad <- !(min <= x & x <= max)
    nbad <- length(bad)
    if (nbad > 0) {
      i <- 1:nx
      if (replace == "reference") {
        x[bad] <- approx(i[!bad], x.gapless[!bad], i[bad], rule=2)$y
      } else {
        x[bad] <- NA
      }
    }
  } else {
    stop("unknown reference ", reference)
  }
  x
}