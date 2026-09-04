#' @useDynLib eigenPyR
#' @importFrom Rcpp evalCpp
"_PACKAGE"

.onLoad = function(libname, pkgname) {
  Rcpp::loadModule("eigenPyR_MODULE", TRUE)
}
