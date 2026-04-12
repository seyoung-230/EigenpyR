#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "lm_ols.h"

namespace py = pybind11;

void bind_lm_ols(py::module_& m) {
    py::class_<LmOLS>(m, "LmOLS")
        .def(py::init<>())
        .def("fit", &LmOLS::fit, py::arg("X"), py::arg("y"))
        .def("predict", &LmOLS::predict, py::arg("X_new"))
        .def_property_readonly("coef", &LmOLS::coef)
        .def_property_readonly("intercept", &LmOLS::intercept);
}