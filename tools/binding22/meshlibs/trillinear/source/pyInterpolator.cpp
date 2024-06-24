#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>
#include <omp.h>

#include "trillinear.h"

namespace py = pybind11;

template<typename Real>
void pybind_interpolatorf(py::module_ &m){
	py::class_<CTrillinear<Real>> trilf(m, "CTrillinearf");

	trilf.def(py::init<>())
	    .def("interpolate", &CTrillinear<Real>::interpolate, "some method interpolate")
	    .def_readwrite("_method", &CTrillinear<Real>::_method)
        .def("debug_print", &CTrillinear<Real>::debug_print)
        .def(py::init([](const py::buffer &b, bool bound_error, Real fill_value) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<Real>::format() || info.ndim != 3) {
                throw std::runtime_error("Incompatible buffer format!");
            }
            auto *v = new CTrillinear<Real>();//(info.shape[0], info.shape[1], (Real*)info.ptr);
            v->_img_array_ptr = (Real*)(info.ptr);


            for(int i = 0; i < info.ndim; i++)
                v->img_shape(i) = info.shape[i];
            v->_bound_error = bound_error;
            v->_bound_value = fill_value;
		    v->step0 = v->img_shape(2);
		    v->step1 = v->img_shape(1) * v->img_shape(2);
            //memcpy(v->data(), info.ptr, sizeof(Real) * (size_t) (v->rows() * v->cols()));

            return v;
        }));

    py::enum_<Method>(trilf, "Method", py::module_local())
        .value("NEAREST", Method::NEAREST)
        .value("LINEAR", Method::LINEAR)
        .export_values();
}



template<typename Real>
void pybind_interpolatord(py::module_& m) {
    py::class_<CTrillinear<Real>> trild(m, "CTrillineard");

    trild.def(py::init<>())
        .def("interpolate", &CTrillinear<Real>::interpolate, "some method interpolate")
        .def_readwrite("_method", &CTrillinear<Real>::_method)
        .def("debug_print", &CTrillinear<Real>::debug_print)
        .def(py::init([](const py::buffer& b, bool bound_error, Real fill_value) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<Real>::format() || info.ndim != 3) {
            throw std::runtime_error("Incompatible buffer format!");
        }
        auto* v = new CTrillinear<Real>();//(info.shape[0], info.shape[1], (Real*)info.ptr);
        v->_img_array_ptr = (Real*)(info.ptr);


        for (int i = 0; i < info.ndim; i++)
            v->img_shape(i) = info.shape[i];
        v->_bound_error = bound_error;
        v->_bound_value = fill_value;
        v->step0 = v->img_shape(2);
        v->step1 = v->img_shape(1) * v->img_shape(2);
        //memcpy(v->data(), info.ptr, sizeof(Real) * (size_t) (v->rows() * v->cols()));

        return v;
            }));

    // https://stackoverflow.com/questions/76072215/pybind11-generic-type-type-is-already-registered-when-importing-two-modul
//    py::enum_<Method>(trild, "Method", py::module_local())
//         .value("NEAREST", Method::NEAREST)
//         .value("LINEAR", Method::LINEAR)
//        .export_values();
}


void implicit_pybind_interpolator()
{
    py::module_ m;    
    pybind_interpolatorf<float>(m);
    pybind_interpolatord<double>(m);
}
//template<typename Real>
//void pybind_interpolator(py::module_& m, const std::string& name) {
//    py::class_<CTrillinear<Real>> trilf(m, name);