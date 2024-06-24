#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>
#include <omp.h>

#include <iostream>

#include <Eigen/Core>

#include "volume_merge.h"
namespace py = pybind11;

/**
 * 
 * 
 * @param[in] info
 * @param[out] lhs
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
	os << "[";
	int i = 1;
	for (const T& v : vec)
	{
		os << v;
		if (i++ < vec.size())
			os << ",";
	}
	os << "]";
	return os;
}


void assign(const py::buffer_info& info, Volume& rhs, size_t check_dim = -1)
{
	if (info.format != py::format_descriptor<float>::format())
	{
		std::cout << "format:" << info.format << "\n";
		throw std::runtime_error("Incompatible buffer format:" + info.format);
	}
	if (check_dim > 0)
	{
		if (info.ndim != check_dim)
		{
			throw std::runtime_error("Incompatible dimension:" + info.ndim);
		}
	}

	rhs.ptr = (float*)info.ptr;
	rhs.ndim = info.ndim;
	rhs.itemsize = info.itemsize;
	rhs.shape.reserve(info.ndim);
	rhs.strides.reserve(info.ndim);

	for (const auto& v : info.shape)
		rhs.shape.push_back(v);

	// itemsize나눈 값으로 포인터 설정
	for (const auto& v : info.strides)
		rhs.strides.push_back(v/rhs.itemsize);

	rhs.format = info.format;
}



void assign(const std::vector<py::buffer> & buffers, std::vector<Volume> & rhs, size_t check_dim = -1)
{
	rhs.reserve(buffers.size());
	for (const py::buffer& buf : buffers)
	{
		const py::buffer_info info =  buf.request();
		Volume vol;
		assign(info, vol, check_dim);
		rhs.push_back(vol);
		//vol.debug_print();
	}
}


void volumemerge(VolumeMerge& self, 
	const py::buffer &full_vol,
	const std::vector<py::buffer> &split_vol,
	Eigen::MatrixXi &bboxes)
{
	py::buffer_info full_buffer = full_vol.request();
	
	Volume full;
	std::vector<Volume> splits;
	assign(full_buffer, full, 3);
	assign(split_vol, splits, 3);

	if (splits.size() == bboxes.rows())
	{
		std::runtime_error("not same split volume size & bboxes size");
	}

	self.merge(
		full, splits, bboxes
	);
}


void pybind_volume_merge(py::module_& m)
{
	py::class_<VolumeMerge>(m, "VolumeMerge")
		.def(py::init<>())
		.def("merge", &volumemerge,			
			py::arg_v("full_vol", "full volume array-3dim"),
			py::arg_v("split_vol", "split volume array-4dim"),
			py::arg_v("bboxes", "bboxes info (N, 6)")
			);



}
