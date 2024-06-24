#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>

#include "SharedMemoryHeader.h"

namespace py = pybind11;


void write_buffer(SharedMemoryHeader& self, py::bytes buffer)
{
	py::buffer_info info(py::buffer(buffer).request());
	
	self.write_buffer(info.ptr);
}


void read_buffer(SharedMemoryHeader& self, py::bytes buffer)
{
	py::buffer_info info = py::buffer(buffer).request();
	//std::cout << "loaded:\n" << info.ptr << ":" << info.size << "\n";
	self.read_buffer(info.ptr);
}


void pybind_sharedmemoryheader(py::module_& m) {
	py::class_<SharedMemoryHeader> mem(m, "SharedMemoryHeader");
	
	mem.def(py::init<>())
		.def_property("dtype", [](SharedMemoryHeader &self) {
		return std::string(self.dtype);
			},
			[](SharedMemoryHeader& self, const std::string &val) {
				
				strcpy(self.dtype, val.c_str());
				//std::cout << "set dtype:" << val << "to" << self.dtype << "\n";
			}, py::return_value_policy::copy)
		.def_property("shape", &SharedMemoryHeader::get_shape, &SharedMemoryHeader::set_shape,
			py::return_value_policy::copy)
		.def_readwrite("ndim", &SharedMemoryHeader::ndim)
		.def_readwrite("number_of_memory", &SharedMemoryHeader::number_of_memory)
		.def_property("name", [](SharedMemoryHeader& self) {
		return std::string(self.name);
			},
			[](SharedMemoryHeader& self, const std::string& val) {
				strcpy(self.name, val.c_str());
			}, py::return_value_policy::copy)
		.def_readwrite("total_memory_size", &SharedMemoryHeader::total_memory_size)
		.def("print", &SharedMemoryHeader::print)
		.def("write", &SharedMemoryHeader::write)
		.def("read", &SharedMemoryHeader::read)
		.def("write_buffer", &write_buffer)
		.def("read_buffer", &read_buffer)
		.def("tobytes", [](SharedMemoryHeader &self) {
			return py::bytes((char*)&self, sizeof(self));
			})
		.def("bytesize", &SharedMemoryHeader::bytesize);

		
	
}

