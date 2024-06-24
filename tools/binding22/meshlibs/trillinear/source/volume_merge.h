#pragma once

#include <vector>
#include <Eigen/Core>

#include <omp.h>


/*
* @brief pybind11 buffer_info와 동일하게 구성
* /// Information record describing a Python buffer object
struct buffer_info {
    void *ptr = nullptr;          // Pointer to the underlying storage
    ssize_t itemsize = 0;         // Size of individual items in bytes
    ssize_t size = 0;             // Total number of entries
    std::string format;           // For homogeneous buffers, this should be set to
                                  // format_descriptor<T>::format()
    ssize_t ndim = 0;             // Number of dimensions
    std::vector<ssize_t> shape;   // Shape of the tensor (1 entry per dimension)
    std::vector<ssize_t> strides; // Number of bytes between adjacent entries
                                  // (for each per dimension)
    bool readonly = false;        // flag to indicate if the underlying storage may be written to

*/
class Volume
{
public:
	Volume() = default;

	// likewise 
	float* ptr = nullptr;
	std::vector<size_t> shape;
	
	/**
	 * .
	 */
	/*
	* 
	* py::buffer_info의 stirdes와 다르게 itemsSize를 나눈값
	* py::buffer_info::strides(-1) = itemsize 이지만
	* py::buffer_info::strides(-2) = py::buffer_info::strides(-1) *  itemsize 이지만
	* pointer strides로써 관리한다. 즉 다음과 같다.
	* Volume::stirdes(-1) =  itemSize 
	* Volume::strides(-2) = shape(-1) * itemSize
	* .
	* Volume::strides(0) = shape(1) * shape(2) * ... * shape(-1)
	* 현재로썬 data-type float으로 고정되어 있기때문!
	*/
	std::vector<size_t> strides;
	size_t itemsize;
	size_t ndim;
	std::string format;           // For homogeneous buffers, this should be set to

	std::vector<size_t> counters; // divide counters for final-value

	float* _threadptr = nullptr;
	void init_counters();

#ifdef _OPENMP
	void init_thread_pointer(int max_thread)
	{
		this->_threadptr = new float[this->get_total_size() * max_thread];
	}
#endif // 

	size_t get_total_size()
	{
		size_t total_elem = 1;
		for (const auto v : this->shape)
			total_elem *= v;
		return total_elem;
	}
	//float data(size_t i, size_t j, size_t k) const;
	//float data(size_t i, size_t j, size_t k, size_t l) const;
	

	inline float data(size_t i, size_t j, size_t k) const
	{

#ifdef DEBUG
		float val = 0.0f;
		if (this->ndim == 3)
		{
			return this->ptr[i * this->strides[0] + \
				j * this->strides[1] + \
				k];
}
		else
		{

			// error
		}
		return val;
#else
		return this->ptr[i * this->strides[0] + \
			j * this->strides[1] + \
			k];
#endif // 
}

	inline float data(size_t i, size_t j, size_t k, size_t l) const
	{

#ifdef DEBUG
		float val = 0.0f;
		if (this->ndim == 4)
		{
			return this->ptr[i * this->strides[0] + \
				j * this->strides[1] + \
				k * this->strides[2] + \
				l];
			//this->shape[0] * i + l
		}
		else
		{
		}
		return 0.0f;
#else
		return this->ptr[i * this->strides[0] + \
			j * this->strides[1] + \
			k * this->strides[2] + \
			l];
#endif // 
	}

	inline void add(const size_t i, const size_t j, const size_t k, const float val)
	{
#ifdef DEBUG
		if (this->ndim != 3)
		{
			std::cout << "supported funtion for 3dim";
			return;
		}
#endif // DEBUG

#ifdef _OPENMP

		size_t arr = i * this->strides[0] + \
			j * this->strides[1] + \
			k;
		//int tid = omp_get_thread_num();
		//this->ptr[arr] = omp_get_thread_num()* this->ptr[arr] + val;
		//this->counters[arr] = this->counters[arr] + 1;

		this->ptr[arr] += val; //this->ptr[arr] + val;
		this->counters[arr] ++;// this->counters[arr] + 1;
#else
		size_t arr = i * this->strides[0] + \
			j * this->strides[1] + \
			k;
		this->ptr[arr] = this->ptr[arr] + val;
		this->counters[arr] = this->counters[arr] + 1;
#endif //

	}

	void divide_counter();

	bool is_empty()
	{
		return ptr == nullptr;
	}

	void debug_print();
};


class VolumeMerge
{
public:
	VolumeMerge() = default;
	~VolumeMerge() = default;

	bool merge(Volume& fullvolume, const std::vector<Volume>& split_volume, const Eigen::MatrixXi& bboxes);
};
