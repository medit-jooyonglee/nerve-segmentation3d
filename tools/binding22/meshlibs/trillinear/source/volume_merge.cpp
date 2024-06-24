#include "volume_merge.h"

#include <iostream>
#include <omp.h>

#ifdef _WIN32
#else
#ifndef __min
    #define __min(a, b) (((a) < (b)) ? (a) : (b))
#endif //

#ifndef __max
    #define __max(a, b) (((a) > (b)) ? (a) : (b))
#endif //
#endif //

bool VolumeMerge::merge( Volume& fullvolume, const std::vector<Volume>& split_volume, const Eigen::MatrixXi& bboxes)
{
	if (split_volume.size() == bboxes.size() && split_volume.size() > 0 )
	{
		std::runtime_error(std::string("zero split-volume or different split & box size.") + \
			std::string("\tsplit-vols:#") + std::to_string(split_volume.size()) + \
			std::string("\tboxes:#") + std::to_string(bboxes.size())
		);		
		return false;
	}

	


#ifdef _OPENMP
	int max_thread = omp_get_max_threads() / 2;
	omp_set_num_threads(max_thread);
	fullvolume.init_thread_pointer(max_thread);
#endif //
//#ifdef _OPENMP
//	this->_threadptr = new float[max_thread * ];
//#endf // 

	if (fullvolume.ndim == 3 && split_volume[0].ndim == 3 && 
		split_volume.size() == bboxes.rows() && bboxes.cols() == 6)
	{
		fullvolume.init_counters();
		size_t x_max_index = fullvolume.shape[0];
		size_t y_max_index = fullvolume.shape[1];
		size_t z_max_index = fullvolume.shape[2];
		//std::vector<int> k;
		//k[0]

		int i = 0;
#pragma omp parallel for
		for ( i = 0; i < split_volume.size(); i++)
		{
			size_t x1 = __min(__max(bboxes(i, 0), 0), x_max_index);
			size_t y1 = __min(__max(bboxes(i, 1), 0), y_max_index);
			size_t z1 = __min(__max(bboxes(i, 2), 0), z_max_index);
			size_t x2 = __min(__max(bboxes(i, 3), 0), x_max_index);
			size_t y2 = __min(__max(bboxes(i, 4), 0), y_max_index);
			size_t z2 = __min(__max(bboxes(i, 5), 0), z_max_index);

			for (size_t j = x1, j0 = 0; j < x2; j++, j0++)
			{
				for (size_t k = y1, k0 = 0; k < y2; k++, k0++)
				{
					for (size_t l = z1, l0 = 0; l < z2; l++, l0++)
					{	
						fullvolume.add(j, k, l,
							split_volume[i].data(j0, k0, l0)
						);
					}
				}
			}
		}

		fullvolume.divide_counter();
	}
	else
	{
		std::cout << "invalid dimension: full volume dim #" << fullvolume.ndim << "//" << "split volume dim : " << split_volume.size();
		return false;
	}
	return true;
}


void Volume::init_counters()
{
	long total_elem = 1;
	for (const auto v : this->shape)
		total_elem *= v;

	this->counters.resize(total_elem, 0);
}

void Volume::divide_counter()
{
	long total_elem = 1;
	for (const auto v : this->shape)
		total_elem *= v;

	int i = 0;
#pragma omp parallel for
	for (i = 0; i < total_elem; i++)
	{
		if (this->counters[i] > 0)
		{
			this->ptr[i] = this->ptr[i] / this->counters[i];
		}
	}
}

void Volume::debug_print()
{
	if (this->ndim == 3)
	{
		for (size_t i = 0; i < this->shape[0]; i++)
		{
			for (size_t j = 0; j < this->shape[1]; j++)
			{
				for (size_t k = 0; k < this->shape[2]; k++)
				{
					std::cout << i << "," << j << "," << k << ":" << this->data(i, j, k) << "\n";
				}
			}
		}

			
	}
	else if (this->ndim == 4)
	{

		for (size_t i = 0; i < this->shape[0]; i++)
		{
			for (size_t j = 0; j < this->shape[1]; j++)
			{
				for (size_t k = 0; k < this->shape[2]; k++)
				{
					for (size_t l = 0; l < this->shape[3]; l++)
						std::cout << i << "," << j << "," << k << "," << l << ":" << this->data(i, j, k, l) << "\n";
				}
			}
		}
	}

}
