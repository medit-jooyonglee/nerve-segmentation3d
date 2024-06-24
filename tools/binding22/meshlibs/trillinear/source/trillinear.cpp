#include "trillinear.h"
#include <algorithm>
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

template<typename T>
T interpolate_trillinear(T p[], T x, T y, T z)
{
	T v = 0;
	return v;
}


template<typename T>
CTrillinear<T>::CTrillinear(const MatrixXr& img,
        Eigen::Vector3i & shape,
        Method method,
        bool bound_error, float bound_value)
	: step0(0)
	, step1(0)
	, _method(method)
	, _bound_value(bound_value)
	, _bound_error(bound_error)
{	
	if (img.size() == shape.prod())
	{
		this->_img_array_ptr = (float*)img.data();
		img_shape = shape;
		this->step0 = shape(2);
		this->step1 = shape(1) * shape(2);
	}
	else
	{
		throw "Invalid Image-Data and Shape";
	}
}

template<typename T>
int CTrillinear<T>::_toIndex(int x, int y, int z)
{
	return x * this->step1 + y * this->step0 + z;	
}

template<typename T>
T CTrillinear<T>::getValue(int x, int y, int z)
{
	int index = this->_toIndex(x, y, z);
	return this->_img_array_ptr[index];
}

template<typename T>
void CTrillinear<T>::debug_print()
{
    for (int ix = 0; ix < this->img_shape(0); ix++)
    {
        for (int iy = 0; iy < this->img_shape(1); iy++)
        {

            for (int iz = 0; iz < this->img_shape(2); iz++)
            {
                std::cout << this->getValue(ix, iy, iz) << ", ";
            }
            std::cout << "\n";

        }
    }
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> \
CTrillinear<T>::interpolate(const MatrixXr& warps)
{
	assert(warps.cols() == 3);
    MatrixXr out;
	if (_img_array_ptr == nullptr)
	{
		return out;
	}
	int max_index_x = img_shape(0) - 1;
	int max_index_y = img_shape(1) - 1;
	int max_index_z = img_shape(2) - 1;

	out.resize(warps.rows(), 1);
	if ( this->_method == NEAREST )
	{
        #pragma omp parallel for
        for (int r = 0; r < warps.rows(); r++)
        {
            float x = warps(r, 0);
            float y = warps(r, 1);
            float z = warps(r, 2);

            float cx = __min(__max(x, 0.f), max_index_x);
            float cy = __min(__max(y, 0.f), max_index_y);
            float cz = __min(__max(z, 0.f), max_index_z);
            bool inner_bound = \
                    0.f <= x && x <= max_index_x && \
                    0.f <= y && y <= max_index_y && \
                    0.f <= z && z <= max_index_z;

            if( _bound_error && !inner_bound)
            {
                out(r) = this->_bound_value;

            }
            else
            {
                // to access floor index
                int x0 = (int)(floor(cx));
                int y0 = (int)(floor(cy));
                int z0 = (int)(floor(cz));

                // to access ceil index
                int x1 = std::min(x0 + 1, max_index_x);
                int y1 = std::min(y0 + 1, max_index_y);
                int z1 = std::min(z0 + 1, max_index_z);

                // ratio positive (0~1)
                float px = x - x0;
                float py = y - y0;
                float pz = z - z0;

                // ratio negative (0~1)
                float nx = 1.f - px;
                float ny = 1.f - py;
                float nz = 1.f - pz;

                Eigen::VectorXf weights(8);
                weights <<  nx * ny * nz,\
                            nx * ny * pz,\
                            nx * py * nz,\
                            nx * py * pz,\
                            px * ny * nz,\
                            px * ny * pz,\
                            px * py * nz,\
                            px * py * pz;

                Eigen::VectorXf values(8);
                values <<  getValue(x0, y0, z0),
                            getValue(x0, y0, z1),
                            getValue(x0, y1, z0),
                            getValue(x0, y1, z1),
                            getValue(x1, y0, z0),
                            getValue(x1, y0, z1),
                            getValue(x1, y1, z0),
                            getValue(x1, y1, z1);
                int ix = 0;
                // get index of largest weights
                weights.maxCoeff(&ix);
                out(r) = values(ix);


            }



        }
	}
    else if ( this->_method == LINEAR )
    {
        #pragma omp parallel for
        for (int r = 0; r < warps.rows(); r++)
        {
            float x = warps(r, 0);
            float y = warps(r, 1);
            float z = warps(r, 2);

            // protect over-index
//            float cx = std::min(std::max(x, 0.f), (float)max_index_x);
//            float cy = std::min(std::max(y, 0.f), (float)max_index_y);
//            float cz = std::min(std::max(z, 0.f), (float)max_index_z);

            float cx = __min(__max(x, 0.f), max_index_x);
            float cy = __min(__max(y, 0.f), max_index_y);
            float cz = __min(__max(z, 0.f), max_index_z);

            bool inner_bound = \
                    0.f <= x && x <= max_index_x && \
                    0.f <= y && y <= max_index_y && \
                    0.f <= z && z <= max_index_z;

            if( _bound_error && !inner_bound)
            {
                out(r) = this->_bound_value;

            }
            else
            {
                // to access floor index
                int x0 = (int)(floor(cx));
                int y0 = (int)(floor(cy));
                int z0 = (int)(floor(cz));

                // to access ceil index
                int x1 = std::min(x0 + 1, max_index_x);
                int y1 = std::min(y0 + 1, max_index_y);
                int z1 = std::min(z0 + 1, max_index_z);

                // ratio positive (0~1)
                float px = x - x0;
                float py = y - y0;
                float pz = z - z0;

                // ratio negative (0~1)
                float nx = 1.f - px;
                float ny = 1.f - py;
                float nz = 1.f - pz;

                float value = getValue(x0, y0, z0) * nx * ny * nz + \
                    getValue(x0, y0, z1) * nx * ny * pz + \
                    getValue(x0, y1, z0) * nx * py * nz + \
                    getValue(x0, y1, z1) * nx * py * pz + \
                    getValue(x1, y0, z0) * px * ny * nz + \
                    getValue(x1, y0, z1) * px * ny * pz + \
                    getValue(x1, y1, z0) * px * py * nz + \
                    getValue(x1, y1, z1) * px * py * pz;
                out(r) = value;

            }

        }
    }
    else
    {
    // error
    }

	return out;
}


void implicit()
{
    CTrillinear<float> interpf;
    CTrillinear<double> interpd;
    interpf.debug_print();
    interpd.debug_print();

    //typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  MatrixXr;
    //typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  Matrixdr;
    MatrixXfr xf;
    MatrixXdr xd;
    interpf.interpolate(xf);
    interpd.interpolate(xd);
}
