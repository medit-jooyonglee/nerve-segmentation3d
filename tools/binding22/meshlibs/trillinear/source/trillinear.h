#pragma once

#include <math.h>
#include <Eigen/Core>
#include <assert.h>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  MatrixXdr;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  MatrixXfr;
typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  MatrixXur;

template<typename T>
inline T interpolate_trillinear(T p[2][2][2], T x, T y, T z)
{
	x = min(max(x, 0), 1);
	y = min(max(y, 0), 1);
	z = min(max(z, 0), 1);

	T px = x;
	T py = y;
	T pz = z;
	T nx = 1. - x;
	T ny = 1. - y;
	T nz = 1. - z;

	return	p[0][0][0] * nx * nx * nz + \
		p[0][0][1] * nx * nx * pz + \
		p[0][1][0] * nx * px * nz + \
		p[0][1][1] * nx * px * pz + \
		p[1][0][0] * px * nx * nz + \
		p[1][0][1] * nx * nx * nz + \
		p[1][1][0] * nx * nx * nz + \
		p[1][1][1] * nx * nx * nz;
}

enum Method {
    NEAREST = 0,
    LINEAR
};

template<typename T>
class CTrillinear
{
public:
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  MatrixXr;

	CTrillinear() = default;	
	CTrillinear(const MatrixXr&img, Eigen::Vector3i &shape, Method method=LINEAR, bool bound_error=true, float bound_value = 0.0f);

	MatrixXr interpolate(const MatrixXr&warps);
	
	T getValue(int x, int y, int z);
	Eigen::Vector3i img_shape;
	MatrixXr img_array;
	T *_img_array_ptr = nullptr;
	int step0;
	int step1;
	Method _method = Method::LINEAR;
	bool _bound_error = true;
	T _bound_value = 0.f;

	void debug_print();
private:
	int _toIndex(int x, int y, int z);

};

