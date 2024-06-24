#pragma once

#include <vector>
#include <iostream>

/**
 * @brief python 과 연동되는 헤더 정보이다. 변경되면 반드시 python process 도 리빌드(binding) 해줘야한다.
 * POD type? strict?
 * 
 * POD 자세하게는 다음의 특성을 가진다.
 * 가상 함수나 가상 기저 클래스를 갖지 않는다.
 * 비정적 데이터 멤버가 동일한 접근 제어를 가진다.
 * 모든 비정적 데이터 멤버가 표준 레이아웃이다.
 * 모든 기저 클래스가 표준 레이아웃이다.
 * 첫번째 비정적 멤버로 기저 클래스와 동일한 타입의 데이터를 갖지 않는다.
 * 레퍼런스 멤버를 갖지 않는다.
 * 한 개 이상의 기저 클래스 혹은 파생 및 기저 클래스 둘 다에서 비정적 데이터를 갖지 않는다.
 *  pybind11::buffer_info 구조체와 거의 동일하게 처리한다.
 *   ssize_t itemsize = 0;         // Size of individual items in bytes
 *   ssize_t size = 0;             // Total number of entries
 *   std::string format;           // For homogeneous buffers, this should be set to
 *                                 // format_descriptor<T>::format()
 *   ssize_t ndim = 0;             // Number of dimensions
 *   std::vector<ssize_t> shape;   // Shape of the tensor (1 entry per dimension)
 *   std::vector<ssize_t> strides; // Number of bytes between adjacent entries
 */
class SharedMemoryHeader
{
public:
	SharedMemoryHeader() = default;
	~SharedMemoryHeader() = default;
	
	
	// object type. ex) str, np.array... 현재는 모두 그냥 numpy array로 일괄 처리한다.
	char type[100] = "";	
	// dtype. uint8, int8...uint16... 등등
	char dtype[10] = "";
	// Shape of the tensor (1 entry per dimension)/ shape 정보 5 ndim까지만 지원디도록 5로 고정. 그 이상의 dim은 현재로서는 사용할일이 없다.
	size_t shape[5] = { 0, };
	// Number of bytes between adjacent entries / 마찬가지로 5로 고정.
	size_t strides[5] = { 0, };
	// Number of dimensions
	size_t ndim = 0;
	
	/**
	 * 메모리 몇개 저장했는지 확인용
	 * 디폴트로 1로 지정.
	 * 2개 이상일 경우 다음과 같이 직렬화 된다.
	 * (SharedMemoryHeader#1) - (Btyes #1) - (SharedMemoryHeader#2) - (Bytes #2)
	 */
	size_t number_of_memory = 1;

	/**
	 * memory 이름
	 */
	char name[50] = "";

	/**
	 * 메모리 총 사이즈를 저장
	 */
	size_t total_memory_size = 0;

	/**
	 * 
	 * python shape.setter(...)
	 * @param[in] val (5,) vector shape size 설정
	 */
	void set_shape(const std::vector<size_t> &val)
	{
		if (val.size() < 5)
		{
			this->ndim = val.size();
			for (size_t i = 0; i < val.size(); i++)
			{
				this->shape[i] = val[i];
			}
		}
		else
		{
			throw std::runtime_error("cannot 5 dimension");
			// invalid size
		}
	}

	/**
	 * @brief 디버깅용 프린트 
	 * 
	 */
	void print()
	{
		std::cout << *this;
	}


	/**
	 * python shape.getter(...)
	 * 
	 * @return 
	 */
	std::vector<size_t> get_shape()
	{
		std::vector<size_t> out;
		out.resize(this->ndim);
		for (size_t i = 0; i < this->ndim; i++)
		for (size_t i = 0; i < this->ndim; i++)
		{
			out[i] = this->shape[i];
		}
		return out;
	}

	
	
	friend std::ostream& operator<<(std::ostream& os, const SharedMemoryHeader& rhs)
	{
		os << "type:" << rhs.type << ", "
			<< "dtype:" << rhs.dtype <<", ";
		os << "ndim:" << rhs.ndim;
		os << "shape: [";
		for (size_t i = 0; i < rhs.ndim; i++)
		{
			os << rhs.shape[i] << ",";
		}
		os << "]";
		return os;
	}

	/**
	 * @brief 파일로 저장
	 * 
	 * @param filename
	 * @return 
	 */
	bool write(const std::string& filename);

	/**
	 * @brief 파일에서 읽어온다.
	 *
	 * @param filename
	 * @param mem
	 */
	bool read(const std::string& filename);


	/**
	 * @brief buffer에서 읽어온다. 구조체 사이즈로 읽어온다. 파이썬에서도 사용
	 * 
	 * @param buffer
	 * @return 
	 */
	bool read_buffer(const void* buffer);

	/**
	 * @brief buffer에 구조체 정보를 저장한다. 구조체 사이즈 그대로 저장한다.
	 * 
	 * @param buffer
	 * @return 
	 */
	bool write_buffer(const void* buffer);

	size_t bytesize()
	{
		return sizeof(*this);
	}

};



