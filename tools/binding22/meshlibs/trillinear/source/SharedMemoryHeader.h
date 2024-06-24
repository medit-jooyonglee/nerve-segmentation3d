#pragma once

#include <vector>
#include <iostream>

/**
 * @brief python �� �����Ǵ� ��� �����̴�. ����Ǹ� �ݵ�� python process �� ������(binding) ������Ѵ�.
 * POD type? strict?
 * 
 * POD �ڼ��ϰԴ� ������ Ư���� ������.
 * ���� �Լ��� ���� ���� Ŭ������ ���� �ʴ´�.
 * ������ ������ ����� ������ ���� ��� ������.
 * ��� ������ ������ ����� ǥ�� ���̾ƿ��̴�.
 * ��� ���� Ŭ������ ǥ�� ���̾ƿ��̴�.
 * ù��° ������ ����� ���� Ŭ������ ������ Ÿ���� �����͸� ���� �ʴ´�.
 * ���۷��� ����� ���� �ʴ´�.
 * �� �� �̻��� ���� Ŭ���� Ȥ�� �Ļ� �� ���� Ŭ���� �� �ٿ��� ������ �����͸� ���� �ʴ´�.
 *  pybind11::buffer_info ����ü�� ���� �����ϰ� ó���Ѵ�.
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
	
	
	// object type. ex) str, np.array... ����� ��� �׳� numpy array�� �ϰ� ó���Ѵ�.
	char type[100] = "";	
	// dtype. uint8, int8...uint16... ���
	char dtype[10] = "";
	// Shape of the tensor (1 entry per dimension)/ shape ���� 5 ndim������ �����𵵷� 5�� ����. �� �̻��� dim�� ����μ��� ��������� ����.
	size_t shape[5] = { 0, };
	// Number of bytes between adjacent entries / ���������� 5�� ����.
	size_t strides[5] = { 0, };
	// Number of dimensions
	size_t ndim = 0;
	
	/**
	 * �޸� � �����ߴ��� Ȯ�ο�
	 * ����Ʈ�� 1�� ����.
	 * 2�� �̻��� ��� ������ ���� ����ȭ �ȴ�.
	 * (SharedMemoryHeader#1) - (Btyes #1) - (SharedMemoryHeader#2) - (Bytes #2)
	 */
	size_t number_of_memory = 1;

	/**
	 * memory �̸�
	 */
	char name[50] = "";

	/**
	 * �޸� �� ����� ����
	 */
	size_t total_memory_size = 0;

	/**
	 * 
	 * python shape.setter(...)
	 * @param[in] val (5,) vector shape size ����
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
	 * @brief ������ ����Ʈ 
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
	 * @brief ���Ϸ� ����
	 * 
	 * @param filename
	 * @return 
	 */
	bool write(const std::string& filename);

	/**
	 * @brief ���Ͽ��� �о�´�.
	 *
	 * @param filename
	 * @param mem
	 */
	bool read(const std::string& filename);


	/**
	 * @brief buffer���� �о�´�. ����ü ������� �о�´�. ���̽㿡���� ���
	 * 
	 * @param buffer
	 * @return 
	 */
	bool read_buffer(const void* buffer);

	/**
	 * @brief buffer�� ����ü ������ �����Ѵ�. ����ü ������ �״�� �����Ѵ�.
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



