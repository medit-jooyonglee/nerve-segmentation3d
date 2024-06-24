#include "SharedMemoryHeader.h"

#include <cstring>
#include <fstream>

/**
 * .
 * 
 * 
 * @param filename
 * @param mem
 */
bool SharedMemoryHeader::write(const std::string& filename)
{
	std::ofstream file(filename, std::ios::binary);

	if (file.is_open())
	{
		file.write((char*)this, sizeof(*this));
	}
	return true;
}



bool SharedMemoryHeader::read(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	bool breturn = true;
	if (file.is_open())
	{
		file.seekg(0, std::ios::end);
		size_t fsize = file.tellg();
		//std::cout << "mem size:" << sizeof(*this) << ","
		//	<< "read memory actucal size" << fsize << "\n";
		if (fsize == sizeof(*this))
		{
			file.seekg(0, std::ios::beg);
			file.read((char*)this, sizeof(*this));
		}
		else
		{
			breturn = false;
		}

	}
	else
	{
		breturn = false;
	}
	return breturn;
}

//bool read(const std::vector<uint8_t>& buffer, SharedMemoryHeader& mem)
//{
//	if (buffer.size() >= sizeof(mem))
//	{
//		return read(buffer.data(), mem);
//	}
//	else
//	{
//		return false;
//	}
//}


bool SharedMemoryHeader::write_buffer(const void* buffer)
{	
	if (buffer)
	{
		memcpy((void*)buffer, this, sizeof(*this));
	}
	
	return true;
}


bool SharedMemoryHeader::read_buffer(const void* buffer )
{
	memcpy(this, buffer, sizeof(*this));
	return true;
}
