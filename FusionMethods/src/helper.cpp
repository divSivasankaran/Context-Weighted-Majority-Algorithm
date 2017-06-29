#include<../include/helper.h>



template<typename T>
void PrintVector(std::vector<T> p)
{
	std::cout << "[";
	for(auto i = 0; i<p.size(); i++)
		std::cout << p[i]<<" , ";
	std::cout <<"]" << std::endl; 
}
