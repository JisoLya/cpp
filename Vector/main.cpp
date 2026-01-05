#include<iostream>
#include<vector>
struct vertex {
	int x, y, z;
	vertex(int x, int y, int z): x(x), y(y), z(z){
		std::cout << "construct!" << std::endl;
	}
	vertex(const vertex& v) : x(v.x), y(v.y), z(v.z) {
		std::cout << "copy vertex: " << v.x << ", " << v.y << ", " << v.z << std::endl;
	}
};


int main(){
	//这里会调用默认的构造函数，但是我们其实没有定义，我们只希望他留有三个对象的内存空间而不是构造对象
	//std::vector<vertex> v(3);
	std::vector<vertex> v;
	//利用reserve函数来预留空间
	v.reserve(3);
	//这样会调用拷贝构造函数
	//由于一开始的vector是空的，所以会调用拷贝构造函数，进行扩容
	/*
	v.push_back(vertex(1, 2, 3));
	v.push_back(vertex(4, 5, 6));
	v.push_back(vertex(7, 8, 9));
	*/

	//上面的写法过程是这样的，vertex对象首先被构造，然后复制到vector中
	//我们希望的是直接在现有的内存空间中构造对象，而不是先构造后复制
	//看1， 2， 3 的这个构造函数就可以知道调用构造函数实际上是先创建再拷贝，而直接传参数列表则是在现有的内存空间中构造对象
	v.emplace_back(vertex(1, 2, 3));
	v.emplace_back(4, 5, 6);
	v.emplace_back(7, 8, 9);
}