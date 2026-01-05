#include<iostream>
#include<string>

//函数模板只在调用的时候被创建，当调用时才会实例化
template<typename T>
void Print(T value) {
	std::cout << value << std::endl;
}

//类模板
//考虑现在这个需求，我们希望创建一个Array，这个Array需要在编译期确定大小并且分配在栈空间中的对象
//此时我们可以这样，这里其实说明了typename 也是cpp内置的一个关键字
//那么，当我们需要这个数组存储不同的类型时，可以再添加一个typename字段
template<typename T,int N>
class Array {
private:
	T m_array[N];
public:
	int GetSize()const{
		return N;
	}
};
int main() {
	Print(42); // Prints an integer
	Print<float>(3.2f);
	std::cout << sizeof(char) << std::endl;
	Array<int, 1024 * 1024 * 4>* b;
	Array<int, 1024 * 1024 * 4>* c;
	{
		Array<int, 1024 * 1024 * 4>* a = new Array<int, (int)(sizeof(char) * 1024 * 1024 * 4)>();
		Array<int, 1024 * 1024 * 4> d;
		c = &d;
		b = a;
	}
	std::cout << b->GetSize() << std::endl;
	std::cout << c->GetSize() << std::endl;
	//std::cout << "Size of array: " << array->GetSize() << std::endl;
}