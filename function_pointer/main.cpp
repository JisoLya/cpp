#include<iostream>
#include<vector>

void HelloWorld() {
	std::cout << "Hello world" << std::endl;

}

//函数指针的作用
template<typename T>
void forEach(std::vector<int>& values, T f) {
	for (int& val : values) {
		f(val);
	}
}

void Print(int& num) {
	std::cout << "value: " << num << std::endl;
}

int main() {
	//这个函数指针的类型实际上是void(*)()
	//
	void(*soya)();
	soya = HelloWorld;
	auto f = HelloWorld;

	soya();
	f();
		
	auto func = [](int& c) {
		c += 2;
		};
	std::vector<int> values = { 1,2,3,4,5 };
	forEach(values, func);
	for (int i : values) {
		std::cout << i << std::endl;
	}
	forEach(values, Print);
}