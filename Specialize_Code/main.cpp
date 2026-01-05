#include<iostream>


template<typename T>
class Printer {
public:
	void Print() {
		std::cout << "Hello from template class!"<< std::endl;
	}
	friend std::ostream& operator<<(std::ostream& os, const Printer<T>& p) {
		os << "Generic Printer";
		return os;
	}
};

// 特化模板，泛型指定为int的时候会调用这个类进行初始化
template<>
class Printer<int> {
public:
	void Print() {
		std::cout << "Hello from template specialization!" << std::endl;
	}
	friend std::ostream& operator<<(std::ostream& os, const Printer<int>& p) {
		std::cout << "Int Printer";
		return os;
	}
};


int main() {
	Printer<double> p1;
	Printer<int> p2;

	p1.Print();
	p2.Print();

	std::cout << p1 << std::endl;
	std::cout << p2 << std::endl;
}