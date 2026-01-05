#include<iostream>


class Entity {
private:
	int m_x, m_y;
	mutable int var;
public:
	//不能修改成员变量的值
	int GetX() const {
		//m_x = 2;
		//mutable关键字的作用是允许在const成员函数中修改这个变量的值
		var = 1;
		return m_x;
	}

	int GetX() {
		return m_x;
	}
	//这个函数说的是，返回一个不能修改指向的指针，这个指针指向的数据也不能被修改，同时这个函数不能修改成员变量
	const int* const GetConstX() const {
		return &m_x;
	}
};

void PrintEntity(const Entity* e) {
	//这个函数是可以改变指针指向的
	//e = nullptr;
	std::cout << e->GetX() << std::endl;
}

void PrintEntity(const Entity& e) {
	//这个函数不可以改变指针指向，因为引用本质是一个不可改变指向的指针
	//e = nullptr;
	//如果把GetX的const去掉(只有不含const的GetX)，那么这里会报错，因为这个函数需要不能修改成员变量的值，
	// 因而我们需要在方法签名中声明这个方法没有修改成员变量的值
	std::cout << e.GetX() << std::endl;
}

int main() {
	const int MAX_AGE = 90;

	//不能修改指针的指向
	int* const ptr1 = new int;
	*ptr1 = 25;
	//ptr = new int;

	//不能修改指向的内容
	//这两个写法是一样的，关键在于const关键字在*的前面还是后面
	const int* ptr2 = new int;
	//int const* ptr2 = new int;
	//*ptr2 = 10;
}