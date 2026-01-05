#include<iostream>
/*
类中的静态变量可以通过类名::变量名来访问
所有的实例对象共享同一份静态变量
*/


struct Person
{
	//不提供构造函数
	//Person() = delete;
	static int x, y;
};

int Person::x;
int Person::y;


//static in function
void Function() {
	static int i = 0;
	i++;
	std::cout << "i = " << i << std::endl;
}

//Enum
enum Type: unsigned char
{
	A,
	B,
	C
};

int main() {
	Person p1, p2;
	p1.x = 10;
	p1.y = 20;
	std::cout << "Address of p1.x: " << &p1.x << std::endl;
	std::cout << "Address of p1.y " << &p1.y << std::endl;
	//same
	std::cout << "Address of p2.x: " << &p2.x << std::endl;
	std::cout << "Address of p2.y: " << &p2.y << std::endl;
	

	Function();
	Function();
	Function();
	return 0;
}