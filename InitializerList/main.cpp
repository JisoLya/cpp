#include<iostream>
#include<string>

class Entity {
public:
	std::string name;
	int x;
	Entity(): name("Unknow"), x(0) {
		std::cout << "Deffault Entity constructor!" << std::endl;
	}

	//这个构造函数实际上做了两个事，先根据初始化列表初始化name，再重新对name进行赋值
	Entity(int) : x(0) {
		std::cout << "Before set : " << this->name << std::endl;
		this->name = "onePaa";
		//跟下边这句话做的事情一样
		//this->name = std::string("onePaa")
		std::cout << this->name << std::endl;
	}
};

class Example {
public:
	Example() {
		std::cout << "Deffault Example constructor!" << std::endl;
	}

	Example(int x) {
		std::cout << "Create Example with var : " << x << std::endl;
	}
};

class Player {
public:
	std::string name;
	Example e;
	//case2
	Player(){
		this->name = "Soyaa";
		std::cout << "Deffault Player constructor!" << std::endl;
		this->e = Example(8);
	}

	//case3
	/*Player() :e(Example(8)), name("Soyaa") {
		std::cout << "Deffault Player constructor!" << std::endl;
	}*/
};

int main() {
	//case1
	Entity e(0);

	//case2
	//可以观察到，先调用了Example的默认构造函数，然后调用了Player的默认构造函数，再创建了一个新的Example对象
	//由于初始化列表先于构造函数执行，所以e对象的构造函数会在Player的构造函数之前被调用
	//如果我们使用case3的方式，那么只会创建一个Example对象
	Player p;

}