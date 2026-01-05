#include<iostream>
#include<string>
class Entity {
private:
	std::string pos;
	mutable int debug_var;
public:
	Entity() : pos("default"), debug_var(0) {}
	void Print_pos() const {
		//加入mutable允许修饰为const的函数修改变量
		debug_var++;
		std::cout << "Position: " << pos << std::endl;
	}
};

int main() {
	//const才可以调用const
	const Entity e;
	e.Print_pos();

	//lambda表达式 [capture list] (parameter list) -> return type { function body }

	//捕获方式:
	//值捕获 = 或直接传值，值捕获的lambda表达式会复制捕获的变量，并且在function body中不可修改,如果用lambda被mutable修饰，那么可以
	// 在function body修改传递的值
	// 
	//引用捕获 & 或者 &var：可以修改变量，并且是修改原变量
	int x = 20;
	auto f = [x](int a) mutable{
		std::cout << "Before addition x = " << x << std::endl;
		x += a;	
	};
	f(3);
	std::cout << x << std::endl;

}