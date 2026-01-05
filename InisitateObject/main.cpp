#include<iostream>
#include<string>
using String = std::string;


class Entity {
private:
	String name;
public:
	
	Entity():name("Unknow") {};
	Entity(const String& name) : name(name) {}

	const String GetName() const {
		return name;
	}
};

void Function() {
	//当我们希望这个变量的生命周期存在于函数之外时
	//我们需要再堆中申请内存
}

int main() {

	//这里被释放掉了
	Entity* e;
	{
		Entity entity("Soya");
		e = &entity;
		std::cout << e->GetName() << std::endl;
	}
	//在代码块之外，entity被释放掉了,这时是危险操作, __不要__这么做
	//std::cout << e->GetName() << std::endl;

	//new和malloc的唯一区别是new会调用对象的构造函数
	{
		Entity* entity = new Entity("Soya");
		e = entity;
		std::cout << e->GetName() << std::endl;
	}
	//这里还是可以访问到e的
	std::cout << e->GetName() << std::endl;
	delete e;


	//
	int* b = new int[50]; // 200bytes

	delete[] b;
	
	std::cin.get();


}