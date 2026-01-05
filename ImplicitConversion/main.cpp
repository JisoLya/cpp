#include<iostream>



class Entity {
private:
	std::string m_name;
	int m_age;
public:
	Entity(const std::string& name)
		:m_name(name), m_age(0){ }

	Entity(int age)
		:m_name("Unknown"), m_age(age) { }

	//当把构造函数声明为explicit的时候，说明这个构造函数必须显示的调用
	/*explicit Entity(int age)
		:m_name("Unknown"), m_age(age) {
	}*/

	std::string get_name() const {
		return this->m_name;
	}

	int get_age() const {
		return this->m_age;
	}
};


void PrintEntity(const Entity& en) {
	std::cout << "Entity name: " << en.get_name() << "\nEntity age: " << en.get_age() << std::endl;
}

int main() {
	//这里会隐式的类型转化,不建议这么用
	PrintEntity(22);

	//这样会报错
	//PrintEntity("Soya");
	//这里的隐式转换为 "Soya"(char[]) -> std::string -> 调用了字符串的构造函数
	PrintEntity(Entity("Soya"));
}