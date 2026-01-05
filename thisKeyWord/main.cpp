#include<iostream>

class Entity {
private:
	int x, y;

public:
	Entity(int x, int y) {
		this->x = x;
		this->y = y;
	}
	~Entity() {
		std::cout << "Entity destroyed at (" << this->x << ", " << this->y << ")\n";
	}

	int GetX() const {
		//不可改变内容的
		const Entity* e = this;
		//这个语句会报错，因为这里声明的a是一个不可改变指向的指针，但是内容可以变,这个和声明为const的函数冲突了，因为声明为const的函数不能修改成员的值。
		//Entity* const a = this;

	}

};


int main() {
	int num = 1;
	int num2 = 2;
	int& const a = num;

	std::cout << a << std::endl;
	a = num2;
	std::cout << a << std::endl;
}