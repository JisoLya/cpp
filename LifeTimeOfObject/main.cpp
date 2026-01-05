#include<iostream>

class Entity {
public:
	int x, y;

	Entity(){
		std::cout << "Create Entity!" << std::endl;
	}
	Entity(int x, int y) : x(x), y(y) {}

	~Entity() {
		std::cout << "Destory Entity" << std::endl;
	}

};


class ScopePtr {
public:
	Entity* e;

	ScopePtr(Entity* ptr):e(ptr) {
	}

	~ScopePtr() {
		delete e;
	}
};

int main() {
	{
		//这里发生了隐式转换
		//这样可以利用栈来管理堆的内存资源
		ScopePtr e = new Entity();
	}

}