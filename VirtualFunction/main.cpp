#include<iostream>

//如果有纯虚函数，那么不可以实例化这个类
class Entity {
public:
	virtual void printName() {
		std::cout << "Entity" << std::endl;
	};
	//virtual void mustBeOverridden() = 0; // 纯虚函数，必须被重写
};


class Player : public Entity {
public:

	//override关键字可以帮助我们检查代码是否规范
	void printName() override {
		std::cout << "Player" << std::endl;
	};

	void mustBeOverridden() {
		std::cout << "Player must be overridden" << std::endl;
	};
};

void print(Entity& const entity) {
	entity.printName();
}

int main() {
	Player player;
	Entity entity;

	print(player); // Player
	print(entity); // Entity
}