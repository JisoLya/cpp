#include<iostream>
#include<string>
#include<memory>


class Entity {
public:
	int x, y;

	Entity() {
	}
	Entity(int x, int y) :x(x), y(y) {
	}
	~Entity() {
		std::cout << "Destory Entity!" << std::endl;
	}
	void Print() {
		std::cout << "x = " << this->x << ", y = " << this->y << std::endl;
	}
};

int main() {
	//case1 unique_ptr
	{
		//可以这样创建unique_ptr
		std::unique_ptr<Entity> entity1(new Entity(1, 2));
		
		//这样是不允许的，因为unique_ptr的构造函数声明为了explicit
		//std::unique_ptr<Entity> entity1 = new Entity(1, 2);
		
		//这样是推荐做法
		std::unique_ptr<Entity> entity = std::make_unique<Entity>();
		//这种声明也是不被允许的，因为std::unique_ptr的拷贝构造函数声明为delete
		//std::unique_ptr<Entity> e = entity;
	}

	//case2 shared_ptr:引用计数
	{	

		std::shared_ptr<Entity> e0;
		//这里是因为shared_ptr需要内存来保存引用计数，所以不能使用new来创建(因为new只会分配Entity的内存，而不存在这个保存引用计数部分的内存)
		{
			std::shared_ptr<Entity> entity = std::make_shared<Entity>();
			e0 = entity;
		}
		//利用shared_pointer走出上个作用域的时候，这个entity其实不会被删除，因为shared_ptr
		//仍然持有对Entity的引用
	}

	//case3 weak_ptr
	{

		std::weak_ptr<Entity> e0;
		//这里是因为shared_ptr需要内存来保存引用计数，所以不能使用new来创建
		{
			std::shared_ptr<Entity> entity = std::make_shared<Entity>();
			e0 = entity;
		}
		//当这个是weak_ptr的时候，不会给引用计数+1，那么在走出作用域的时候，entity会被删除
	}
}