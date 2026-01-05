#include<iostream>

class Entity {
private:
	int x;
public:
	Entity(){}
	Entity(int x) : x(x){}
	void Print() const {
		std::cout << "x = " << x << std::endl;
	}
};

class ScpoePtr {
private:
	Entity* m_e;
public:
	
	ScpoePtr(Entity* en) : m_e(en) {
	}

	~ScpoePtr() {
		delete m_e;
	}

	Entity* GetEntity() const {
		return m_e;
	}

	const Entity* operator->() const {
		return m_e;
	}
};

//
struct Vector3
{
	float x, y, z;
};

int main() {
	//隐式类型转换
	ScpoePtr p = new Entity(2);
	//如果此时我想调用p指针的Print方法，由于m_e是private的，所以无法获取，那么我需要去写一个方法来获取m_e
	//此时的调用如下：
	p.GetEntity()->Print();
	//显然十分的不好用，那么这时我们可以重载->操作符
	//这样我们就可以直接调用p->Print()了
	p->Print();

	//同样，我们可以利用箭头函数来获取变量在结构体中的偏移量

	int offset = (int) & (((Vector3*)nullptr)->x);
	std::cout << "offset of x: " << offset << std::endl;

	offset = (int)&(((Vector3*)nullptr)->y);
	std::cout << "offset of y: " << offset << std::endl;

	offset = (int)&(((Vector3*)nullptr)->z);
	std::cout << "offset of z: " << offset << std::endl;
}