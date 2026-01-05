#include<iostream>

/*
* 拷贝构造函数
*/

class String {
private:
	char* m_buffer;
	int m_size;
public:
	String(const char* string) {
		m_size = strlen(string);
		m_buffer = new char[m_size + 1];
		memcpy(m_buffer, string, m_size);
		m_buffer[m_size] = '\0';
	}

	//cpp默认会提供一个拷贝构造函数,大概长这样，但是实际上还是浅拷贝
	/*String(const String& other): m_buffer(other.m_buffer), m_size(other.m_size) {
	}*/

	String(const String& other) : m_size(other.m_size) {
		std::cout << "拷贝构造函数被调用" << std::endl;
		m_buffer = new char[m_size + 1];
		memcpy(m_buffer, other.m_buffer, m_size + 1);
	}

	~String() {
		delete[] m_buffer;
	}
	//友元函数，可以访问类的私有成员
	friend std::ostream& operator<<(std::ostream& os, const String& str);
};

std::ostream& operator<<(std::ostream& os, const String& str) {
	os << str.m_buffer;
	return os;
}

//如果用这种方式传递，会默认调用拷贝构造函数，然后再释放掉
/*
void PrintString(String str) {
	std::cout << str << std::endl;
}
*/
//这样就是引用传递，不会调用拷贝构造函数
void PrintString(const String& str) {
	std::cout << str << std::endl;
}

int main() {
	String str = "Hello";
	String s = str;
	
	//添加拷贝构造函数之前
	//这里会发生错误，打断点会发现，这里的s=str只复制了指针，而堆上的内存并没有被复制，s和str的m_buffer指向同一块内存，而走出作用域之后会重复释放内存，
	//这不是我们想要的结果
	std::cout << str << std::endl;
	std::cout << s << std::endl;

	
	PrintString(str);
	std::cin.get();
}