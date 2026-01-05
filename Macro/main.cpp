#include<iostream>
#define WAIT std::cin.get()
//在编译器的预处理阶段，所有的WAIT会被替换为std::cin.get()，

#ifdef PR_DEBUG 
#define LOG(x) std::cout << x << std::endl;
#elif defined PR_RELEASE
#define LOG(x)
#endif


//同时可以利用 \来定义多行的宏
int main() {
	LOG(5);
}