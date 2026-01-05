#include<iostream>
#include<vector>

void Foreach(std::vector<int>& vec,void(*func)(int&)) {
	
	//如果这里不是引用传递，那么默认是值传递， 这里会先复制出一份value的值，从而使得这个vec的值不会改变
	for (int& value: vec) {
		func(value);
	}
}

void Print(int& val) {
	std::cout << val << std::endl;
}

int main() {
	auto lambda = [](int& a) {a++;};
	
	std::vector<int> vec = { 1,2,3,4,5 };
	Foreach(vec, lambda);
	Foreach(vec,Print);

	//lambda表达式的结构
	// [capture](parameters) -> return_type { body } []捕获lambda表达式外部的值
	// [=]捕获所有变量，值传递
	// [&]捕获所有变量，引用传递
	// [a]捕获a
	// [=, &b]捕获所有变量，值传递，b引用传递
	//...
	int a = 1;
	auto lam = [](int& val) mutable { val++;};
	lam(a);
	std::cout << a << std::endl; // 1
}