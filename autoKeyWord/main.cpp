#include<iostream>
#include<string>

int main() {
	int a[10];
	memset(a, 0, sizeof(a));
	int i = 0;
	//一种自动的类型推断... 很简单
	for (auto& value : a) {
		value += 1;
		std::cout << a[i] << std::endl;
		i++;
	}
}