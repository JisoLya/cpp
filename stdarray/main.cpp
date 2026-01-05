#include<iostream>
#include<array>


//这样我们就可以使用std::array了
template<size_t N>
void Print(const std::array<int, N>& arr) {
	for (int i = 0;i < arr.size();i++) {
		std::cout << arr[i] << " ";
	}
}


int main() {
	std::array<int, 6> arr;
	arr[0] = 3;
	arr[1] = 3;
	arr[2] = 3;
	arr[3] = 3;
	arr[4] = 4;
	Print(arr);
	//事实上， array不存在一个变量来存储这个size()，查看源码可以发现他的size()是一个常量表达式
	//模板类在编译的时候被创建，当我们传入模板参数size_t的时候，编译得到的代码会把模板参数直接传递给size()方法的返回值
}