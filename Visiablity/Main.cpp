#include<iostream>
#include<array>
int main() {
	int example[5];
	//int 4 字节，32位，那么16进制表示需要4位
	for (int i = 0;i < 5;i++) {
		example[i] = i;
	}

	//allocate on heap
	int* array = new int[5];
	for (int i = 0;i < 5;i++) {
		array[i] = i + 1000;
	}
	//但是这个array需要自己保存size字段，可以用std::array
	std::array<int, 5> stdarray;
	for (int i = 0;i < stdarray.size();i++) {
		stdarray[i] = i + 2000;
	}

	std::cin.get();
	delete[] array;
}