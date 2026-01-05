#include<iostream>  
#include<string>  

//我们打印的时候不希望修改字符串并且也不希望传递一个字符串的拷贝(这个操作十分耗时)
void PrintString(const std::string& str) {  
	std::cout << str << std::endl;  
}  

int main() {  
	const char* name = "Soya";  
	std::string str = std::string("Hello, ") + name + "!";  
	bool find = str.find("Soya") != std::string::npos;
	std::cout << "find: " << find << std::endl;
	PrintString(str);


	/*
		string literials
		这里会发生截断
	*/
	//这样才可以，字符串字面量本质是const char*类型的
	char* literials_name = (char*)"Soya";
	const char literial[13] = "Soya\0 hello";
	std::cout << strlen(literial) << std::endl;

	//分别为
	const char* myname = "Hello, Soya!";
	std::cout << sizeof(myname) << std::endl;
	std::cout << "size of char: " << sizeof(char) << std::endl;
	const wchar_t* wstr = L"Hello, Soya!";
	std::cout << sizeof(wstr) << std::endl; 
	std::cout << "size of wchar_t: " << sizeof(wchar_t) << std::endl;
	const char16_t* u16str = u"Hello, Soya!";
	std::cout << sizeof(u16str) << std::endl;
	std::cout << "size of char16_t: " << sizeof(char16_t) << std::endl;
	const char32_t* u32str = U"Hello, Soya!";
	std::cout << sizeof(u32str) << std::endl;
	std::cout << "size of char32_t: " << sizeof(char32_t) << std::endl;

}