#include<iostream>
#include<GLFW/glfw3.h>  

int main() {
   //这里就算我们include头文件，但是头文件里只有函数定义而没有函数实现，我们需要在项目链接中添加实际的目录+对应的lib文件
    //详细参考项目的linker的输入的常规/附加库目录+输入/附加依赖项   以及 c/c++/general/附加包含目录
    int a = glfwInit();
    return 0;  
}