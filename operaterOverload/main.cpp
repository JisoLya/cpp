#include<iostream>

struct Vector2 {
	float x, y;

	Vector2(float x, float y) : x(x), y(y) {}

	Vector2 Add(const Vector2& other) const {
		//return operator+(other);
		//return *this + other;

		return Vector2(this->x + other.x, this->y + other.y);
	}

	//operator overloading
	Vector2 operator+(const Vector2& other) const {
		return Add(other);
	}

	Vector2 Multiply(const Vector2& other) const {
		return Vector2(this->x * other.x, this->y * other.y);
	}

	Vector2 operator*(const Vector2& other) const {
		return Multiply(other);
	}

	/*
	如果是重载双目操作符（即为类的成员函数），就只要设置一个参数作为右侧运算量，而左侧运算量就是对象本身。
	而 >>  或<< 左侧运算量是 cin或cout 而不是对象本身，所以不满足后面一点就只能申明为友元函数了。
	如果一定要声明为成员函数，只能成为如下的形式：
	ostream & operator<<(ostream &output)
	*/
	friend std::ostream& operator<<(std::ostream& stream, const Vector2& vec) {
		stream << "Friend func" << "Vector2 x = " << vec.x << ",y =  " << vec.y << std::endl;
		return stream;
	}

	std::ostream& operator<<(std::ostream& stream) {
		stream << "Non-Friend func" << "Vector2(" << this->x << ", " << this->y << ")" << std::endl;
		return stream;
	}
};

int main() {
	Vector2 v1(1.0f, 2.0f);
	Vector2 v2(3.0f, 4.0f);
	Vector2 result = v1 + v2; // Calls operator+
	std::cout << "Result of addition: (" << result.x << ", " << result.y << ")" << std::endl;
	result = v1 * v2; // Calls operator*
	std::cout << "Result of multiplication: (" << result.x << ", " << result.y << ")" << std::endl;


	std::cout << result << std::endl;
	//不声明为友元函数只能这样调用了
	result << std::cout << std::endl;
	return 0;
}