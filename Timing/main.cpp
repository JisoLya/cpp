#include<iostream>
#include<chrono>
#include<thread>

class Timer {
public:
	std::chrono::time_point<std::chrono::steady_clock> start, end;
	std::chrono::duration<float> duration;
	Timer() {
		start = std::chrono::steady_clock::now();
	}

	~Timer() {
		end = std::chrono::steady_clock::now();
		duration = end - start;
		
		int ms = duration.count() * 1000;
		std::cout << "Elapsed time: " << ms << "ms" << std::endl;
	}
};

int main() {
	Timer t;
	for (int i = 0;i < 100000;i++) {
		std::cout << "Hello World!\n";
	}
}