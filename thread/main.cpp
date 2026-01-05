#include<iostream>
#include<thread>

static bool is_Finish = false;

void doWork() {
	while (!is_Finish) {
		std::cout << "Working" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

}

int main() {
	std::thread worker(doWork);
	std::cin.get();
	// join会让当前线程等待worker线程结束
	worker.join();
	is_Finish = true;
	std::cout << "Finished" << std::endl;
	return 0;
}