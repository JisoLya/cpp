#include "threadpool.h"

SafeQueue::SafeQueue(size_t cap) : cap_(cap), m_queue(std::queue<Task>()) {
}
SafeQueue::~SafeQueue() {}
void SafeQueue::push(Task&& t) {
	std::unique_lock<std::mutex> lock(mtx_);
	// 阻塞直到不满
	not_full_cv.wait(lock, [this] { return m_queue.size() < cap_; });
	m_queue.push(std::move(t));
	not_empty_cv.notify_one(); // 通知消费者
}

Task SafeQueue::pop() {
	std::unique_lock<std::mutex> lock(mtx_);
	// 阻塞直到不空
	not_empty_cv.wait(lock, [this] { return !m_queue.empty(); });
	Task t = std::move(m_queue.front());
	m_queue.pop();
	not_full_cv.notify_one(); // 通知生产者有空位了
	return t;
}


ThreadPool::ThreadPool(int worker_count)
	: is_stopped(false), task_queue(10), buffer_queue(10) {
	for (int i = 0; i < worker_count; ++i) {
		workers.emplace_back([this]() {
			this->worker_loop();
		});
	}

	producer_thread_ = std::thread([this]() {
		this->producer_loop();
		});
}
ThreadPool::~ThreadPool() {
	this->stop();
}
void ThreadPool::addTask(Task&& t) {
	buffer_queue.push(std::move(t));
}

void ThreadPool::worker_loop() {
	while (true) {
		Task t = task_queue.pop();
		t.run();
	}
}

void ThreadPool::producer_loop() {
	while (true) {
		Task t = buffer_queue.pop(); 
        
        // 2. push 会自动阻塞直到 task_queue 有空位，放进去后会自动通知 worker
        task_queue.push(std::move(t));
	}
}

void ThreadPool::stop() {

}