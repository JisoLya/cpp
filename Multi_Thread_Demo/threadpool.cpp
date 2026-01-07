#include "threadpool.h"

SafeQueue::SafeQueue(size_t cap) : cap_(cap), m_queue(std::queue<Task>()), is_shutdown(false) {
}
SafeQueue::~SafeQueue() {}
void SafeQueue::push(Task&& t) {
	std::unique_lock<std::mutex> lock(mtx_);
	// ����ֱ������
	not_full_cv.wait(lock, [this] { return m_queue.size() < cap_ || is_shutdown; });
	m_queue.push(std::move(t));
	not_empty_cv.notify_one(); // ֪ͨ������
}

Task SafeQueue::pop() {
	std::unique_lock<std::mutex> lock(mtx_);
	// ����ֱ������
	not_empty_cv.wait(lock, [this] { return !m_queue.empty() || is_shutdown ; });
	Task t = std::move(m_queue.front());
	m_queue.pop();
	not_full_cv.notify_one(); // ֪ͨ�������п�λ��
	return t;
}

void SafeQueue::shutdown(){
	std::unique_lock<std::mutex> lock(mtx_);
	is_shutdown = true;
	not_empty_cv.notify_all();
	not_full_cv.notify_all();
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
		if (is_stopped) break;
		Task t = task_queue.pop();
		t.run();
	}
}

void ThreadPool::producer_loop() {
	while (true) {
		if (is_stopped) break;
		Task t = buffer_queue.pop(); 
        
        task_queue.push(std::move(t));
	}
}

void ThreadPool::stop() {
	
}