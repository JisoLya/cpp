#pragma once
#include <iostream>
#include <functional>
#include <string>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include <thread>

// 任务类， 封装成一个函数
class Task {
private:
    int t_id;
    std::function<void()> t_action;

public:
	template<typename F, typename... Args>
    Task(int id, F&& f, Args&&... args)
		: t_id(id), t_action(std::bind(std::forward<F>(f), std::forward<Args>(args)...)) {
	}
    ~Task(){}

    void run() {
        if (t_action) {
            t_action();
        }
    }
};

// 2. 线程安全队列类
class SafeQueue {
private:
	std::queue<Task> m_queue;
    size_t cap_;
	std::mutex mtx_;

    std::condition_variable not_full_cv;
    std::condition_variable not_empty_cv;

public:
    SafeQueue(size_t cap);
    ~SafeQueue();
    bool is_empty() {
		std::lock_guard<std::mutex> lock(mtx_);
        return m_queue.empty();
    }
    bool is_full() {
		std::lock_guard<std::mutex> lock(mtx_);
		return m_queue.size() >= cap_;
    }
	void push(Task&& t);
	Task pop();
};

// 3. 核心处理器类
class ThreadPool {
public:
    ThreadPool(int worker_count);
    ~ThreadPool();

    void addTask(Task&& t);
    void stop();
    void print_info() {
        std::cout << "ThreadPool info: is_stopped = " << is_stopped 
                  << ", task_queue size = " << (is_stopped ? 0 : task_queue.is_empty() ? 0 : 1)
			<< ", worker count = " << workers.size() << std::endl;
    };

private:
    void worker_loop(); // 核心线程循环
	void producer_loop(); // 生产者线程循环
	bool is_stopped;

	SafeQueue task_queue;
    SafeQueue buffer_queue;

	std::vector<std::thread> workers;

	std::thread producer_thread_; // 向队列中添加任务的线程
};
