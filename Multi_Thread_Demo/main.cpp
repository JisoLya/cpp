#include "threadpool.h"
#include <chrono>
#include <iomanip>

// 模拟一个耗时任务
void long_task(int id) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    // 修改这里：
    struct tm buf;
    localtime_s(&buf, &in_time_t); // 将结果存入 buf 中

    std::cout << "[任务 " << id << "] 开始执行 | 线程 ID: "
        << std::this_thread::get_id() << " | 时间: "
        << std::put_time(&buf, "%X") << std::endl; // 传入 &buf

    // 模拟不同长度的计算任务
    std::this_thread::sleep_for(std::chrono::milliseconds(500 + (id % 500)));

    std::cout << "[任务 " << id << "] 完成 √" << std::endl;
}

int main() {
    std::cout << "--- 正在初始化线程池 (4个工作线程) ---" << std::endl;
    ThreadPool pool(4);

    // 1. 快速生产 20 个任务
    std::cout << "--- 正在添加 20 个任务 ---" << std::endl;
    for (int i = 1; i <= 20; ++i) {
        // 使用你的 Task 类包装函数
        pool.addTask(Task(i, long_task, i));

        // 如果你的 addTask 逻辑正确，这里在超过缓冲区大小时会阻塞
        if (i % 5 == 0) {
            std::cout << "已添加 " << i << " 个任务..." << std::endl;
        }
    }

    // 2. 在任务执行期间打印线程池状态
    std::this_thread::sleep_for(std::chrono::seconds(1));
    pool.print_info();

    // 3. 等待一段时间，让任务处理完
    std::cout << "--- 主线程等待中 ---" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 4. 退出（会自动触发析构中的 stop）
    std::cout << "--- 测试结束，正在关闭线程池 ---" << std::endl;

    return 0;
}