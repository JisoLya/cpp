#include<iostream>


int m_static = 10;

//extern int test;
//这个代码会去外部找定义的test变量

class Log {
public:
	const int LogLevelError = 0;
	const int LogLevelWarning = 1;
	const int LogLevelInfo = 2;
	
private:
	int level = LogLevelInfo;
public:
	Log(int logLevel){
		this->level = logLevel;
	}

	void Info(const char* message) {
		if (level >= LogLevelInfo)
			std::cout << "[INFO]: " << message << std::endl;
	}

	void Warning(const char* message) {
		if (level >= LogLevelWarning)
			std::cout << "[WARNING]: " << message << std::endl;
	}

	void Error(const char* message) {
		if (level >= LogLevelError)
			std::cout << "[ERROR]: " << message << std::endl;
	}

};

int main() {
	Log log(1);

	log.Info("Hello World");
	log.Warning("This is a warning");
	log.Error("This is an error");
	return 0;
}