#include<iostream>

class Entity {
public:
	float X, Y;

	void Move(float xa, float ya) {
		X += xa;
		Y += ya;
	}
};

class Player : public Entity {
public:
	const char* name;

	Player(const char* name){
		this->name = name;
		X = 0.0f;
		Y = 0.0f;
	}
	void PrintName() {
		std::cout << "Player name: " << name << std::endl;
	}
};

int main() {
	Player player("Soya");
	player.Move(5.0f, 3.0f);
	std::cout << player.X << ", " << player.Y << std::endl;
}