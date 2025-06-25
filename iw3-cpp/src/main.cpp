#include <iostream>
#include <format>
#include <string_view> 

int main() {
    constexpr std::string_view message = "Hello, World!";
    std::cout << std::format("Message: {}\n", message);
    return 0;
}