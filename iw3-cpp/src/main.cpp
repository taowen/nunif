#include <iostream>
#include <format>  // C++20 特性
#include <string_view>  // C++17 特性

int main() {
    constexpr std::string_view message = "Hello, World!";
    
    std::cout << std::format("Message: {}\n", message);
    return 0;
}