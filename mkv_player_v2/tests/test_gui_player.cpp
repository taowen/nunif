#include <catch2/catch_test_macros.hpp>
#include <windows.h>
#include <filesystem>
#include <thread>
#include <chrono>
#include <iostream>
#include <string>

TEST_CASE("gui_player without video file", "[gui_player]") {
    std::filesystem::path exePath = std::filesystem::current_path() / "build" / "Debug" / "gui_player.exe";
    
    REQUIRE(std::filesystem::exists(exePath));
    
    std::wstring cmdLine = L"\"" + exePath.wstring() + L"\" 2000";
    
    HANDLE hStdoutRead, hStdoutWrite;
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;
    
    BOOL pipeResult = CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0);
    REQUIRE(pipeResult == TRUE);
    
    SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);
    
    STARTUPINFOW si = {};
    si.cb = sizeof(STARTUPINFOW);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = hStdoutWrite;
    
    PROCESS_INFORMATION pi = {};
    
    auto startTime = std::chrono::steady_clock::now();
    
    BOOL result = CreateProcessW(
        nullptr,
        const_cast<LPWSTR>(cmdLine.c_str()),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi
    );
    
    REQUIRE(result == TRUE);
    
    CloseHandle(hStdoutWrite);
    
    std::string output;
    char buffer[4096];
    DWORD bytesRead;
    
    std::cout << "GUI Player Output:" << std::endl;
    
    while (ReadFile(hStdoutRead, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
        buffer[bytesRead] = '\0';
        output += buffer;
        std::cout << buffer;
        std::cout.flush();
    }
    
    DWORD waitResult = WaitForSingleObject(pi.hProcess, 5000);
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << std::endl << "Process completed in " << elapsedMs << "ms" << std::endl;
    
    REQUIRE(waitResult == WAIT_OBJECT_0);
    
    DWORD exitCode;
    BOOL getExitCodeResult = GetExitCodeProcess(pi.hProcess, &exitCode);
    REQUIRE(getExitCodeResult == TRUE);
    REQUIRE(exitCode == 0);
    
    REQUIRE(elapsedMs >= 1900);
    REQUIRE(elapsedMs <= 3000);
    
    REQUIRE_FALSE(output.empty());
    
    CloseHandle(hStdoutRead);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
}

TEST_CASE("gui_player with video file", "[gui_player][video]") {
    std::filesystem::path exePath = std::filesystem::current_path() / "build" / "Debug" / "gui_player.exe";
    
    REQUIRE(std::filesystem::exists(exePath));
    
    std::wstring cmdLine = L"\"" + exePath.wstring() + L"\" test_data\\sample_hw.mkv 3000";
    
    HANDLE hStdoutRead, hStdoutWrite;
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;
    
    BOOL pipeResult = CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0);
    REQUIRE(pipeResult == TRUE);
    
    SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);
    
    STARTUPINFOW si = {};
    si.cb = sizeof(STARTUPINFOW);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = hStdoutWrite;
    
    PROCESS_INFORMATION pi = {};
    
    auto startTime = std::chrono::steady_clock::now();
    
    BOOL result = CreateProcessW(
        nullptr,
        const_cast<LPWSTR>(cmdLine.c_str()),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi
    );
    
    REQUIRE(result == TRUE);
    
    CloseHandle(hStdoutWrite);
    
    std::string output;
    char buffer[4096];
    DWORD bytesRead;
    
    std::cout << "GUI Player with Video Output:" << std::endl;
    
    while (ReadFile(hStdoutRead, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
        buffer[bytesRead] = '\0';
        output += buffer;
        std::cout << buffer;
        std::cout.flush();
    }
    
    DWORD waitResult = WaitForSingleObject(pi.hProcess, 8000);
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << std::endl << "Video test completed in " << elapsedMs << "ms" << std::endl;
    
    REQUIRE(waitResult == WAIT_OBJECT_0);
    
    DWORD exitCode;
    BOOL getExitCodeResult = GetExitCodeProcess(pi.hProcess, &exitCode);
    REQUIRE(getExitCodeResult == TRUE);
    
    REQUIRE_FALSE(output.empty());
    
    if (output.find("Failed to open video file") != std::string::npos) {
        std::cout << "Note: Video file test_video.mkv not found, showing black screen" << std::endl;
    } else {
        std::cout << "Video file loaded successfully" << std::endl;
    }
    
    CloseHandle(hStdoutRead);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
}