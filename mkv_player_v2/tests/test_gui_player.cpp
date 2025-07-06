#include <catch2/catch_test_macros.hpp>
#include <windows.h>
#include <filesystem>
#include <thread>
#include <chrono>
#include <iostream>
#include <string>

TEST_CASE("gui_player basic functionality", "[gui_player]") {
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