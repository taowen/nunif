pushd build
cmake -S .. -B .
cmake --build . --config Release
popd