pushd build
cmake -S .. -B .
cmake --build . --config Debug
popd