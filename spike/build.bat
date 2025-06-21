pushd spike\build
cmake -S .. -B .
cmake --build . --config Debug
popd