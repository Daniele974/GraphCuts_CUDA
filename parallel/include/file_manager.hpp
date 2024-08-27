#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "rapidjson/document.h" 
#include "rapidjson/filewritestream.h" 
#include "rapidjson/writer.h" 
#include "rapidjson/ostreamwrapper.h"

int readGraphFromFile(std::string filename, int &n, int **capacity);

int writeResultsToFile(std::string, int, std::vector<int>, std::chrono::duration<double>, std::chrono::duration<double>, std::chrono::duration<double>);