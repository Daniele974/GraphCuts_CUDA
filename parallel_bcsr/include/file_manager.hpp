#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include "rapidjson/document.h" 
#include "rapidjson/filewritestream.h" 
#include "rapidjson/writer.h" 
#include "rapidjson/ostreamwrapper.h"

int readGraphFromFile(std::string filename, int &n, int **capacity);

int writeResultsToFile(std::string, int, std::vector<int>, float, float, float, int, int);