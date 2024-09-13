#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include "rapidjson/document.h" 
#include "rapidjson/filewritestream.h" 
#include "rapidjson/writer.h" 
#include "rapidjson/ostreamwrapper.h"

int readBCSRGraphFromFile(std::string filename, int &V, int &E, int &source, int &sink, int **offset, int **column, int **capacities, int **forwardFlow);

int readGraphFromFileOLD(std::string filename, int &n, int **capacity);

int writeResultsToFile(std::string, int, std::vector<int>, std::chrono::duration<double>, std::chrono::duration<double>, std::chrono::duration<double>);