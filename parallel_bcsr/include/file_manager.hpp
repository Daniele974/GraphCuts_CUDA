#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include "rapidjson/document.h" 
#include "rapidjson/filewritestream.h" 
#include "rapidjson/writer.h" 
#include "rapidjson/ostreamwrapper.h"

int readBCSRGraphFromFile(std::string filename, int &V, int &E, int &realE, int &source, int &sink, int **offset, int **column, int **capacities, int **forwardFlow);

int readBCSRGraphFromDIMACSFile(std::string filename, int &V, int &E, int &realE, int &source, int &sink, int **offset, int **column, int **capacities, int **forwardFlow);

int writeResultsToFile(std::string, int, std::vector<int>, float, float, float, int, int);