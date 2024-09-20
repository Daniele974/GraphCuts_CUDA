#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <chrono>
#include <ctime>
#include "rapidjson/document.h" 
#include "rapidjson/filewritestream.h" 
#include "rapidjson/writer.h" 
#include "rapidjson/ostreamwrapper.h"

using namespace std;

int readGraphFromFile(string, int&, int&, vector<vector<int>>&);

int writeResultsToFile(string, int, vector<int>, double, double, double, int, int);