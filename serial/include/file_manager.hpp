#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <chrono>
#include "rapidjson/document.h" 
#include "rapidjson/filewritestream.h" 
#include "rapidjson/writer.h" 
#include "rapidjson/ostreamwrapper.h"

using namespace std;

int readGraphFromFile(string, int*, vector<vector<int>>*);

int writeResultsToFile(string, int, vector<int>, chrono::duration<double>, chrono::duration<double>, chrono::duration<double>);

string vectorToString(const vector<int>*);