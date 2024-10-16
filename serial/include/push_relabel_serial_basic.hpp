#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <chrono>
#include "file_manager.hpp"

using namespace std;

void push(int, int);
void relabel(int);
void pushRelabel(int);
vector<int> findMinCutSetFromS();
vector<int> findMinCutSetFromT();
int executePushRelabel(string, string, bool);