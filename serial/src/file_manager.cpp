#include "../include/file_manager.hpp"

using namespace std;

int readGraphFromFile(string filename, int *n, vector<vector<int>> *capacity)
{
    /// Apertura file
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "File not found" << endl;
        return 1;
    }

    /// Lettura file
    string line;
    int lineno = 0;
    
    // Espressione regolare per trovare numeri
    regex number_regex(R"(\d+)");
    
    while (file)
    {
        getline(file, line);
        lineno++;
        auto words_begin = sregex_iterator(line.begin(), line.end(), number_regex);
        auto words_end = sregex_iterator();
        vector<int> numbers;
        for (sregex_iterator i = words_begin; i != words_end; ++i) {
            smatch match = *i;
            numbers.push_back(stoi(match.str()));
        }
        if (line[0] == 'n' && numbers.size() > 0) {
            *n = numbers[0];
            (*capacity).resize(*n, vector<int>(*n, 0));
        }else if(line[0] == 'e' && numbers.size() > 2){ 
            (*capacity)[numbers[0]][numbers[1]] = numbers[2];
        }else{
            cout << "Error in line " << lineno << endl;
        }
        numbers.clear();
    }
    
    file.close();
    
    return 0;
}