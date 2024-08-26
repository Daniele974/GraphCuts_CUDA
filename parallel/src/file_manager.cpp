#include "../include/file_manager.hpp"

bool debug = false;

int readGraphFromFile(std::string filename, int &n, int *capacity)
{
    /// Apertura file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "File not found" << std::endl;
        return 1;
    }

    /// Lettura file
    std::string line;
    int lineno = 0;
    
    std::vector<std::string> linewords;
    
    while (std::getline(file, line))
    {
        lineno++;

        if (line.empty()) continue;

        std::string word;
        for (char c : line)
        {
            if(std::isalnum(c)){
                word += c;
            }
            else if (!word.empty()){
                linewords.push_back(word);
                word.clear();
            }
        }

        // Aggiungi l'ultima parola se la linea non termina con un delimitatore
        if (!word.empty()) {
            linewords.push_back(word);
        }

        if(linewords[0] == "n" && linewords.size() > 1){
            if (debug) std::cout << "n: " << linewords[1] << std::endl;
            n = std::stoi(linewords[1]);
            capacity = (int *)malloc(n * n * sizeof(int));
        }
        else if(linewords[0] == "e" && linewords.size() > 3){
            if (debug) std::cout << "e: " << linewords[1] << " " << linewords[2] << " " << linewords[3] << std::endl;
            capacity[std::stoi(linewords[1]) * n + std::stoi(linewords[2])] = std::stoi(linewords[3]);
        }
        else{
            if(!linewords.empty()) std::cout << "Error in line " << lineno << std::endl;
        }

        linewords.clear();
    }
    
    file.close();

    return 0;
}