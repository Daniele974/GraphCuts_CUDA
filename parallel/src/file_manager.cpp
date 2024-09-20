#include "../include/file_manager.hpp"

bool debug = false;

int readGraphFromFile(std::string filename, int &n, int **capacity){
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

            (*capacity) = (int *)malloc(n * n * sizeof(int));
            std::fill_n(*capacity, n * n, 0);
        }
        else if(linewords[0] == "e" && linewords.size() > 3){
            if (debug) std::cout << "e: " << std::stoi(linewords[1]) << " " << std::stoi(linewords[2]) << " " << std::stoi(linewords[3]) << std::endl;
            (*capacity)[std::stoi(linewords[1]) * n + std::stoi(linewords[2])] = std::stoi(linewords[3]);
        }
        else{
            if(!linewords.empty()) std::cout << "Error in line " << lineno << std::endl;
        }

        linewords.clear();
    }
    
    file.close();

    return 0;
}

int writeResultsToFile(std::string filename, int maxFlow, std::vector<int> minCut, float initializationTime, float executionTime, float totalTime, int V, int E){
    // Creazione del documento JSON 
    rapidjson::Document d; 
    d.SetObject(); 

    // Aggiunta campi al documento 
    d.AddMember("maxFlow", maxFlow, d.GetAllocator()); 
    rapidjson::Value minCutSet(rapidjson::kArrayType);
    for (int i = 0; i < minCut.size(); i++) minCutSet.PushBack(minCut[i], d.GetAllocator());
    d.AddMember("minCut", minCutSet, d.GetAllocator());
    d.AddMember("initializationTime", initializationTime, d.GetAllocator());
    d.AddMember("executionTime", executionTime, d.GetAllocator());
    d.AddMember("totalTime", totalTime, d.GetAllocator());
    d.AddMember("V", V, d.GetAllocator());
    d.AddMember("E", E, d.GetAllocator());

    // Generazione del timestamp
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
 
    // Formattazione del timestamp
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &tm);
 
    // Trovare la posizione dell'estensione .json
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) {
        pos = filename.length();  // Se non c'è un'estensione, appenderlo alla fine
    }
 
    // Creare il nuovo nome del file con il timestamp
    filename.insert(pos, "_" + std::string(timestamp));
    
    // Apertura file di output
    std::ofstream file(filename); 
    if (!file.is_open()) {
        std::cout << "Errore apertura file" << std::endl;
        return 1;
    }
    rapidjson::OStreamWrapper osw(file);
  
    // Scrittura dei dati JSON nel file
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw); 
    d.Accept(writer); 

    return 0;
}