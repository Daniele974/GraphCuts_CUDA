#include "../include/file_manager.hpp"

using namespace std;
using namespace rapidjson;

int readGraphFromFile(string filename, int &n, int &e, vector<vector<int>> &capacity)
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
    e = 0;
    while (getline(file, line))
    {
        lineno++;
        auto words_begin = sregex_iterator(line.begin(), line.end(), number_regex);
        auto words_end = sregex_iterator();
        vector<int> numbers;
        for (sregex_iterator i = words_begin; i != words_end; ++i) {
            smatch match = *i;
            numbers.push_back(stoi(match.str()));
        }
        if (line[0] == 'n' && numbers.size() > 0) {
            n = numbers[0];
            capacity.resize(n, vector<int>(n, 0));
        }else if(line[0] == 'e' && numbers.size() > 2){ 
            capacity[numbers[0]][numbers[1]] = numbers[2];
            e++;
        }else{
            if(!line.empty()) cout << "Error in line " << lineno << endl;
        }
        numbers.clear();
    }
    
    file.close();
    
    return 0;
}

int writeResultsToFile(string filename, int maxFlow, vector<int> minCut, double initializationTime, double executionTime, double totalTime, int n, int e){
    // Creazione del documento JSON 
    Document d; 
    d.SetObject(); 

    // Aggiunta campi al documento 
    d.AddMember("maxFlow", maxFlow, d.GetAllocator()); 
    Value minCutSet(kArrayType);
    for (int i = 0; i < minCut.size(); i++) minCutSet.PushBack(minCut[i], d.GetAllocator());
    d.AddMember("minCut", minCutSet, d.GetAllocator());
    d.AddMember("initializationTime", initializationTime, d.GetAllocator());
    d.AddMember("executionTime", executionTime, d.GetAllocator());
    d.AddMember("totalTime", totalTime, d.GetAllocator());
    d.AddMember("V", n, d.GetAllocator());
    d.AddMember("E", e, d.GetAllocator());
    
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
    ofstream file(filename); 
    if (!file.is_open()) {
        cout << "Errore apertura file" << endl;
        return 1;
    }
    rapidjson::OStreamWrapper osw(file);
  
    // Scrittura dei dati JSON nel file
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw); 
    d.Accept(writer); 

    return 0;
}