#include "../include/file_manager.hpp"

bool debug = false;

int readBCSRGraphFromFile(std::string filename, int &V, int &E, int &source, int &sink, int **offset, int **column, int **capacities, int **forwardFlow){
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
    
    std::vector<std::vector<std::pair<int, int>>> graph;

    int numEdges = 0;

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
            V = std::stoi(linewords[1]);
            graph.resize(V);
        }
        else if(linewords[0] == "e" && linewords.size() > 3){
            if (debug) std::cout << "e: " << std::stoi(linewords[1]) << " " << std::stoi(linewords[2]) << " " << std::stoi(linewords[3]) << std::endl;
            int u = std::stoi(linewords[1]);
            int v = std::stoi(linewords[2]);
            int c = std::stoi(linewords[3]);
            
            int insertIndex = -1;
            if(graph[u].empty()){
                graph[u].push_back(std::make_pair(v, c));
                numEdges++;
            }else if(graph[u].back().first < v){
                graph[u].push_back(std::make_pair(v, c));
                numEdges++;
            }else{
                for (int i = 0; i < graph[u].size(); i++){   
                    if(graph[u][i].first == v){
                        graph[u][i].second = c;
                        break;
                    }
                    if(graph[u][i].first > v){
                        insertIndex = i;
                        graph[u].insert(graph[u].begin() + insertIndex, std::make_pair(v, c));
                        numEdges++;
                        break;
                    }
                }
            }
            
            insertIndex = -1;
            if(graph[v].empty()){
                graph[v].push_back(std::make_pair(u, 0));
                numEdges++;
            }else if(graph[v].back().first < u){
                graph[v].push_back(std::make_pair(u, 0));
                numEdges++;
            }else{
                for (int i = 0; i < graph[v].size(); i++){   
                    if(graph[v][i].first == u){
                        break;
                    }
                    if(graph[v][i].first > u){
                        insertIndex = i;
                        graph[v].insert(graph[v].begin() + insertIndex, std::make_pair(u, 0)); // Aggiunta arco inverso
                        numEdges++;
                        break;
                    }
                }
            }
        }
        else{
            if(!linewords.empty()) std::cout << "Error in line " << lineno << std::endl;
        }

        linewords.clear();
    }

    E = numEdges;

    (*offset) = (int *)malloc((V + 1) * sizeof(int));
    (*column) = (int *)malloc(E * sizeof(int));
    (*capacities) = (int *)malloc(E * sizeof(int));
    (*forwardFlow) = (int *)malloc(E * sizeof(int));

    for (int i = 0; i < V; i++){
        (*offset)[i] = i == 0 ? 0 : (*offset)[i - 1] + graph[i - 1].size();
        for (int j = 0; j < graph[i].size(); j++)
        {
            (*column)[(*offset)[i] + j] = graph[i][j].first;
            (*capacities)[(*offset)[i] + j] = graph[i][j].second;
            (*forwardFlow)[(*offset)[i] + j] = graph[i][j].second;
        }
    }
    (*offset)[V] = numEdges;

    source = 0;
    sink = V - 1;
    
    file.close();

    return 0;
}

int readBCSRGraphFromDIMACSFile(std::string filename, int &V, int &E, int &source, int &sink, int **offset, int **column, int **capacities, int **forwardFlow){
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
    
    std::vector<std::vector<std::pair<int, int>>> graph;

    int numEdges = 0;

    while(std::getline(file, line)){
        lineno++;

        if (line.empty() || line[0] == 'c') continue;  // Salta commenti o righe vuote

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

        if(linewords[0] == "p" && linewords.size() >= 4){
            V = std::stoi(linewords[2]);
            E = std::stoi(linewords[3]);
            if (debug)std::cout << "V: " << V <<std::endl;
            if (debug)std::cout << "E: " << E <<std::endl;
            graph.resize(V);
        }else if(linewords[0] == "n" && linewords.size() > 2){
            if(linewords[2] == "s"){
                source = std::stoi(linewords[1])-1;
                if (debug) std::cout << "s: " << source << std::endl;
            }else if(linewords[2] == "t"){
                sink = std::stoi(linewords[1])-1;
                if (debug) std::cout << "t: " << sink << std::endl;
            }
        }else if(linewords[0] == "a" && linewords.size() > 3){
            int u = std::stoi(linewords[1])-1;
            int v = std::stoi(linewords[2])-1;
            int c = std::stoi(linewords[3]);

            if (debug) std::cout << "a: " << u << " " << v << " " << c << std::endl;

            int insertIndex = -1;
            if(graph[u].empty()){
                graph[u].push_back(std::make_pair(v, c));
                numEdges++;
            }else if(graph[u].back().first < v){
                graph[u].push_back(std::make_pair(v, c));
                numEdges++;
            }else{
                for (int i = 0; i < graph[u].size(); i++){   
                    if(graph[u][i].first == v){
                        graph[u][i].second = c;
                        break;
                    }
                    if(graph[u][i].first > v){
                        insertIndex = i;
                        graph[u].insert(graph[u].begin() + insertIndex, std::make_pair(v, c));
                        numEdges++;
                        break;
                    }
                }
            }
            
            insertIndex = -1;
            if(graph[v].empty()){
                graph[v].push_back(std::make_pair(u, 0));
                numEdges++;
            }else if(graph[v].back().first < u){
                graph[v].push_back(std::make_pair(u, 0));
                numEdges++;
            }else{
                for (int i = 0; i < graph[v].size(); i++){   
                    if(graph[v][i].first == u){
                        break;
                    }
                    if(graph[v][i].first > u){
                        insertIndex = i;
                        graph[v].insert(graph[v].begin() + insertIndex, std::make_pair(u, 0)); // Aggiunta arco inverso
                        numEdges++;
                        break;
                    }
                }
            }
        }
        else{
            if(!linewords.empty()) std::cout << "Error in line " << lineno << std::endl;
        }

        linewords.clear();
    }

    E = numEdges;

    (*offset) = (int *)malloc((V + 1) * sizeof(int));
    (*column) = (int *)malloc(E * sizeof(int));
    (*capacities) = (int *)malloc(E * sizeof(int));
    (*forwardFlow) = (int *)malloc(E * sizeof(int));

        for (int i = 0; i < V; i++){
        (*offset)[i] = i == 0 ? 0 : (*offset)[i - 1] + graph[i - 1].size();
        for (int j = 0; j < graph[i].size(); j++)
        {
            (*column)[(*offset)[i] + j] = graph[i][j].first;
            (*capacities)[(*offset)[i] + j] = graph[i][j].second;
            (*forwardFlow)[(*offset)[i] + j] = graph[i][j].second;
        }
    }
    (*offset)[V] = numEdges;

    //source = 0;
    //sink = V - 1;
    
    file.close();

    return 0;
}

int readGraphFromFileOLD(std::string filename, int &n, int **capacity){
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