std::vector<int> findMinCutSetFromSink(int V, int sink, int *offset, int *column, int *forwardFlow){
    std::vector<int> minCutSet;
    std::queue<int> q;
    std::vector<bool> visited(V, false);

    // BFS per trovare il taglio minimo a partire dal nodo sink
    minCutSet.push_back(sink);
    q.push(sink);
    visited[sink] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        // Scansione dei vicini di u che hanno flusso verso u
        for (int v = 0; v < V; v++) {
            for (int i = offset[v]; i < offset[v+1]; i++) {
                if(column[i] == u && forwardFlow[i] > 0 && !visited[v]) {
                    minCutSet.push_back(v);
                    q.push(v);
                    visited[v] = true;
                }
            }    
        }
    }

    return minCutSet;
}



std::vector<int> findMinCutSetFromSinkOMP(int V, int sink, int *offset, int *column, int *forwardFlow){
    std::vector<int> minCutSet;
    std::queue<int> q;
    std::vector<bool> visited(V, false);

    // BFS per trovare il taglio minimo a partire dal nodo sink
    minCutSet.push_back(sink);
    q.push(sink);
    visited[sink] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        // Scansione dei vicini di u che hanno flusso verso u
        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            for (int i = offset[v]; i < offset[v+1]; i++) {
                if(column[i] == u && forwardFlow[i] > 0 && !visited[v]) {
                    #pragma omp critical
                    {
                        minCutSet.push_back(v);
                        q.push(v);
                        visited[v] = true;
                    }
                }
            }    
        }
    }

    return minCutSet;
}

std::vector<int> findMinCutSetFromSinkOMPV2(int V, int sink, int *offset, int *column, int *forwardFlow){
    std::vector<int> minCutSet;
    std::queue<int> q;
    std::vector<bool> visited(V, false);

    // BFS per trovare il taglio minimo a partire dal nodo sink
    minCutSet.push_back(sink);
    q.push(sink);
    visited[sink] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        // Scansione dei vicini di u che hanno flusso verso u
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int v = 0; v < V; v++) {
                for (int i = offset[v]; i < offset[v+1]; i++) {
                    if(column[i] == u && forwardFlow[i] > 0 && !visited[v]) {
                        #pragma omp critical
                        {
                            minCutSet.push_back(v);
                            q.push(v);
                            visited[v] = true;
                        }
                    }
                }    
            }
        }
    }

    return minCutSet;
}

std::vector<int> findMinCutSetFromSinkOMPV3(int V, int sink, int *offset, int *column, int *forwardFlow){
    std::vector<int> minCutSet;
    std::queue<int> q;
    std::vector<bool> visited(V, false);

    // BFS per trovare il taglio minimo a partire dal nodo sink
    minCutSet.push_back(sink);
    q.push(sink);
    visited[sink] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        // Scansione dei vicini di u che hanno flusso verso u
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 100)
            for (int v = 0; v < V; v++) {
                for (int i = offset[v]; i < offset[v+1]; i++) {
                    if(column[i] == u && forwardFlow[i] > 0 && !visited[v]) {
                        #pragma omp critical
                        {
                            minCutSet.push_back(v);
                            q.push(v);
                            visited[v] = true;
                        }
                    }
                }    
            }
        }
    }

    return minCutSet;
}


//--------------------------------------

    if(computeMinCut){
        std::chrono::high_resolution_clock::time_point startS = std::chrono::high_resolution_clock::now();
        minCutOMPV1 = findMinCutSetFromSinkOMP(V, sink, offset, column, forwardFlow);
        std::chrono::high_resolution_clock::time_point endS = std::chrono::high_resolution_clock::now();
        double executionTimeS = std::chrono::duration_cast<std::chrono::microseconds>(endS - startS).count()/1000.0;
        std::cout << "Min cut time (OMP): "<< executionTimeS << " ms" << std::endl;
    }

    //TODO: da rimuovere
    std::this_thread::sleep_for(std::chrono::seconds(3));

    if(computeMinCut){
        std::chrono::high_resolution_clock::time_point startS = std::chrono::high_resolution_clock::now();
        minCutOMPV2 = findMinCutSetFromSinkOMPV2(V, sink, offset, column, forwardFlow);
        std::chrono::high_resolution_clock::time_point endS = std::chrono::high_resolution_clock::now();
        double executionTimeS = std::chrono::duration_cast<std::chrono::microseconds>(endS - startS).count()/1000.0;
        std::cout << "Min cut time (OMP V2): "<< executionTimeS << " ms" << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::seconds(3));

    if(computeMinCut){
        std::chrono::high_resolution_clock::time_point startS = std::chrono::high_resolution_clock::now();
        minCutOMPV3 = findMinCutSetFromSinkOMPV3(V, sink, offset, column, forwardFlow);
        std::chrono::high_resolution_clock::time_point endS = std::chrono::high_resolution_clock::now();
        double executionTimeS = std::chrono::duration_cast<std::chrono::microseconds>(endS - startS).count()/1000.0;
        std::cout << "Min cut time (OMP V3): "<< executionTimeS << " ms" << std::endl;
    }

    // compare all the mincuts finding the differences
    if(computeMinCut){
        if(minCut.size() != minCutOMPV1.size() || minCut.size() != minCutOMPV2.size() || minCut.size() != minCutOMPV3.size()){
            std::cerr << "Error: min cut sets have different sizes" << std::endl;
            std::cout<<minCut.size()<<" "<<minCutOMPV1.size()<<" "<<minCutOMPV2.size()<<" "<<minCutOMPV3.size()<<std::endl;
            return 1;
        }

        //sort the mincuts
        std::sort(minCut.begin(), minCut.end());
        std::sort(minCutOMPV1.begin(), minCutOMPV1.end());
        std::sort(minCutOMPV2.begin(), minCutOMPV2.end());
        std::sort(minCutOMPV3.begin(), minCutOMPV3.end());

        for(int i = 0; i < minCut.size(); i++){
            if(minCut[i] != minCutOMPV1[i] || minCut[i] != minCutOMPV2[i] || minCut[i] != minCutOMPV3[i]){
                std::cerr << "Error: min cut sets are different" << std::endl;
                return 1;
            }
        }
    }