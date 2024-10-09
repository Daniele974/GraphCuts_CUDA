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

// VERSIONE MIGLIORE
std::vector<int> findMinCutSetFromSinkOMPV2(int n, int t, int *residual){
    std::vector<int> minCutSet;
    std::vector<int> vertexList;
    std::vector<bool> visited(n, false);

    minCutSet.push_back(t);
    vertexList.push_back(t);
    visited[t] = true;
    while (!vertexList.empty()) {
        std::vector<int> newVertexList;
        
        #pragma omp parallel
        {
            std::vector<int> localVertexList;
            std::vector<int> localMinCutSet;
            #pragma omp for nowait schedule(dynamic)
            for (int i = 0; i < vertexList.size(); i++) {
                int u = vertexList[i];
                for (int v = 0; v < n; v++) {
                    if (!visited[v] && residual[v*n + u] > 0) {
                        #pragma omp critical
                        {
                            localMinCutSet.push_back(v);
                            localVertexList.push_back(v);                            
                            visited[v] = true;
                        }
                    }
                }
            }

            #pragma omp critical
            {
                minCutSet.insert(minCutSet.end(), localMinCutSet.begin(), localMinCutSet.end());
                newVertexList.insert(newVertexList.end(), localVertexList.begin(), localVertexList.end());
            }
        }
        vertexList = newVertexList;
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


// ---------------------------------------------------------------
// MINCUT 09/10/2024
// ---------------------------------------------------------------

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

// VERSIONE MIGLIORE
std::vector<int> findMinCutSetFromSinkOMPV2(int V, int sink, int *offset, int *column, int *forwardFlow){
    std::vector<int> minCutSet;
    std::vector<int> vertexList;
    std::vector<bool> visited(V, false);

    minCutSet.push_back(sink);
    vertexList.push_back(sink);
    visited[sink] = true;
    while (!vertexList.empty()) {
        std::vector<int> newVertexList;
        
        #pragma omp parallel
        {
            std::vector<int> localVertexList;
            std::vector<int> localMinCutSet;
            #pragma omp for nowait schedule(dynamic)
            for (int i = 0; i < vertexList.size(); i++) {
                int u = vertexList[i];
                for (int v = 0; v < V; v++) {
                    for (int j = offset[v]; j < offset[v+1]; j++) {
                        if (column[j] == u && !visited[v] && forwardFlow[j] > 0) {
                            #pragma omp critical (localPushBack)
                            {
                                localMinCutSet.push_back(v);
                                localVertexList.push_back(v);                            
                                visited[v] = true;
                            }
                        }
                    }
                }
            }

            #pragma omp critical (insert)
            {
                minCutSet.insert(minCutSet.end(), localMinCutSet.begin(), localMinCutSet.end());
                newVertexList.insert(newVertexList.end(), localVertexList.begin(), localVertexList.end());
            }
        }
        vertexList = newVertexList;
    }

    return minCutSet;
}

std::vector<int> findMinCutSetFromSinkOMPV3(int V, int E, int sink, int *offset, int *column, int *forwardFlow){

    int *t_offset = (int*)malloc((V+1)*sizeof(int));
    int *t_column = (int*)malloc(E*sizeof(int));
    int *t_forwardFlow = (int*)malloc(E*sizeof(int));

    for(int i = 0; i < V+1; i++){
        t_offset[i] = 0;
    }

    for(int i = 0; i < E; i++){
        t_column[i]= 0;
        t_forwardFlow[i] = 0;
    }

    computeTranspose(V, E, offset, column, forwardFlow, t_offset, t_column, t_forwardFlow);

    std::vector<int> minCutSet;
    std::vector<int> vertexList;
    bool *visited = (bool*)malloc(V*sizeof(bool));
    for(int i = 0; i < V; i++) {
        visited[i] = false;
    }

    minCutSet.push_back(sink);
    vertexList.push_back(sink);
    visited[sink] = true;
    while (!vertexList.empty()) {
        std::vector<int> newVertexList;
        
        #pragma omp parallel
        {
            std::vector<int> localVertexList;
            std::vector<int> localMinCutSet;
            
            #pragma omp for nowait schedule(dynamic)
            for (int i = 0; i < vertexList.size(); i++) {
                
                int u = vertexList[i];
                
                for (int j = t_offset[u]; j < t_offset[u+1]; j++) {
                    
                    int v = t_column[j];
                    bool shouldAdd = false;

                    #pragma omp critical (checkVisited)
                    {
                        if(!visited[v] && t_forwardFlow[j] > 0) {
                            visited[v] = true;
                            shouldAdd = true;
                        }
                    }
                    if(shouldAdd) {
                        localMinCutSet.push_back(v);
                        localVertexList.push_back(v);
                    }
                }
            }

            #pragma omp critical (insert)
            {
                minCutSet.insert(minCutSet.end(), localMinCutSet.begin(), localMinCutSet.end());
                newVertexList.insert(newVertexList.end(), localVertexList.begin(), localVertexList.end());
            }
        }
        vertexList = newVertexList;
    }

    return minCutSet;
}


// ---------------------------------------------------------------
// CALCOLO TRASPOSTA
// ---------------------------------------------------------------
    std::chrono::high_resolution_clock::time_point startS = std::chrono::high_resolution_clock::now();
    computeTranspose(V, E, offset, column, forwardFlow, t_offset, t_column, t_forwardFlow);
    std::chrono::high_resolution_clock::time_point endS = std::chrono::high_resolution_clock::now();
    double executionTimeS = std::chrono::duration_cast<std::chrono::microseconds>(endS - startS).count()/1000.0;
    std::cout << "Transpose OMP: "<< executionTimeS << " ms" << std::endl;    

    void computeTranspose(int V, int E, int *offset, int *column, int *forwardFlow, int *t_offset, int *t_column, int *t_forwardFlow) {

    for(int i = 0; i < E; i++) {
        t_offset[column[i]+2]++;
    }
    
    for(int i = 2; i < V+1; i++) {
        t_offset[i] += t_offset[i-1];
    }
    
    for(int i = 0; i < V; i++) {
        for(int j = offset[i]; j < offset[i+1]; j++) {
            int v = column[j];
            int newIndex = t_offset[v+1]++;
            t_column[newIndex] = i;
            t_forwardFlow[newIndex] = forwardFlow[j];
        }
    }
}

void computeTransposeOMP(int V, int E, int *offset, int *column, int *forwardFlow, int *t_offset, int *t_column, int *t_forwardFlow) {
    #pragma omp parallel for
    for(int i = 0; i < E; i++) {
        #pragma omp atomic update
        t_offset[column[i]+2]++;
    }
    
    for(int i = 2; i < V+1; i++) {
        t_offset[i] += t_offset[i-1];
    }
    
    #pragma omp parallel for
    for(int i = 0; i < V; i++) {
        for(int j = offset[i]; j < offset[i+1]; j++) {
            int v = column[j];

            int newIndex = 0;
            #pragma omp atomic capture
            newIndex = t_offset[v+1]++;
            
            t_column[newIndex] = i;
            t_forwardFlow[newIndex] = forwardFlow[j];
        }
    }
}