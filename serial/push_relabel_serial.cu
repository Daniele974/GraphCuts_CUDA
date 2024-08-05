#include <iostream>
#include <vector>
#include <queue>

using namespace std;

const int inf = 1000000000;

int debug = 0;

int n, s, t;
vector<vector<int>> capacity, flow;
vector<int> height, excess;

void push(int u, int v)
{
    int d = min(excess[u], capacity[u][v] - flow[u][v]);
    flow[u][v] += d;
    flow[v][u] -= d;
    excess[u] -= d;
    excess[v] += d;
}

void relabel(int u)
{
    int d = inf;
    for (int i = 0; i < n; i++) {
        if (capacity[u][i] - flow[u][i] > 0)
            d = min(d, height[i]);
    }
    if (d < inf)
        height[u] = d + 1;
}

vector<int> find_max_height_vertices(int s, int t) {
    vector<int> max_height;
    for (int i = 0; i < n; i++) {
        if (i != s && i != t && excess[i] > 0) {
            if (!max_height.empty() && height[i] > height[max_height[0]])
                max_height.clear();
            if (max_height.empty() || height[i] == height[max_height[0]])
                max_height.push_back(i);
        }
    }
    return max_height;
}


int main(int argc, char *argv[]){

    // Inizializzazione grafo
    n = 4; // Numero di nodi
    s = 0; // Sorgente
    t = n-1; // Destinazione

    // Inizializzazione capacità archi
    capacity.assign(n, vector<int>(n, 0)); // Inizializzo la matrice delle capacità a 0

    // Inizializzazione flusso archi
    flow.assign(n, vector<int>(n, 0)); // Inizializzo la matrice dei flussi a 0

    // Settaggio capacità archi
    capacity[s][1] = 2;
    capacity[s][2] = 4;
    capacity[1][2] = 3;
    capacity[1][t] = 1;
    capacity[2][t] = 5;

    // Inizializzazione altezze
    height.assign(n, 0);
    height[s] = n;

    // Inizializzazione eccedenze
    excess.assign(n, 0);
    excess[s] = inf;

    if(debug) cout << "Inizializzazione grafo completata" << endl;

    // Inizializzazione flusso
    for (int i = 0; i < n; i++) {
        if (i != s)
            push(s, i);
    }

    if(debug) cout << "Inizializzazione flusso completata" << endl;

    // Calcolo flusso massimo
    vector<int> active_nodes;
    while (!(active_nodes = find_max_height_vertices(s, t)).empty()) {
        for (int i : active_nodes) {
            bool pushed = false;
            for (int j = 0; j < n && excess[i]; j++) {
                if (capacity[i][j] - flow[i][j] > 0 && height[i] == height[j] + 1) {
                    push(i, j);
                    pushed = true;
                }
            }
            if (!pushed) {
                relabel(i);
                break;
            }
        }
    }

    if(debug) cout << "Calcolo flusso massimo completato" << endl;

    // Restituzione flusso massimo
    cout << "Flusso massimo: " << excess[t] << endl;

    return 0;
}