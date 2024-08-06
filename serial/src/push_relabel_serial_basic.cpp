#include "../include/push_relabel_serial_basic.hpp"

using namespace std;

const int inf = 1000000000;

int debug = 0;

int n, s, t;
vector<vector<int>> capacity, flow;
vector<int> height, excess, seen;
queue<int> active_nodes;

void push(int u, int v)
{
    int d = min(excess[u], capacity[u][v] - flow[u][v]);    // Calcolo il flusso da spingere
    
    // Aggiorno il flusso
    flow[u][v] += d;    
    flow[v][u] -= d;
    
    // Aggiorno le eccedenze
    excess[u] -= d;
    excess[v] += d;

    // Aggiungo il nodo v alla lista dei nodi attivi
    if (d && excess[v] == d)
        active_nodes.push(v);
}

void relabel(int u)
{
    int d = inf;
    
    // Cerco il minimo tra le altezze dei nodi adiacenti
    for (int i = 0; i < n; i++) {
        if (capacity[u][i] - flow[u][i] > 0)
            d = min(d, height[i]);
    }
    
    // Aggiorno l'altezza del nodo
    if (d < inf)
        height[u] = d + 1;
}

void pushRelabel(int u)
{
    if (u != s && u != t) {

        // Finché l'eccedenza del nodo u è maggiore di 0, provo push o relabel
        while (excess[u] > 0) {

            // Se il nodo u non ha visto tutti i nodi adiacenti, provo push
            if (seen[u] < n) {
                int v = seen[u];    // Nodo adiacente

                // Se la capacità è maggiore di 0 e l'altezza del nodo u è maggiore di quella del nodo v, eseuguo push
                // Altrimenti incremento il contatore dei nodi visti (passo al nodo successivo)
                if (capacity[u][v] - flow[u][v] > 0 && height[u] > height[v])
                    push(u, v);
                else 
                    seen[u]++;
            } 
            else {

                // Se non sono riuscito a fare push, eseguo relabel
                relabel(u);
                seen[u] = 0;    // Resetto il contatore dei nodi visti
            }
        }
    }

    // Eseguo push dopo relabel su tutti i nodi adiacenti con altezza inferiore
    for (int i = 0; i < n; i++) {
        if (capacity[u][i] - flow[u][i] > 0 && height[u] == height[i] + 1)
            push(u, i);
    }
}


int executePushRelabel(string filename){

    // Inizializzazione grafo
    readGraphFromFile(filename, &n, &capacity);
        
    s = 0; // Sorgente
    t = n-1; // Destinazione

    // Inizializzazione flusso archi
    flow.assign(n, vector<int>(n, 0)); // Inizializzo la matrice dei flussi a 0

    // Inizializzazione altezze
    height.assign(n, 0);
    height[s] = n;

    // Inizializzazione eccedenze
    excess.assign(n, 0);
    excess[s] = inf;

    // Inizializzazione nodi visti
    seen.assign(n, 0);

    if(debug) cout << "Inizializzazione grafo completata" << endl;

    // Inizializzazione flusso
    for (int i = 0; i < n; i++) {
        if (i != s)
            push(s, i);
    }

    if(debug) cout << "Inizializzazione flusso completata" << endl;

    // Calcolo flusso massimo
    while(!active_nodes.empty()){
        int u = active_nodes.front();
        active_nodes.pop();
        pushRelabel(u);
    }

    if(debug) cout << "Calcolo flusso massimo completato" << endl;

    // Restituzione flusso massimo
    if(debug) cout << "Flusso massimo: " << excess[t] << endl;

    return excess[t];
}