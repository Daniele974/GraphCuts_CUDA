#include "include/file_manager.hpp"
#include "include/push_relabel_serial_basic.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cout << "Utilizzo: ./prserial <input_file> <output_file> <mincut flag 0/1>" << endl;
        return 1;
    }
    
    string filename = argv[1];
    string output = argv[2];
    bool computeMinCut = true;
    if (argc == 4){
        if(std::atoi(argv[3]) == 0){
            computeMinCut = false;
        }
    }

    cout<<"Seriale - Eseguendo file: " <<filename <<"...";
    cout.flush(); //per "forzare" la stampa su terminale
    int maxflow = executePushRelabel(filename, output, computeMinCut);
    cout<<"Finito!"<<endl;
    return 0;
}