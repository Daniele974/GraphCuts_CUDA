#include "include/push_relabel_parallel.cuh"

int main(int argc, char **argv){
    if (argc != 3)
    {
        std::cout << "Utilizzo: ./prparallel <input_file> <output_file>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    std::string output = argv[2];
    std::cout<<"Parallelo - Eseguendo file: " <<filename <<"...";
    std::cout.flush(); //per "forzare" la stampa su terminale
    int maxflow = executePushRelabel(filename, output);
    std::cout<<"Finito!"<<std::endl;
    return 0;
}