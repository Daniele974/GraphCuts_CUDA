#include "include/push_relabel_parallel.cuh"

int main(int argc, char **argv){
    if (argc < 3)
    {
        std::cout << "Utilizzo: ./prparallel <input_file> <output_file> <mincut flag 0/1 facoltativo default=1>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    std::string output = argv[2];
    bool computeMinCut = true;
    if (argc == 4){
        if(std::atoi(argv[3]) == 0){
            computeMinCut = false;
        }
    }

    std::cout<<"Parallelo - Eseguendo file: " <<filename <<"...";
    std::cout.flush(); //per "forzare" la stampa su terminale
    int maxflow = executePushRelabel(filename, output, computeMinCut);
    std::cout<<"Finito!"<<std::endl;
    return 0;
}