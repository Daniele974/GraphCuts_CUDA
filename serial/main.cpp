#include "include/file_manager.hpp"
#include "include/push_relabel_serial_basic.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Utilizzo: ./prserial <input_file> <output_file>" << endl;
        return 1;
    }
    
    string filename = argv[1];
    string output = argv[2];
    int maxflow = executePushRelabel(filename, output);
    cout<<"Max flow: " <<maxflow <<endl;
    return 0;
}