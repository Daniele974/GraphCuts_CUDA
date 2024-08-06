#include "include/file_manager.hpp"
#include "include/push_relabel_serial_basic.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    string filename = "input_data/input.txt";
    int maxflow = executePushRelabel(filename);
    cout<<"Max flow: " <<maxflow <<endl;
    return 0;
}