
#include <functional>
#include <queue>
#include <utility>

// https://stackoverflow.com/questions/59463921/a-priority-queue-with-a-custom-comparator-in-cython

typedef struct tuple7 {
    float x1;
    int x2;
    int x3;
    int x4;
    int x5;
    int x6;
    int x7;

};

// typedef struct tuple3 {
//     int x1;
//     int x2;
//     int x3;
// };

using cpp_pq = std::priority_queue<tuple7,std::vector<tuple7>,std::function<bool(tuple7,tuple7)>>;
// using cpp_set = std::set<tuple3,std::vector<tuple3>,std::function<bool(tuple3,tuple3)>>;