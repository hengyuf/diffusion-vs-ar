#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct Solver {
    std::array<int, 81> grid{};
    std::array<int, 9> rows{}, cols{}, boxes{};

    bool load(const std::string& puzzle) {
        if (puzzle.size() != 81) return false;
        for (int i = 0; i < 81; ++i) {
            char ch = puzzle[i];
            int value = (ch >= '1' && ch <= '9') ? ch - '0' : 0;
            grid[i] = value;
            if (!value) continue;
            int bit = 1 << (value - 1), r = i / 9, c = i % 9;
            int b = (r / 3) * 3 + c / 3;
            if ((rows[r] | cols[c] | boxes[b]) & bit) return false;
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit;
        }
        return true;
    }

    bool solve() {
        int best = -1, best_mask = 0, best_count = 10;
        for (int i = 0; i < 81; ++i) {
            if (grid[i]) continue;
            int r = i / 9, c = i % 9, b = (r / 3) * 3 + c / 3;
            int mask = 0x1ff & ~(rows[r] | cols[c] | boxes[b]);
            int count = __builtin_popcount(static_cast<unsigned>(mask));
            if (!count) return false;
            if (count < best_count) {
                best = i; best_mask = mask; best_count = count;
                if (count == 1) break;
            }
        }
        if (best < 0) return true;
        int r = best / 9, c = best % 9, b = (r / 3) * 3 + c / 3;
        while (best_mask) {
            int bit = best_mask & -best_mask;
            best_mask -= bit;
            grid[best] = __builtin_ctz(static_cast<unsigned>(bit)) + 1;
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit;
            if (solve()) return true;
            rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit;
            grid[best] = 0;
        }
        return false;
    }

    std::string answer() const {
        std::string out;
        out.reserve(81);
        for (int value : grid) out.push_back(static_cast<char>('0' + value));
        return out;
    }
};

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: solve_sudoku_exchange INPUT OUTPUT SPLIT BUCKET\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) return 2;
    output << "quizzes,solutions,source,dataset,official_rating,rating_type,"
              "difficulty_bucket,clues,split\n";
    std::string line, hash, puzzle, rating;
    long count = 0;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        if (!(fields >> hash >> puzzle >> rating)) continue;
        Solver solver;
        if (!solver.load(puzzle) || !solver.solve()) {
            std::cerr << "could not solve record " << (count + 1) << '\n';
            return 1;
        }
        int clues = 0;
        for (char ch : puzzle) clues += ch != '0' && ch != '.';
        for (char& ch : puzzle) if (ch == '.') ch = '0';
        output << puzzle << ',' << solver.answer()
               << ",sudoku_exchange,sudoku_exchange," << rating
               << ",sukaku_explainer," << argv[4] << ',' << clues << ','
               << argv[3] << '\n';
        ++count;
    }
    std::cerr << "converted " << count << " records\n";
}
