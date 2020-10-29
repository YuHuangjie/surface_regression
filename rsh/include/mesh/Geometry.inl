#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "Geometry.hpp"

using namespace std;

inline Geometry::Geometry() 
	: nVertexAttribs(0)
{}

static inline string NextLine(ifstream &infile)
{
	string line;

	if (!infile.eof()) {
		getline(infile, line);
		return line;
	}

	throw ios_base::failure("End of file");
}

inline std::vector<Geometry> Geometry::FromObj(const std::string &filename)
{
	vector<Geometry> geometry(1);
	Geometry &g = geometry[0];
	vector<float> &v = g.mVertices;
	vector<int32_t> &f = g.mIndices;
	ifstream infile(filename);
	string line = "";
	bool oneshot = false;

	if (!infile.is_open()) {
		cerr << "Unable to open " << filename << endl;
		throw runtime_error(string("Unable to open ") + filename);
	}

	try {
		while (true) {
			line = NextLine(infile);
			istringstream iss(line);
			string tag;
			int va = 0;
			
			iss >> tag;
			if (line.empty() || line[0] == '#')
				continue;
			if (tag == "v") {
				float f;
				while (iss.good()) {
					iss >> f;
					v.push_back(f);
					va++;
				}
				if (!oneshot) {
					oneshot = true;
					g.nVertexAttribs = va;
				}
			}
			else if (tag == "f") {
				int32_t f1, f2, f3;
				iss >> f1 >> f2 >> f3;
				f.push_back(f1-1);
				f.push_back(f2-1);
				f.push_back(f3-1);
			}
			else {
				cerr << "Obj parser: unknown tag " << tag << endl;
			}
		}
	} catch (ios_base::failure &e) {
		;
	}

	g.mDrawOption = Element;
	return geometry;
}
