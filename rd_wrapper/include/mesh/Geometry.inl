#include "Geometry.hpp"

inline Geometry::Geometry() 
{}

inline std::vector<Geometry> Geometry::FromObj(const std::string &filename)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string error = "";
	std::vector<Geometry> geometry;
	std::vector<bool> reconstruct;

	bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &error, filename.c_str());
	if (!result) {
		throw std::runtime_error("[Mesh] " + error);
	}

	reconstruct.resize(shapes.size(), false);
	geometry.resize(shapes.size());
	/* Check whether vertices should be duplicated
	 * i.e., each face has different vertex/normal/uv indices
	 */
	for (int meshid = 0; meshid != shapes.size(); ++meshid) {
		const tinyobj::shape_t &shape = shapes[meshid];
		const tinyobj::mesh_t &mesh = shape.mesh;

		for (int i = 0; i != mesh.indices.size(); ++i) {
			if ((mesh.indices[i].normal_index != -1 && 
				mesh.indices[i].vertex_index != mesh.indices[i].normal_index) ||
				(mesh.indices[i].texcoord_index != -1 &&
				mesh.indices[i].vertex_index != mesh.indices[i].texcoord_index)) 
			{
				reconstruct[meshid] = true;
			}
		}
	}

	/* Construct geometry */
	for (int meshid = 0; meshid != shapes.size(); ++meshid) {
		Geometry g;
		
		if (!reconstruct[meshid]) {
			const tinyobj::mesh_t &mesh = shapes[meshid].mesh;

			for (int i = 0; i != mesh.indices.size(); ++i) {
				g.mIndices.push_back(mesh.indices[i].vertex_index);
			}

			// copy vertices, normals and texcoord 
			if (!attrib.vertices.empty()) {
				g.mVertices = std::move(attrib.vertices);
			}
			if (!attrib.normals.empty()) {
				g.mNormals = std::move(attrib.normals);
			}
			if (!attrib.texcoords.empty()) {
				g.mTexcoords = std::move(attrib.texcoords);
			}

			// Recommend glDrawElement.
			g.mDrawOption = DrawOption::Element;
		}
		else {
			// reconstruct vertices, normals and texcoords
			g.mVertices.reserve(attrib.vertices.size() * 6);
			g.mNormals.reserve(attrib.normals.size() * 6);
			g.mTexcoords.reserve(attrib.texcoords.size() * 6);

			const tinyobj::mesh_t &mesh = shapes[meshid].mesh;

			for (int i = 0; i != mesh.indices.size(); ++i) {
				int vertexIndex = mesh.indices[i].vertex_index;
				int normalIndex = mesh.indices[i].normal_index;
				int texcoordIndex = mesh.indices[i].texcoord_index;

				if (vertexIndex != -1) {
					g.mVertices.push_back(attrib.vertices[3 * (vertexIndex)]);
					g.mVertices.push_back(attrib.vertices[3 * (vertexIndex)+1]);
					g.mVertices.push_back(attrib.vertices[3 * (vertexIndex)+2]);
				}
				if (normalIndex != -1) {
					g.mNormals.push_back(attrib.normals[3 * (normalIndex)]);
					g.mNormals.push_back(attrib.normals[3 * (normalIndex)+1]);
					g.mNormals.push_back(attrib.normals[3 * (normalIndex)+2]);
				}
				if (texcoordIndex != -1) {
					g.mTexcoords.push_back(attrib.texcoords[2 * (texcoordIndex)]);
					g.mTexcoords.push_back(attrib.texcoords[2 * (texcoordIndex)+1]);
				}
			}

			// Recommend glDrawArray; Hence no indices are needed.
			g.mDrawOption = DrawOption::Array;
		}

		geometry[meshid] = std::move(g);
	}

	return geometry;
}