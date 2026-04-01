// This file is covered by the LICENSE file in the root of this project.
// contact: Neng Wang, <neng.wang@hotmail.com>

#include "SemGraph.hpp" 


namespace graph_slam{



/*
    Building semantic graph from clustered bounding boxes
*/
Graph BuildGraph(const std::vector<Bbox> &cluster_boxes, double edge_dis_th,int subinterval,int graph_node_dimension,double subgraph_edge_th){
    // A Graph is a compact semantic representation of the current frame.
    // Nodes represent clustered objects (vehicle/trunk/pole-like), and edges
    // connect nearby objects using Euclidean distance.
    Graph frame_graph;

    // Distance bins for local descriptor histograms.
    // Example: edge_dis_th=60 and subinterval=30 => each bin is 2 meters wide.
    int sub_interval_value = int(edge_dis_th/subinterval);

    // Number of detected object clusters in this frame.
    size_t N = cluster_boxes.size();
    // TODO(M1-FPGA): Baseline this function with benchmarks/benchmark_buildgraph.cpp and track
    // avg/p95 latency vs node count for KITTI and MulRAN scenes.
    // float subgraph_edge_th = 20; // 20m for sub triangle edge threshold
    // AdjacencyMatrix(i,j)=1 means node i and node j are connected (within edge_dis_th).
    Eigen::MatrixXd AdjacencyMatrix = Eigen::MatrixXd::Zero(N,N);
    // EdgeMatrix(i,j) stores metric distance between connected nodes.
    Eigen::MatrixXd EdgeMatrix = Eigen::MatrixXd::Zero(N,N);

    // Final per-node descriptor: [local radial histogram | global spectral embedding].
    // local dimension = 3 semantic classes * subinterval radial bins.
    // global dimension = graph_node_dimension from eigendecomposition.
    Eigen::MatrixXd NodeEmbeddings = Eigen::MatrixXd::Zero(N,subinterval*3+graph_node_dimension);
    Eigen::MatrixXd NodeEmbeddings_Local = Eigen::MatrixXd::Zero(N,subinterval*3);

    for(size_t i = 0; i < N; i++){
        // Copy basic object attributes into graph node arrays.
        frame_graph.node_labels.emplace_back(cluster_boxes[i].label);  // node labels
        frame_graph.node_centers.emplace_back(cluster_boxes[i].center); // node centers
        frame_graph.node_dimensions.emplace_back(cluster_boxes[i].dimension); // node bounding box dimensions
        frame_graph.points_num.emplace_back(cluster_boxes[i].points_num); // node points number
        

        // Track class counts to support downstream semantic checks/heuristics.
        if(cluster_boxes[i].label==1) frame_graph.vehicle_num = frame_graph.vehicle_num+1;
        else if(cluster_boxes[i].label==2) frame_graph.trunk_num = frame_graph.trunk_num+1;
        else if(cluster_boxes[i].label==3) frame_graph.pole_like_num = frame_graph.pole_like_num+1;

        // TODO(M1-FPGA): This O(N^2) distance loop is the primary kernel candidate.
        // Capture tripcounts and memory access pattern before translating to HLS.
        // Temporary list for node i: (neighbor_label, distance_to_neighbor).
        // It is later transformed into a class-aware radial histogram descriptor.
        std::vector<std::pair<int,double>> vertex_edges;
        for(size_t j = 0; j< N; j++){
            // Geometric relation between objects i and j in 3D.
            double edge = (cluster_boxes[i].center - cluster_boxes[j].center).norm();
            if(edge<edge_dis_th){
                vertex_edges.emplace_back(std::make_pair(cluster_boxes[j].label,edge));
                AdjacencyMatrix(i,j) = 1;
                EdgeMatrix(i,j) = edge;
                if(j>=i+1){                                          // only count undirected edge once
                    frame_graph.edges.emplace_back(std::make_pair(i,j));
                    frame_graph.edge_value.emplace_back(edge);
                    // Closer pairs get higher weight, far pairs approach zero weight.
                    frame_graph.edge_weights.emplace_back((edge_dis_th-edge)/edge_dis_th); //[0,edge_dis_th]->[1,0]
                }
            }
        }

        // Build per-node LOCAL descriptor used for node correspondence.
        // Layout:
        // [0 ... subinterval-1]                  -> label 1 (vehicle)
        // [subinterval ... 2*subinterval-1]      -> label 2 (trunk)
        // [2*subinterval ... 3*subinterval-1]    -> label 3 (pole-like)
        // Each bucket counts how many neighbors of each class fall into that radial bin.
        for(size_t m=0;m<vertex_edges.size();m++){
            if(vertex_edges[m].first == 1){ // x - vehicle
                NodeEmbeddings_Local(i,int(vertex_edges[m].second/sub_interval_value))++;
            }
            else if(vertex_edges[m].first == 2){ // x - truck
                NodeEmbeddings_Local(i,subinterval+int(vertex_edges[m].second/sub_interval_value))++;
            }
            else if(vertex_edges[m].first == 3){ // x - pole
                NodeEmbeddings_Local(i,subinterval*2+int(vertex_edges[m].second/sub_interval_value))++;
            }
        }

    }

    // Fast return for empty frame (no semantic clusters).
    if(frame_graph.node_labels.size()==0) return  frame_graph;
    // TODO(M1-FPGA): Eigen eigendecomposition is difficult to synthesize directly.
    // For accelerator draft, replace with a fixed-point approximation or offload only
    // adjacency/edge histogram construction while keeping decomposition on CPU.
    // only 0.0xms -> 0.1ms 
    // GLOBAL descriptor from graph structure:
    // take dominant eigenvectors of adjacency matrix as a topology-aware embedding.
    Eigen::MatrixXd NodeEmbeddings_Global= MatrixDecomposing(AdjacencyMatrix,graph_node_dimension);

    // Normalize local descriptor row-wise (L2 norm) to reduce scale dependence.
    NodeEmbeddings_Local = NodeEmbeddings_Local.array().colwise()/NodeEmbeddings_Local.rowwise().norm().array();

    // Normalize global descriptor row-wise (sum norm) for comparable magnitude.
    NodeEmbeddings_Global = NodeEmbeddings_Global.array().colwise()/NodeEmbeddings_Global.rowwise().sum().array();

    // Concatenate local + global blocks.
    NodeEmbeddings.leftCols(subinterval*3) = NodeEmbeddings_Local;
    NodeEmbeddings.rightCols(graph_node_dimension) = NodeEmbeddings_Global;

    for(size_t i=0;i<N;i++){
        // Export descriptor row into std::vector<float> for Graph container.
        Eigen::MatrixXd evec_sort_row = NodeEmbeddings.row(i);
        std::vector<float> node_desf(evec_sort_row.data(),evec_sort_row.data()+evec_sort_row.size());
        frame_graph.node_desc.emplace_back(node_desf);  // node descriptors
    }


    // TODO(M1-FPGA): Consider moving this triangle enumeration to a separate kernel only
    // if profiling shows non-trivial cost after accelerating edge construction.
    // Build local triangles for geometric consistency checks.
    // Intuition: triangle side-length patterns are more stable than single edges,
    // so they are useful for pruning wrong correspondences/outliers.
    for(size_t i=0;i<N;i++){
        frame_graph.node_sub_triangles.emplace_back(std::vector<Eigen::Vector3d>());
        std::vector<int> indices;
        std::vector<Eigen::Vector3d> nodeSubgraphTriangle;

        // Find neighbors close enough to be part of local triangles around node i.
        auto edge_list = EdgeMatrix.row(i);
        for(int m=0;m<edge_list.size();m++){
            if(edge_list[m]<subgraph_edge_th && edge_list[m]!=0) indices.push_back(m);
        }

        // Need at least 3 nodes total (center i + two neighbors) to form triangles.
        if(indices.size()>2) {
            for(size_t m=0; m<indices.size();m++){
                if(m==(indices.size()-1)) break;
                for(size_t n=m+1; n<indices.size();n++){
                    std::vector<float> sub_triangle(3);
                    sub_triangle[0] = (float)(frame_graph.node_centers[i]-frame_graph.node_centers[indices[m]]).norm();
                    sub_triangle[1] = (float)(frame_graph.node_centers[i]-frame_graph.node_centers[indices[n]]).norm();
                    sub_triangle[2] = (float)(frame_graph.node_centers[indices[m]]-frame_graph.node_centers[indices[n]]).norm();
                    // Sort side lengths to make descriptor invariant to point ordering.
                    std::sort(sub_triangle.begin(), sub_triangle.end()); // sort triangle edges
                    nodeSubgraphTriangle.emplace_back(Eigen::Vector3d(sub_triangle[0],sub_triangle[1],sub_triangle[2]));
                }
            }
            frame_graph.node_sub_triangles[i].assign(nodeSubgraphTriangle.begin(),nodeSubgraphTriangle.end());
        }
        
    }

    // Keep full pairwise edge matrix for downstream correspondence pruning.
    frame_graph.edge_matrix = EdgeMatrix; // for subsequent correspondences pruning

    return frame_graph;
}

/*
    Rebuilding graph for local graph map
*/
Graph ReBuildGraph(const std::vector<InsNode> &cluster_boxes, double edge_dis_th,int subinterval,int graph_node_dimension,double subgraph_edge_th){
    // Same graph construction logic as BuildGraph, but input centers come from
    // InsNode::pose (nodes already maintained in local map coordinates).
    // Keeping both functions aligned ensures consistent descriptors and matching.
    Graph frame_graph;
    int sub_interval_value = int(edge_dis_th/subinterval);
    size_t N = cluster_boxes.size();
    // TODO(M1-FPGA): Keep this function aligned with BuildGraph so software-vs-hardware
    // comparisons use equivalent logic for local-map updates.
    // float subgraph_edge_th = 20; // 20m for sub triangle edge threshold
    Eigen::MatrixXd AdjacencyMatrix = Eigen::MatrixXd::Zero(N,N);
    Eigen::MatrixXd EdgeMatrix = Eigen::MatrixXd::Zero(N,N);
    Eigen::MatrixXd NodeEmbeddings = Eigen::MatrixXd::Zero(N,subinterval*3+graph_node_dimension);
    Eigen::MatrixXd NodeEmbeddings_Local = Eigen::MatrixXd::Zero(N,subinterval*3);

    for(size_t i = 0; i < N; i++){
        frame_graph.node_labels.emplace_back(cluster_boxes[i].label);
        frame_graph.node_centers.emplace_back(cluster_boxes[i].pose);
        frame_graph.node_dimensions.emplace_back(cluster_boxes[i].dimension);
        frame_graph.points_num.emplace_back(cluster_boxes[i].points_num);
        

        if(cluster_boxes[i].label==1) frame_graph.vehicle_num = frame_graph.vehicle_num+1;
        else if(cluster_boxes[i].label==2) frame_graph.trunk_num = frame_graph.trunk_num+1;
        else if(cluster_boxes[i].label==3) frame_graph.pole_like_num = frame_graph.pole_like_num+1;

        std::vector<std::pair<int,double>> vertex_edges;
        for(size_t j = 0; j< N; j++){
            double edge = (cluster_boxes[i].pose - cluster_boxes[j].pose).norm();
            if(edge<edge_dis_th){
                vertex_edges.emplace_back(std::make_pair(cluster_boxes[j].label,edge));
                AdjacencyMatrix(i,j) = 1;
                EdgeMatrix(i,j) = edge;
                if(j>=i+1){                                          // only count once
                    frame_graph.edges.emplace_back(std::make_pair(i,j));
                    frame_graph.edge_value.emplace_back(edge);
                    frame_graph.edge_weights.emplace_back((edge_dis_th-edge)/edge_dis_th); //[0,edge_dis_th]->[1,0]
                }
            }
        }

        // build vertes desc: main for loop clousre detection
        for(size_t m=0;m<vertex_edges.size();m++){
            if(vertex_edges[m].first == 1){ // x - vehicle
                NodeEmbeddings_Local(i,int(vertex_edges[m].second/sub_interval_value))++;
            }
            else if(vertex_edges[m].first == 2){ // x - truck
                NodeEmbeddings_Local(i,subinterval+int(vertex_edges[m].second/sub_interval_value))++;
            }
            else if(vertex_edges[m].first == 3){ // x - pole
                NodeEmbeddings_Local(i,subinterval*2+int(vertex_edges[m].second/sub_interval_value))++;
            }
        }

    }

    if(frame_graph.node_labels.size()==0) return  frame_graph;
    // only 0.0xms -> 0.1ms 
    Eigen::MatrixXd NodeEmbeddings_Global= MatrixDecomposing(AdjacencyMatrix,graph_node_dimension);
    NodeEmbeddings_Local = NodeEmbeddings_Local.array().colwise()/NodeEmbeddings_Local.rowwise().norm().array();
    NodeEmbeddings_Global = NodeEmbeddings_Global.array().colwise()/NodeEmbeddings_Global.rowwise().sum().array();
    NodeEmbeddings.leftCols(subinterval*3) = NodeEmbeddings_Local;
    NodeEmbeddings.rightCols(graph_node_dimension) = NodeEmbeddings_Global;

    for(size_t i=0;i<N;i++){
        Eigen::MatrixXd evec_sort_row = NodeEmbeddings.row(i);
        std::vector<float> node_desf(evec_sort_row.data(),evec_sort_row.data()+evec_sort_row.size());
        frame_graph.node_desc.emplace_back(node_desf);
    }
    

    // build local sub triangle for outlier pruning
    for(size_t i=0;i<N;i++){
        frame_graph.node_sub_triangles.emplace_back(std::vector<Eigen::Vector3d>());
        std::vector<int> indices;
        std::vector<Eigen::Vector3d> nodeSubgraphTriangle;
        auto edge_list = EdgeMatrix.row(i);
        for(int m=0;m<edge_list.size();m++){
            if(edge_list[m]<subgraph_edge_th && edge_list[m]!=0) indices.push_back(m);
        }

        if(indices.size()>2) {
            for(size_t m=0; m<indices.size();m++){
                if(m==(indices.size()-1)) break;
                for(size_t n=m+1; n<indices.size();n++){
                    std::vector<float> sub_triangle(3);
                    sub_triangle[0] = (float)(frame_graph.node_centers[i]-frame_graph.node_centers[indices[m]]).norm();
                    sub_triangle[1] = (float)(frame_graph.node_centers[i]-frame_graph.node_centers[indices[n]]).norm();
                    sub_triangle[2] = (float)(frame_graph.node_centers[indices[m]]-frame_graph.node_centers[indices[n]]).norm();
                    std::sort(sub_triangle.begin(), sub_triangle.end()); // sort triangle edges
                    nodeSubgraphTriangle.emplace_back(Eigen::Vector3d(sub_triangle[0],sub_triangle[1],sub_triangle[2]));
                }
            }
        }
        frame_graph.node_sub_triangles[i].assign(nodeSubgraphTriangle.begin(),nodeSubgraphTriangle.end());
    }

    frame_graph.edge_matrix = EdgeMatrix; // for subsequent correspondences pruning

    return frame_graph;
}



/*
    Decomposing adjacency matrix to get node vector
*/
Eigen::MatrixXd MatrixDecomposing(Eigen::MatrixXd MatrixInput,int Dimension){ 

    // Spectral decomposition of adjacency matrix:
    // eigenvectors capture global connectivity patterns of the graph.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
        es.compute(MatrixInput);
        Eigen::MatrixXcd evecs = es.eigenvectors();            
        Eigen::MatrixXd evecs_abs = evecs.real().cwiseAbs();   // get abs eigen vector

        Eigen::MatrixXcd evals = es.eigenvalues();             
        Eigen::MatrixXd abs_evals = evals.real().cwiseAbs();

        // Sort eigenvalues descending and select corresponding top eigenvectors.
        std::vector<float> vec_evals(&abs_evals(0, 0),abs_evals.data()+abs_evals.size()); // Eigen::MatrixXf --> std::vector
        std::vector<int> indices(vec_evals.size());
        std::iota(indices.begin(), indices.end(), 0);  // sort: get d eigen vector corresponding to the d largest eigen value
        std::sort(indices.begin(), indices.end(), [&vec_evals](int i, int j) { return vec_evals[i] > vec_evals[j]; });

        // Build node embedding matrix with requested dimensionality.
        // If graph is smaller than Dimension, only available columns are filled.
        Eigen::MatrixXd evecs_sort = Eigen::MatrixXd::Zero(vec_evals.size(),Dimension);
        size_t iter = std::min(Dimension,static_cast<int>(indices.size()));
        for(size_t i=0;i<iter;i++){
            evecs_sort.col(i) = evecs_abs.col(indices[i]);
        }

        return evecs_sort;
    }

}

