#[derive(Debug)]
pub struct Node<T> {
    pub id: usize,
    pub data: T,
}

impl<T> Node<T> {
    pub fn new(id: usize, data: T) -> Self {
        Self { id, data }
    }
}

#[derive(Debug)]
pub enum GraphBuildingError {
    NodeNotFound(usize),
    NodeAlreadyExists,
    EdgeAlreadyExists,
}

/// A simple directed graph structure for using in the SDDP algorithm, to
/// make the temporal decomposition of the problem, store the power system
/// configurations, the built policies and simulation results.
pub struct DirectedGraph<T> {
    nodes: Vec<Node<T>>,
    // adjacency_list[i] contains the IDs of nodes that node 'i' points to
    adjacency_list: Vec<Vec<usize>>,
    // reverse_adjacency_list[i] contains the IDs of nodes pointing to 'i'
    reverse_adjacency_list: Vec<Vec<usize>>,
}

impl<T> DirectedGraph<T> {
    pub fn new() -> Self {
        DirectedGraph {
            nodes: vec![],
            adjacency_list: vec![],
            reverse_adjacency_list: vec![],
        }
    }

    /// Adds a new node to the node collection. Since the graph is only
    /// built at the beginning of the algorithm, the `push` call is not
    /// expensive for the total time
    pub fn add_node(&mut self, data: T) -> Result<usize, GraphBuildingError> {
        let id = self.node_count();
        if self.get_node(id).is_some() {
            return Err(GraphBuildingError::NodeAlreadyExists);
        }
        self.nodes.push(Node::new(id, data));
        self.adjacency_list.push(vec![]);
        self.reverse_adjacency_list.push(vec![]);
        Ok(id)
    }

    /// Adds a new edge to the adjancency maps. Since the graph is only
    /// built at the beginning of the algorithm, the `push` call is not
    /// expensive for the total time
    pub fn add_edge(
        &mut self,
        source_id: usize,
        target_id: usize,
    ) -> Result<(), GraphBuildingError> {
        // validation
        if source_id > self.nodes.len() {
            return Err(GraphBuildingError::NodeNotFound(source_id));
        }
        if target_id > self.nodes.len() {
            return Err(GraphBuildingError::NodeNotFound(target_id));
        }
        if self.adjacency_list[source_id].contains(&target_id) {
            return Err(GraphBuildingError::EdgeAlreadyExists);
        }

        // adding to the topology
        self.adjacency_list[source_id].push(target_id);
        self.reverse_adjacency_list[target_id].push(source_id);
        Ok(())
    }

    pub fn get_node(&self, id: usize) -> Option<&Node<T>> {
        self.nodes.get(id)
    }

    pub fn get_node_id_with<F>(&self, f: F) -> Option<usize>
    where
        F: Fn(&T) -> bool,
    {
        self.nodes.iter().position(|node| f(&node.data))
    }

    pub fn get_all_node_ids_with<F>(&self, f: F) -> Vec<usize>
    where
        F: Fn(&T) -> bool,
    {
        self.nodes
            .iter()
            .filter(|node| f(&node.data))
            .map(|node| node.id)
            .collect()
    }

    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut Node<T>> {
        self.nodes.get_mut(id)
    }

    pub fn get_children(&self, id: usize) -> Option<&[usize]> {
        self.adjacency_list
            .get(id)
            .map(|indices| indices.as_slice())
    }

    pub fn get_parents(&self, id: usize) -> Option<&[usize]> {
        self.reverse_adjacency_list
            .get(id)
            .map(|indices| indices.as_slice())
    }

    pub fn get_bfs(&self, root_id: usize, reverse: bool) -> Vec<usize> {
        let adjacency = if reverse {
            &self.reverse_adjacency_list
        } else {
            &self.adjacency_list
        };
        let node_count = self.node_count();
        let mut visited = vec![false; node_count];
        let mut queue = vec![root_id];
        let mut bfs = Vec::<usize>::new();
        visited[root_id] = true;
        while queue.len() > 0 {
            let node = queue.pop().unwrap();
            bfs.push(node);
            for id in 0..node_count {
                if adjacency[node].contains(&id) && !visited[id] {
                    queue.push(id);
                    visited[id] = true;
                }
            }
        }
        if reverse {
            bfs.reverse()
        }
        bfs.pop();

        bfs
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_root(&self, id: usize) -> bool {
        self.reverse_adjacency_list.get(id).unwrap().len() == 0
    }

    pub fn is_leaf(&self, id: usize) -> bool {
        self.adjacency_list.get(id).unwrap().len() == 0
    }
}

impl<T> DirectedGraph<T> {
    pub fn map_topology_with_default<U: Default>(&self) -> DirectedGraph<U> {
        let num_nodes = self.node_count();
        let mut g = DirectedGraph::<U>::new();
        for _ in self.nodes.iter() {
            g.add_node(U::default()).unwrap();
        }
        for source_id in 0..num_nodes {
            if let Some(children_ids) = self.adjacency_list.get(source_id) {
                for &target_id in children_ids.iter() {
                    g.add_edge(source_id, target_id).unwrap();
                }
            }
        }
        g
    }

    pub fn map_topology_with<U, F>(&self, mut f: F) -> DirectedGraph<U>
    where
        F: FnMut(&T, usize) -> U,
    {
        let num_nodes = self.node_count();
        let mut g = DirectedGraph::<U>::new();
        for node in self.nodes.iter() {
            g.add_node(f(&node.data, node.id)).unwrap();
        }
        for source_id in 0..num_nodes {
            if let Some(children_ids) = self.adjacency_list.get(source_id) {
                for &target_id in children_ids.iter() {
                    g.add_edge(source_id, target_id).unwrap();
                }
            }
        }
        g
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_directed_graph() {
        let graph = DirectedGraph::<f64>::new();
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_add_node_to_directed_graph() {
        let mut graph = DirectedGraph::<f64>::new();
        graph.add_node(10.0).unwrap();
        graph.add_node(20.0).unwrap();
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_add_edge_to_directed_graph() {
        let mut graph = DirectedGraph::<f64>::new();
        graph.add_node(10.0).unwrap();
        graph.add_node(20.0).unwrap();
        let edge_add_status = graph.add_edge(0, 1);
        assert!(edge_add_status.is_ok())
    }

    #[test]
    fn test_map_topology_with_default() {
        let mut graph = DirectedGraph::<f64>::new();
        graph.add_node(10.0).unwrap();
        graph.add_node(20.0).unwrap();
        let edge_add_status = graph.add_edge(0, 1);
        assert!(edge_add_status.is_ok());
        let new_graph = graph.map_topology_with_default::<usize>();
        assert_eq!(new_graph.node_count(), 2);
        assert!(new_graph.get_node(0).is_some());
        assert!(new_graph.get_node(1).is_some());
        assert!(new_graph.is_root(0));
        assert!(new_graph.is_leaf(1));
    }

    #[test]
    fn test_map_topology_with() {
        let mut graph = DirectedGraph::<f64>::new();
        graph.add_node(10.0).unwrap();
        graph.add_node(20.0).unwrap();
        let edge_add_status = graph.add_edge(0, 1);
        assert!(edge_add_status.is_ok());
        let new_graph = graph.map_topology_with(|value, _id| *value as usize);
        assert_eq!(new_graph.node_count(), 2);
        assert!(new_graph.get_node(0).is_some());
        assert!(new_graph.get_node(1).is_some());
        assert!(new_graph.is_root(0));
        assert!(new_graph.is_leaf(1));
    }
}
