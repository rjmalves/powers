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
    pub fn add_node(
        &mut self,
        id: usize,
        data: T,
    ) -> Result<(), GraphBuildingError> {
        if self.get_node(id).is_some() {
            return Err(GraphBuildingError::NodeAlreadyExists);
        }
        self.nodes.push(Node::new(id, data));
        self.adjacency_list.push(vec![]);
        self.reverse_adjacency_list.push(vec![]);
        Ok(())
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
        graph.add_node(0, 10.0).unwrap();
        graph.add_node(1, 20.0).unwrap();
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_add_edge_to_directed_graph() {
        let mut graph = DirectedGraph::<f64>::new();
        graph.add_node(0, 10.0).unwrap();
        graph.add_node(1, 20.0).unwrap();
        let edge_add_status = graph.add_edge(0, 1);
        assert!(edge_add_status.is_ok())
    }
}
