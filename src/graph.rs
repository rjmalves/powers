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
    EdgeAlreadyExists,
    EdgeFormsSelfLoop,
}

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

    pub fn add_node(&mut self, data: T) -> usize {
        let id = self.node_count();
        self.nodes.push(Node::new(id, data));
        self.adjacency_list.push(vec![]);
        self.reverse_adjacency_list.push(vec![]);
        id
    }

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
        if source_id == target_id {
            return Err(GraphBuildingError::EdgeFormsSelfLoop);
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
        graph.add_node(10.0);
        graph.add_node(20.0);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_add_edge_to_directed_graph() {
        let mut graph = DirectedGraph::<f64>::new();
        let id1 = graph.add_node(10.0);
        let id2 = graph.add_node(20.0);
        let edge_add_status = graph.add_edge(id1, id2);
        assert!(edge_add_status.is_ok())
    }
}
