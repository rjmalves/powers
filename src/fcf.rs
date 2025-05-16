use crate::cut;
use crate::state;

pub struct FutureCostFunction {
    pub cut_pool: cut::BendersCutPool,
    pub state_pool: state::VisitedStatePool,
}

impl FutureCostFunction {
    pub fn new() -> Self {
        Self {
            cut_pool: cut::BendersCutPool::new(),
            state_pool: state::VisitedStatePool::new(),
        }
    }

    pub fn add_cut(&mut self, new_cut: cut::BendersCut) {
        self.cut_pool.pool.push(new_cut);
    }

    pub fn add_state(&mut self, new_state: Box<dyn state::State>) {
        self.state_pool.pool.push(new_state);
    }

    pub fn get_total_cut_count(&self) -> usize {
        self.cut_pool.total_cut_count
    }

    pub fn update_existing_cuts_domination(
        &mut self,
        new_state: &mut Box<dyn state::State>,
    ) -> Vec<usize> {
        let mut cut_non_dominated_decrement_ids = Vec::<usize>::new();
        let mut cut_ids_to_return_to_model = Vec::<usize>::new();
        for old_cut in self.cut_pool.pool.iter_mut() {
            match old_cut.active {
                true => continue,
                false => {
                    let height =
                        old_cut.eval_height_at_state(&new_state.coefficients());
                    if height > new_state.get_dominating_objective() {
                        cut_non_dominated_decrement_ids
                            .push(new_state.get_dominating_cut_id());

                        old_cut.non_dominated_state_count += 1;
                        new_state.update_dominating_cut(old_cut, height);
                        cut_ids_to_return_to_model.push(old_cut.id);
                    }
                    continue;
                }
            }
        }
        // Decrements the non-dominating counts
        for cut_id in cut_non_dominated_decrement_ids.iter() {
            self.cut_pool.pool[*cut_id].non_dominated_state_count -= 1;
        }

        cut_ids_to_return_to_model
    }

    /// Tests the new cut on every previously visited state. If this cut dominates,
    /// decrements the previous dominating cut counter and updates this.
    pub fn eval_new_cut_domination(&mut self, new_cut: &mut cut::BendersCut) {
        for state in self.state_pool.pool.iter_mut() {
            let height = new_cut.eval_height_at_state(&state.coefficients());
            if height > state.get_dominating_objective() {
                self.cut_pool.pool[state.get_dominating_cut_id()]
                    .non_dominated_state_count -= 1;
                new_cut.non_dominated_state_count += 1;
                state.update_dominating_cut(new_cut, height);
            }
        }
    }

    /// Tests the cuts that are not in the model for the new state. If any of these cuts
    /// dominate the new state, increment their counter and puts them back inside the model
    pub fn update_old_cuts_domination(
        &mut self,
        new_state: &mut Box<dyn state::State>,
    ) -> Vec<usize> {
        let mut cut_non_dominated_decrement_ids = Vec::<usize>::new();
        let mut cut_ids_to_return_to_model = Vec::<usize>::new();
        for old_cut in self.cut_pool.pool.iter_mut() {
            match old_cut.active {
                true => continue,
                false => {
                    let height =
                        old_cut.eval_height_at_state(&new_state.coefficients());
                    if height > new_state.get_dominating_objective() {
                        cut_non_dominated_decrement_ids
                            .push(new_state.get_dominating_cut_id());

                        old_cut.non_dominated_state_count += 1;
                        new_state.update_dominating_cut(old_cut, height);
                        cut_ids_to_return_to_model.push(old_cut.id);
                    }
                    continue;
                }
            }
        }
        // Decrements the non-dominating counts
        for cut_id in cut_non_dominated_decrement_ids.iter() {
            self.cut_pool.pool[*cut_id].non_dominated_state_count -= 1;
        }

        cut_ids_to_return_to_model
    }

    pub fn update_cut_pool_on_add(&mut self, cut_id: usize) {
        self.cut_pool.active_cut_ids.push(cut_id);
        self.cut_pool.total_cut_count += 1;
    }

    pub fn update_cut_pool_on_return(&mut self, cut_id: usize) {
        self.cut_pool.active_cut_ids.push(cut_id);
        self.cut_pool.pool[cut_id].active = true;
    }

    pub fn get_active_cut_index_by_id(&self, cut_id: usize) -> usize {
        self.cut_pool
            .active_cut_ids
            .iter()
            .position(|&x| x == cut_id)
            .unwrap()
    }

    pub fn update_cut_pool_on_remove(
        &mut self,
        cut_id: usize,
        cut_index: usize,
    ) {
        self.cut_pool.active_cut_ids.remove(cut_index);
        self.cut_pool.pool[cut_id].active = false;
    }
}

pub struct CutStatePair {
    pub cut: cut::BendersCut,
    pub state: Box<dyn state::State>,
}

impl CutStatePair {
    pub fn new(cut: cut::BendersCut, state: Box<dyn state::State>) -> Self {
        Self { cut, state }
    }
}
