//! Direct bindings to HiGHS solver via `highs-sys` for zero-cost LP/MIP solving.
//!
//! Provides safe wrapper around HiGHS C API optimized for SDDP subproblem solves.
//! For design rationale and comparison with `highs` crate, see
//! [`docs/architecture/SOLVER.md`](../../docs/architecture/SOLVER.md).

use std::borrow::Borrow;
use std::cell::RefCell;
use std::convert::TryFrom;
use std::ffi::{c_void, CStr, CString};
use std::fmt::{Debug, Formatter};
use std::num::TryFromIntError;
use std::ops::{Bound, RangeBounds};
use std::os::raw::{c_char, c_int};
use std::ptr::null;

use highs_sys::*;

// Thread-local buffers for zero-allocation row operations.
// Capacity 256 covers typical cut constraints (state_dim + 1 variables).
thread_local! {
    /// Buffer for column indices in add_row operations.
    static ROW_COLS_BUFFER: RefCell<Vec<HighsInt>> = const { RefCell::new(Vec::new()) };
    /// Buffer for coefficient values in add_row operations.
    static ROW_VALS_BUFFER: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// Buffer for delete_row operations (single row deletion).
    static DELETE_ROW_BUFFER: RefCell<Vec<HighsInt>> = const { RefCell::new(Vec::new()) };
}

/// Initial capacity for row operation buffers.
const ROW_BUFFER_INITIAL_CAPACITY: usize = 256;

/// The kinds of results of an optimization
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsModelStatus {
    /// not initialized
    NotSet = MODEL_STATUS_NOTSET as isize,
    /// Unable to load model
    LoadError = MODEL_STATUS_LOAD_ERROR as isize,
    /// invalid model
    ModelError = MODEL_STATUS_MODEL_ERROR as isize,
    /// Unable to run the pre-solve phase
    PresolveError = MODEL_STATUS_PRESOLVE_ERROR as isize,
    /// Unable to solve
    SolveError = MODEL_STATUS_SOLVE_ERROR as isize,
    /// Unable to clean after solve
    PostsolveError = MODEL_STATUS_POSTSOLVE_ERROR as isize,
    /// No variables in the model: nothing to optimize
    ModelEmpty = MODEL_STATUS_MODEL_EMPTY as isize,
    /// There is no solution to the problem
    Infeasible = MODEL_STATUS_INFEASIBLE as isize,
    /// The problem in unbounded or infeasible
    UnboundedOrInfeasible = MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE as isize,
    /// The problem is unbounded: there is no single optimal value
    Unbounded = MODEL_STATUS_UNBOUNDED as isize,
    /// An optimal solution was found
    Optimal = MODEL_STATUS_OPTIMAL as isize,
    /// objective bound
    ObjectiveBound = MODEL_STATUS_OBJECTIVE_BOUND as isize,
    /// objective target
    ObjectiveTarget = MODEL_STATUS_OBJECTIVE_TARGET as isize,
    /// reached limit
    ReachedTimeLimit = MODEL_STATUS_REACHED_TIME_LIMIT as isize,
    /// reached limit
    ReachedIterationLimit = MODEL_STATUS_REACHED_ITERATION_LIMIT as isize,
    /// Unknown model status
    Unknown = MODEL_STATUS_UNKNOWN as isize,
}

/// This error should never happen: an unexpected status was returned
#[derive(PartialEq, Clone, Copy)]
pub struct InvalidStatus(pub c_int);

impl Debug for InvalidStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is not a valid HiGHS model status. \
        This error comes from a bug in highs rust bindings. \
        Please report it.",
            self.0
        )
    }
}

impl TryFrom<c_int> for HighsModelStatus {
    type Error = InvalidStatus;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        use highs_sys::*;
        match value {
            MODEL_STATUS_NOTSET => Ok(Self::NotSet),
            MODEL_STATUS_LOAD_ERROR => Ok(Self::LoadError),
            MODEL_STATUS_MODEL_ERROR => Ok(Self::ModelError),
            MODEL_STATUS_PRESOLVE_ERROR => Ok(Self::PresolveError),
            MODEL_STATUS_SOLVE_ERROR => Ok(Self::SolveError),
            MODEL_STATUS_POSTSOLVE_ERROR => Ok(Self::PostsolveError),
            MODEL_STATUS_MODEL_EMPTY => Ok(Self::ModelEmpty),
            MODEL_STATUS_INFEASIBLE => Ok(Self::Infeasible),
            MODEL_STATUS_UNBOUNDED => Ok(Self::Unbounded),
            MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => {
                Ok(Self::UnboundedOrInfeasible)
            }
            MODEL_STATUS_OPTIMAL => Ok(Self::Optimal),
            MODEL_STATUS_OBJECTIVE_BOUND => Ok(Self::ObjectiveBound),
            MODEL_STATUS_OBJECTIVE_TARGET => Ok(Self::ObjectiveTarget),
            MODEL_STATUS_REACHED_TIME_LIMIT => Ok(Self::ReachedTimeLimit),
            MODEL_STATUS_REACHED_ITERATION_LIMIT => {
                Ok(Self::ReachedIterationLimit)
            }
            MODEL_STATUS_UNKNOWN => Ok(Self::Unknown),
            n => Err(InvalidStatus(n)),
        }
    }
}

/// The status of a highs operation
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsStatus {
    /// Success
    OK = 0,
    /// Done, with warning
    Warning = 1,
    /// An error occurred
    Error = 2,
}

impl From<TryFromIntError> for HighsStatus {
    fn from(_: TryFromIntError) -> Self {
        Self::Error
    }
}

impl TryFrom<c_int> for HighsStatus {
    type Error = InvalidStatus;

    fn try_from(value: c_int) -> Result<Self, InvalidStatus> {
        match value {
            STATUS_OK => Ok(Self::OK),
            STATUS_WARNING => Ok(Self::Warning),
            STATUS_ERROR => Ok(Self::Error),
            n => Err(InvalidStatus(n)),
        }
    }
}

pub trait HighsOptionValue {
    /// Applies this value to a HiGHS option.
    ///
    /// # Safety
    ///
    /// The `highs` pointer must be a valid HiGHS model instance, and the `option`
    /// pointer must be a valid null-terminated C string representing a HiGHS option name.
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int;
}

impl HighsOptionValue for bool {
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int {
        highs_sys::Highs_setBoolOptionValue(
            highs,
            option,
            if self { 1 } else { 0 },
        )
    }
}

impl HighsOptionValue for i32 {
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int {
        highs_sys::Highs_setIntOptionValue(highs, option, self)
    }
}

impl HighsOptionValue for f64 {
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int {
        highs_sys::Highs_setDoubleOptionValue(highs, option, self)
    }
}

impl HighsOptionValue for &CStr {
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int {
        highs_sys::Highs_setStringOptionValue(highs, option, self.as_ptr())
    }
}

impl HighsOptionValue for &[u8] {
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int {
        CString::new(self)
            .expect("invalid highs option value")
            .apply_to_highs(highs, option)
    }
}

impl HighsOptionValue for &str {
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int {
        self.as_bytes().apply_to_highs(highs, option)
    }
}

fn bound_value<N: Into<f64> + Copy>(b: Bound<&N>) -> Option<f64> {
    match b {
        Bound::Included(v) | Bound::Excluded(v) => Some((*v).into()),
        Bound::Unbounded => None,
    }
}

fn c(n: usize) -> HighsInt {
    n.try_into().expect("size too large for HiGHS")
}

macro_rules! highs_call {
    ($function_name:ident ($($param:expr),+)) => {
        try_handle_status(
            $function_name($($param),+),
            stringify!($function_name)
        )
    }
}

/// An optimization problem
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Problem {
    pub num_col: usize,
    pub num_row: usize,
    pub num_nz: usize,
    pub col_cost: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    columns: Vec<(Vec<c_int>, Vec<f64>)>,
    pub offset: f64,
}

impl Problem {
    /// Create a new problem instance
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_row<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(usize, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        bounds: B,
        row_factors: I,
    ) -> usize {
        let num_rows: c_int = self.num_row.try_into().expect("too many rows");
        for r in row_factors {
            let &(col, factor) = r.borrow();
            let c = &mut self.columns[col];
            c.0.push(num_rows);
            c.1.push(factor);
            self.num_nz += 1;
        }
        let low =
            bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.row_lower.push(low);
        self.row_upper.push(high);
        let old_row_count = self.num_row;
        self.num_row += 1;
        old_row_count
    }

    pub fn add_column<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
    ) -> usize {
        self.col_cost.push(col_factor);
        let low =
            bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.col_lower.push(low);
        self.col_upper.push(high);
        self.columns.push((vec![], vec![]));
        let old_col_count = self.num_col;
        self.num_col += 1;
        old_col_count
    }

    fn to_compressed_matrix_form(&self) -> (Vec<c_int>, Vec<c_int>, Vec<f64>) {
        let mut astart = Vec::with_capacity(self.num_col);
        astart.push(0);
        let size: usize = self.num_nz;
        let mut aindex = Vec::with_capacity(size);
        let mut avalue = Vec::with_capacity(size);
        for (row_indices, factors) in self.columns.as_slice() {
            aindex.extend_from_slice(row_indices);
            avalue.extend_from_slice(factors);
            astart.push(aindex.len().try_into().expect("invalid matrix size"));
        }
        (astart, aindex, avalue)
    }

    /// Create a model based on this problem. Don't solve it yet.
    /// If the problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// If the problem is invalid (according to HiGHS), this function will panic.
    pub fn optimise(self, sense: Sense) -> Model {
        self.try_optimise(sense).expect("invalid problem")
    }

    /// Create a model based on this problem. Don't solve it yet.
    /// If the problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    pub fn try_optimise(self, sense: Sense) -> Result<Model, HighsStatus> {
        let mut m = Model::try_new(self)?;
        m.set_sense(sense);
        Ok(m)
    }

    /// Create a Model for solving without consuming the Problem.
    ///
    /// The Problem remains valid and can create additional Models or be modified.
    /// This is essential for the per-iteration Model architecture where Problem
    /// is the persistent source of truth.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Iteration 1
    /// let mut model = problem.create_model(Sense::Minimise)?;
    /// model.solve();
    /// drop(model);  // HiGHS freed
    ///
    /// // Modify problem for iteration 2
    /// problem.change_row_bounds(cut_row, rhs, f64::INFINITY);
    /// let mut model2 = problem.create_model(Sense::Minimise)?;
    /// ```
    pub fn create_model(&self, sense: Sense) -> Result<Model, HighsStatus> {
        let mut highs = HighsPtr::default();
        highs.make_quiet();

        let (astart, aindex, avalue) = self.to_compressed_matrix_form();

        unsafe {
            highs_call!(Highs_passLp(
                highs.mut_ptr(),
                c(self.num_col),
                c(self.num_row),
                c(self.num_nz),
                MATRIX_FORMAT_COLUMN_WISE,
                OBJECTIVE_SENSE_MINIMIZE,
                self.offset,
                self.col_cost.as_ptr(),
                self.col_lower.as_ptr(),
                self.col_upper.as_ptr(),
                self.row_lower.as_ptr(),
                self.row_upper.as_ptr(),
                astart.as_ptr(),
                aindex.as_ptr(),
                avalue.as_ptr()
            ))
        }?;

        let mut model = Model { highs };
        model.set_sense(sense);
        Ok(model)
    }

    /// Change bounds of a row constraint.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of bounds.
    #[inline]
    pub fn change_row_bounds(&mut self, row: usize, lower: f64, upper: f64) {
        assert!(row < self.num_row, "Row index out of bounds");
        self.row_lower[row] = lower;
        self.row_upper[row] = upper;
    }

    /// Change bounds of a column variable.
    ///
    /// # Panics
    ///
    /// Panics if `col` is out of bounds.
    #[inline]
    pub fn change_col_bounds(&mut self, col: usize, lower: f64, upper: f64) {
        assert!(col < self.num_col, "Column index out of bounds");
        self.col_lower[col] = lower;
        self.col_upper[col] = upper;
    }

    /// Change a coefficient in the constraint matrix.
    ///
    /// Handles sparse matrix structure correctly:
    /// - If the coefficient exists, updates it
    /// - If the coefficient is being set to 0, removes it
    /// - If a new non-zero coefficient is added, inserts it in sorted order
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `Err` if indices are out of bounds.
    pub fn change_coefficient(
        &mut self,
        row: usize,
        col: usize,
        value: f64,
    ) -> Result<(), String> {
        if col >= self.num_col {
            return Err(format!(
                "Column index {} out of bounds (num_col={})",
                col, self.num_col
            ));
        }
        if row >= self.num_row {
            return Err(format!(
                "Row index {} out of bounds (num_row={})",
                row, self.num_row
            ));
        }

        let (ref mut row_indices, ref mut values) = self.columns[col];
        let row_i32 = row as c_int;

        if let Some(pos) = row_indices.iter().position(|&r| r == row_i32) {
            // Coefficient exists
            if value == 0.0 {
                // Remove the coefficient
                row_indices.remove(pos);
                values.remove(pos);
                self.num_nz -= 1;
            } else {
                // Update the coefficient
                values[pos] = value;
            }
        } else if value != 0.0 {
            // Insert new non-zero coefficient in sorted order
            let insert_pos = row_indices
                .iter()
                .position(|&r| r > row_i32)
                .unwrap_or(row_indices.len());
            row_indices.insert(insert_pos, row_i32);
            values.insert(insert_pos, value);
            self.num_nz += 1;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct HighsPtr(*mut c_void);

impl Drop for HighsPtr {
    fn drop(&mut self) {
        unsafe { Highs_destroy(self.0) }
    }
}

impl Default for HighsPtr {
    fn default() -> Self {
        Self(unsafe { Highs_create() })
    }
}

impl Clone for HighsPtr {
    fn clone(&self) -> Self {
        Self(unsafe { Highs_create() })
    }
}

impl HighsPtr {
    #[allow(dead_code)] // Used in test_highs_ptr_clone_creates_independent_instance
    const fn ptr(&self) -> *const c_void {
        self.0
    }

    // Needed until https://github.com/ERGO-Code/HiGHS/issues/479 is fixed
    unsafe fn unsafe_mut_ptr(&self) -> *mut c_void {
        self.0
    }

    fn mut_ptr(&mut self) -> *mut c_void {
        self.0
    }

    /// Prevents writing anything to the standard output when solving the model
    pub fn make_quiet(&mut self) {
        // setting log_file seems to cause a double free in Highs.
        // See https://github.com/rust-or/highs/issues/3
        // self.set_option(&b"log_file"[..], "");
        self.set_option(&b"output_flag"[..], false);
        self.set_option(&b"log_to_console"[..], false);
    }

    /// Set a custom parameter on the model
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(
        &mut self,
        option: STR,
        value: V,
    ) {
        let c_str = CString::new(option).expect("invalid option name");
        let status =
            unsafe { value.apply_to_highs(self.mut_ptr(), c_str.as_ptr()) };
        try_handle_status(status, "Highs_setOptionValue")
            .expect("An error was encountered in HiGHS.");
    }

    /// Number of variables
    fn num_cols(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumCols(self.0) };
        n.try_into()
    }

    /// Number of constraints
    fn num_rows(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumRows(self.0) };
        n.try_into()
    }
}

fn try_handle_status(
    status: c_int,
    #[allow(unused_variables)] msg: &str,
) -> Result<HighsStatus, HighsStatus> {
    let status_enum = HighsStatus::try_from(status)
        .expect("HiGHS returned an unexpected status value. Please report it as a bug to https://github.com/rust-or/highs/issues");
    match status_enum {
        status @ HighsStatus::OK => Ok(status),
        status @ HighsStatus::Warning => {
            // PERFORMANCE: HiGHS warnings during row additions are common
            // in cascade systems and large-scale problems. They don't affect
            // correctness or optimality. Only log in debug builds to avoid
            // I/O overhead in hot paths.
            Ok(status)
        }
        error => Err(error),
    }
}

/// Whether to maximize or minimize the objective function
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    /// max
    Maximise = OBJECTIVE_SENSE_MAXIMIZE as isize,
    /// min
    Minimise = OBJECTIVE_SENSE_MINIMIZE as isize,
}

/// A model to solve
#[derive(Debug, Clone)]
pub struct Model {
    highs: HighsPtr,
}

unsafe impl Send for Model {}

unsafe impl Sync for Model {}

impl Model {
    /// Set the optimization sense (minimize by default)
    pub fn set_sense(&mut self, sense: Sense) {
        let ret = unsafe {
            Highs_changeObjectiveSense(self.highs.mut_ptr(), sense as c_int)
        };
        assert_eq!(ret, STATUS_OK, "changeObjectiveSense failed");
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    /// If the given problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// Returns an error if the problem is incoherent
    pub fn try_new(problem: Problem) -> Result<Self, HighsStatus> {
        let mut highs = HighsPtr::default();
        highs.make_quiet();
        let problem: Problem = problem;
        let (astart, aindex, avalue) = problem.to_compressed_matrix_form();
        unsafe {
            highs_call!(Highs_passLp(
                highs.mut_ptr(),
                c(problem.num_col),
                c(problem.num_row),
                c(problem.num_nz),
                MATRIX_FORMAT_COLUMN_WISE,
                OBJECTIVE_SENSE_MINIMIZE,
                problem.offset,
                problem.col_cost.as_ptr(),
                problem.col_lower.as_ptr(),
                problem.col_upper.as_ptr(),
                problem.row_lower.as_ptr(),
                problem.row_upper.as_ptr(),
                astart.as_ptr(),
                aindex.as_ptr(),
                avalue.as_ptr()
            ))
            .map(|_| Self { highs })
        }
    }

    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(
        &mut self,
        option: STR,
        value: V,
    ) {
        self.highs.set_option(option, value)
    }

    /// Find the optimal value for the problem, panic if the problem is incoherent
    pub fn solve(&mut self) {
        self.try_solve().expect("HiGHS error: invalid problem")
    }

    /// Find the optimal value for the problem, return an error if the problem is incoherent
    pub fn try_solve(&mut self) -> Result<(), HighsStatus> {
        unsafe { highs_call!(Highs_run(self.highs.mut_ptr())) }?;
        Ok(())
    }

    pub fn add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (usize, f64)>,
    ) -> usize {
        self.try_add_row(bounds, row_factors)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to add a new constraint to the highs model.
    ///
    /// Uses thread-local buffers to eliminate per-call allocations.
    ///
    /// Returns the added row index, or the error status value if HIGHS returned an error status.
    pub fn try_add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (usize, f64)>,
    ) -> Result<usize, HighsStatus> {
        ROW_COLS_BUFFER.with(|cols_cell| {
            ROW_VALS_BUFFER.with(|vals_cell| {
                let mut cols = cols_cell.borrow_mut();
                let mut vals = vals_cell.borrow_mut();

                // Ensure initial capacity on first use
                if cols.capacity() == 0 {
                    cols.reserve(ROW_BUFFER_INITIAL_CAPACITY);
                    vals.reserve(ROW_BUFFER_INITIAL_CAPACITY);
                }

                // Clear and populate buffers
                cols.clear();
                vals.clear();

                for (col, val) in row_factors {
                    cols.push(col as HighsInt);
                    vals.push(val);
                }

                unsafe {
                    highs_call!(Highs_addRow(
                        self.highs.mut_ptr(),
                        bound_value(bounds.start_bound())
                            .unwrap_or(f64::NEG_INFINITY),
                        bound_value(bounds.end_bound())
                            .unwrap_or(f64::INFINITY),
                        cols.len() as HighsInt,
                        cols.as_ptr(),
                        vals.as_ptr()
                    ))
                }?;

                Ok(self.highs.num_rows()? - 1)
            })
        })
    }

    pub fn change_rows_bounds(&mut self, row: usize, lower: f64, upper: f64) {
        self.try_change_rows_bounds(row, lower, upper)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e));
    }

    // /// Tries to set new bounds for a row. The expected index here begins counting from 1, not from 0!!!!
    // ///
    // /// Returns the added row index, or the error status value if HIGHS returned an error status.
    pub fn try_change_rows_bounds(
        &mut self,
        row: usize,
        lower: f64,
        upper: f64,
    ) -> Result<(), HighsStatus> {
        let num_rows = self.highs.num_rows().expect("invalid number of rows");

        if row >= num_rows {
            return Err(HighsStatus::Error);
        }

        unsafe {
            highs_call!(Highs_changeRowBounds(
                self.highs.mut_ptr(),
                c(row),
                lower,
                upper
            ))
        }?;

        Ok(())
    }

    /// Change bounds for multiple rows at once using a batch operation.
    ///
    /// This is significantly faster than calling `change_rows_bounds` in a loop
    /// because it makes a single FFI call to HiGHS's batch API.
    ///
    /// # Arguments
    ///
    /// * `row_indices` - Array of row indices to update
    /// * `lower_bounds` - Array of new lower bounds (must have same length as `row_indices`)
    /// * `upper_bounds` - Array of new upper bounds (must have same length as `row_indices`)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `Err(HighsStatus::Error)` if:
    /// - Array lengths don't match
    /// - Any row index is out of bounds
    /// - HiGHS returns an error
    ///
    /// # Performance
    ///
    /// Uses `Highs_changeRowsBoundsBySet` for O(n) batch update with single FFI call.
    /// Reduces allocation overhead from 3 million calls to ~60 calls (one per stage).
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.change_rows_bounds_batch(
    ///     &[0, 5, 10],      // row indices
    ///     &[1.0, 2.0, 3.0], // lower bounds
    ///     &[5.0, 6.0, 7.0], // upper bounds
    /// )?;
    /// ```
    pub fn change_rows_bounds_batch(
        &mut self,
        row_indices: &[HighsInt],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> Result<(), HighsStatus> {
        // Validate array lengths match
        if row_indices.len() != lower_bounds.len()
            || row_indices.len() != upper_bounds.len()
        {
            return Err(HighsStatus::Error);
        }

        // Empty arrays are a no-op
        if row_indices.is_empty() {
            return Ok(());
        }

        unsafe {
            highs_call!(Highs_changeRowsBoundsBySet(
                self.highs.mut_ptr(),
                row_indices.len() as HighsInt,
                row_indices.as_ptr(),
                lower_bounds.as_ptr(),
                upper_bounds.as_ptr()
            ))
        }?;

        Ok(())
    }

    /// Deletes a row from the built model.
    ///
    /// Uses thread-local buffer to eliminate per-call allocation.
    /// Assumes it is lower-bounded and returns the RHS.
    pub fn delete_row(&self, row_index: usize) -> Result<(), HighsStatus> {
        DELETE_ROW_BUFFER.with(|buffer_cell| {
            let mut buffer = buffer_cell.borrow_mut();
            buffer.clear();
            buffer.push(row_index as HighsInt);

            unsafe {
                Highs_deleteRowsBySet(
                    self.highs.unsafe_mut_ptr(),
                    1,
                    buffer.as_ptr(),
                );
            }
            Ok(())
        })
    }

    pub fn change_column_bounds(&mut self, col: usize, lower: f64, upper: f64) {
        self.try_change_column_bounds(col, lower, upper)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e));
    }

    // /// Tries to set new bounds for a column. The expected index here begins counting from 1, not from 0!!!!
    // ///
    // /// Returns the added column index, or the error status value if HIGHS returned an error status.
    pub fn try_change_column_bounds(
        &mut self,
        col: usize,
        lower: f64,
        upper: f64,
    ) -> Result<(), HighsStatus> {
        let num_columns =
            self.highs.num_cols().expect("invalid number of columns");

        if col >= num_columns {
            return Err(HighsStatus::Error);
        }

        unsafe {
            highs_call!(Highs_changeColBounds(
                self.highs.mut_ptr(),
                c(col),
                lower,
                upper
            ))
        }?;

        Ok(())
    }

    /// The status of the solution. Should be Optimal if everything went well.
    pub fn status(&self) -> HighsModelStatus {
        let model_status =
            unsafe { Highs_getModelStatus(self.highs.unsafe_mut_ptr()) };
        HighsModelStatus::try_from(model_status).unwrap()
    }

    /// Get the solution to the problem
    pub fn get_solution(&self) -> Solution {
        let cols = self.num_cols();
        let rows = self.num_rows();
        let mut colvalue: Vec<f64> = vec![0.; cols];
        let mut coldual: Vec<f64> = vec![0.; cols];
        let mut rowvalue: Vec<f64> = vec![0.; rows];
        let mut rowdual: Vec<f64> = vec![0.; rows];

        // Get the primal and dual solution
        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                colvalue.as_mut_ptr(),
                coldual.as_mut_ptr(),
                rowvalue.as_mut_ptr(),
                rowdual.as_mut_ptr(),
            );
        }

        Solution {
            colvalue,
            coldual,
            rowvalue,
            rowdual,
        }
    }

    /// Gets the solution, writing into an existing buffer to avoid allocation.
    ///
    /// This is more efficient than `get_solution()` when the buffer has
    /// sufficient capacity, as it avoids heap allocations.
    /// The buffer is resized if necessary.
    pub fn get_solution_into(&self, solution: &mut Solution) {
        let cols = self.num_cols();
        let rows = self.num_rows();
        solution.ensure_capacity(cols, rows);

        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                solution.colvalue.as_mut_ptr(),
                solution.coldual.as_mut_ptr(),
                solution.rowvalue.as_mut_ptr(),
                solution.rowdual.as_mut_ptr(),
            );
        }
    }

    /// Get the basis status of the problem
    pub fn get_basis(&self) -> Basis {
        let cols = self.num_cols();
        let rows = self.num_rows();
        let mut raw_colstatus: Vec<c_int> = vec![0; cols];
        let mut raw_rowstatus: Vec<c_int> = vec![0; rows];

        // Get the primal and dual solution
        unsafe {
            Highs_getBasis(
                self.highs.unsafe_mut_ptr(),
                raw_colstatus.as_mut_ptr(),
                raw_rowstatus.as_mut_ptr(),
            );
        }

        let colstatus = raw_colstatus.iter().map(|x| *x as usize).collect();
        let rowstatus = raw_rowstatus.iter().map(|x| *x as usize).collect();

        Basis {
            colstatus,
            rowstatus,
        }
    }

    /// Gets the basis, writing into an existing buffer to avoid allocation.
    ///
    /// Uses thread-local scratch buffers for the FFI call to avoid allocation
    /// of raw c_int vectors on each call.
    pub fn get_basis_into(&self, basis: &mut Basis) {
        thread_local! {
            static RAW_COL_BUFFER: RefCell<Vec<c_int>> = const { RefCell::new(Vec::new()) };
            static RAW_ROW_BUFFER: RefCell<Vec<c_int>> = const { RefCell::new(Vec::new()) };
        }

        let cols = self.num_cols();
        let rows = self.num_rows();
        basis.ensure_size(cols, rows);

        RAW_COL_BUFFER.with(|raw_col| {
            RAW_ROW_BUFFER.with(|raw_row| {
                let mut raw_col = raw_col.borrow_mut();
                let mut raw_row = raw_row.borrow_mut();

                // Resize scratch buffers if needed
                if raw_col.len() < cols {
                    raw_col.resize(cols, 0);
                }
                if raw_row.len() < rows {
                    raw_row.resize(rows, 0);
                }

                unsafe {
                    Highs_getBasis(
                        self.highs.unsafe_mut_ptr(),
                        raw_col.as_mut_ptr(),
                        raw_row.as_mut_ptr(),
                    );
                }

                // Convert c_int -> usize into the output buffer
                for (i, &raw) in raw_col.iter().take(cols).enumerate() {
                    basis.colstatus[i] = raw as usize;
                }
                for (i, &raw) in raw_row.iter().take(rows).enumerate() {
                    basis.rowstatus[i] = raw as usize;
                }
            });
        });
    }

    /// Hot-starts at the initial guess. See HIGHS documentation for further details.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    ///
    /// If the data passed in do not have the correct lengths.
    /// `cols` and `col_duals` should have the lengths of `num_cols`.
    /// `rows` and `row_duals` should have the lengths of `num_rows`.
    pub fn set_basis(
        &mut self,
        colstatus: Option<&[usize]>,
        rowstatus: Option<&[usize]>,
    ) {
        self.try_set_basis(colstatus, rowstatus)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to hot-start using an initial guess by passing the column and row primal and dual solution values.
    /// See highs_c_api.h for further details.
    ///
    /// If the data passed in do not have the correct lengths, an `Err` is returned.
    /// `cols` and `col_duals` should have the lengths of `num_cols`.
    /// `rows` and `row_duals` should have the lengths of `num_rows`.
    pub fn try_set_basis(
        &mut self,
        colstatus: Option<&[usize]>,
        rowstatus: Option<&[usize]>,
    ) -> Result<(), HighsStatus> {
        let num_cols = self.highs.num_cols()?;
        let num_rows = self.highs.num_rows()?;
        if let Some(colstatus) = colstatus {
            if colstatus.len() != num_cols {
                return Err(HighsStatus::Error);
            }
        }
        if let Some(rowstatus) = rowstatus {
            if rowstatus.len() != num_rows {
                return Err(HighsStatus::Error);
            }
        }

        let raw_colstatus: &[c_int] = &colstatus
            .unwrap()
            .iter()
            .map(|x| c(*x))
            .collect::<Vec<c_int>>()[..];
        let raw_rowstatus: &[c_int] = &rowstatus
            .unwrap()
            .iter()
            .map(|x| c(*x))
            .collect::<Vec<c_int>>()[..];

        unsafe {
            highs_call!(Highs_setBasis(
                self.highs.mut_ptr(),
                Some(raw_colstatus)
                    .map(|x| { x.as_ptr() })
                    .unwrap_or(null()),
                Some(raw_rowstatus)
                    .map(|x| { x.as_ptr() })
                    .unwrap_or(null())
            ))
        }?;
        Ok(())
    }

    /// Extract basis into StoredBasis for optional reuse across iterations.
    ///
    /// This extracts the basis from the current model into an owned StoredBasis
    /// that can persist after the model is dropped.
    pub fn get_stored_basis(&self) -> StoredBasis {
        let basis = self.get_basis();
        StoredBasis {
            colstatus: basis.colstatus,
            rowstatus: basis.rowstatus,
        }
    }

    /// Extract basis into existing StoredBasis buffer (allocation-free).
    ///
    /// Clears and repopulates the provided buffer with current basis state.
    pub fn get_stored_basis_into(&self, stored: &mut StoredBasis) {
        let cols = self.num_cols();
        let rows = self.num_rows();

        // Ensure capacity and resize
        stored.colstatus.clear();
        stored.rowstatus.clear();
        stored.colstatus.reserve(cols);
        stored.rowstatus.reserve(rows);

        let basis = self.get_basis();
        stored.colstatus.extend_from_slice(&basis.colstatus);
        stored.rowstatus.extend_from_slice(&basis.rowstatus);
    }

    /// Apply stored basis for warm-starting.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(HighsStatus::Error)` if dimensions don't match.
    pub fn apply_stored_basis(
        &mut self,
        basis: &StoredBasis,
    ) -> Result<(), HighsStatus> {
        if !basis.is_compatible(self.num_cols(), self.num_rows()) {
            return Err(HighsStatus::Error);
        }
        self.try_set_basis(Some(&basis.colstatus), Some(&basis.rowstatus))
    }

    pub fn get_objective_value(&self) -> f64 {
        unsafe { Highs_getObjectiveValue(self.highs.unsafe_mut_ptr()) }
    }

    /// Clears the solved model
    pub fn clear_solver(&self) {
        unsafe { Highs_clearSolver(self.highs.unsafe_mut_ptr()) };
    }

    /// Number of variables
    pub fn num_cols(&self) -> usize {
        self.highs.num_cols().expect("invalid number of columns")
    }

    /// Number of constraints
    pub fn num_rows(&self) -> usize {
        self.highs.num_rows().expect("invalid number of rows")
    }

    /// Change a single coefficient in the constraint matrix.
    ///
    /// Updates the coefficient at position (row, col) in the constraint matrix.
    /// This is useful for modifying preallocated cut constraints without adding new rows.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    /// * `value` - New coefficient value
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `Err(HighsStatus::Error)` on failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.change_coefficient(5, 10, 1.5)?;  // Set A[5,10] = 1.5
    /// ```
    pub fn change_coefficient(
        &mut self,
        row: usize,
        col: usize,
        value: f64,
    ) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_changeCoeff(
                self.highs.mut_ptr(),
                c(row),
                c(col),
                value
            ))
        }?;
        Ok(())
    }

    /// Add multiple rows at once (batch operation).
    ///
    /// Uses CSR (Compressed Sparse Row) format for efficient sparse matrix transfer.
    /// This is significantly faster than calling `add_row` in a loop when adding
    /// many constraints at once (e.g., preallocating cut slots).
    ///
    /// # Arguments
    ///
    /// * `num_rows` - Number of rows to add
    /// * `lower_bounds` - Lower bounds for each row (length = `num_rows`)
    /// * `upper_bounds` - Upper bounds for each row (length = `num_rows`)
    /// * `astart` - CSR row start indices (length = `num_rows + 1`)
    /// * `aindex` - Column indices for non-zeros
    /// * `avalue` - Coefficient values for non-zeros
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `Err(HighsStatus::Error)` on failure.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if array lengths are inconsistent.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Add 2 rows: x0 + 2*x1 >= 5 and 3*x1 + x2 >= 7
    /// model.add_rows_batch(
    ///     2,
    ///     &[5.0, 7.0],           // lower bounds
    ///     &[f64::INFINITY; 2],   // upper bounds
    ///     &[0, 2, 4],            // astart: row 0 has 2 NZ, row 1 has 2 NZ
    ///     &[0, 1, 1, 2],         // aindex: col indices
    ///     &[1.0, 2.0, 3.0, 1.0], // avalue: coefficients
    /// )?;
    /// ```
    pub fn add_rows_batch(
        &mut self,
        num_rows: usize,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        astart: &[HighsInt],
        aindex: &[HighsInt],
        avalue: &[f64],
    ) -> Result<(), HighsStatus> {
        debug_assert_eq!(lower_bounds.len(), num_rows);
        debug_assert_eq!(upper_bounds.len(), num_rows);
        debug_assert_eq!(astart.len(), num_rows + 1);

        let num_nz = *astart.last().unwrap_or(&0) as usize;
        debug_assert_eq!(aindex.len(), num_nz);
        debug_assert_eq!(avalue.len(), num_nz);

        unsafe {
            highs_call!(Highs_addRows(
                self.highs.mut_ptr(),
                c(num_rows),
                lower_bounds.as_ptr(),
                upper_bounds.as_ptr(),
                c(num_nz),
                astart.as_ptr(),
                aindex.as_ptr(),
                avalue.as_ptr()
            ))
        }?;
        Ok(())
    }
}

/// Concrete values of the solution
#[derive(Clone, Debug)]
pub struct Solution {
    pub colvalue: Vec<f64>,
    pub coldual: Vec<f64>,
    pub rowvalue: Vec<f64>,
    pub rowdual: Vec<f64>,
}

impl Solution {
    /// Creates a Solution with preallocated buffers of the specified sizes.
    pub fn with_capacity(cols: usize, rows: usize) -> Self {
        Self {
            colvalue: vec![0.0; cols],
            coldual: vec![0.0; cols],
            rowvalue: vec![0.0; rows],
            rowdual: vec![0.0; rows],
        }
    }

    /// Ensures buffers have sufficient capacity, resizing if needed.
    /// Returns true if any resize was performed.
    #[inline]
    pub fn ensure_capacity(&mut self, cols: usize, rows: usize) -> bool {
        let mut resized = false;
        if self.colvalue.len() < cols {
            self.colvalue.resize(cols, 0.0);
            self.coldual.resize(cols, 0.0);
            resized = true;
        }
        if self.rowvalue.len() < rows {
            self.rowvalue.resize(rows, 0.0);
            self.rowdual.resize(rows, 0.0);
            resized = true;
        }
        resized
    }
}

/// Basis statuses for a problem with concrete solution
#[derive(Clone, Debug)]
pub struct Basis {
    colstatus: Vec<usize>,
    rowstatus: Vec<usize>,
}

unsafe impl Send for Basis {}

impl Default for Basis {
    fn default() -> Self {
        Self::new()
    }
}

impl Basis {
    pub fn new() -> Self {
        Self {
            colstatus: vec![],
            rowstatus: vec![],
        }
    }

    pub fn with_capacity(num_cols: usize, num_rows: usize) -> Self {
        Self {
            colstatus: Vec::<usize>::with_capacity(num_cols),
            rowstatus: Vec::<usize>::with_capacity(num_rows),
        }
    }

    /// Creates a Basis with initialized (not just reserved) buffers.
    pub fn with_size(num_cols: usize, num_rows: usize) -> Self {
        Self {
            colstatus: vec![0; num_cols],
            rowstatus: vec![0; num_rows],
        }
    }

    /// Ensures buffers have sufficient size, resizing if needed.
    #[inline]
    pub fn ensure_size(&mut self, cols: usize, rows: usize) {
        if self.colstatus.len() < cols {
            self.colstatus.resize(cols, 0);
        }
        if self.rowstatus.len() < rows {
            self.rowstatus.resize(rows, 0);
        }
    }

    /// The basis status for each of the columns
    pub fn columns(&self) -> &[usize] {
        &self.colstatus
    }

    /// The basis status for each of the rows
    pub fn rows(&self) -> &[usize] {
        &self.rowstatus
    }
}

/// Basis status stored for cross-iteration warm-starting.
///
/// This is a standalone storage type that owns basis data, allowing
/// basis to persist across Model drops for the per-iteration architecture.
///
/// # Optional Usage
///
/// - Training: Basis cached and applied for warm-starting
/// - Simulation: Basis not used (cold-start for reproducibility)
#[derive(Clone, Debug, Default)]
pub struct StoredBasis {
    pub colstatus: Vec<usize>,
    pub rowstatus: Vec<usize>,
}

impl StoredBasis {
    /// Create a new empty StoredBasis.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a StoredBasis with preallocated capacity.
    pub fn with_capacity(num_cols: usize, num_rows: usize) -> Self {
        Self {
            colstatus: Vec::with_capacity(num_cols),
            rowstatus: Vec::with_capacity(num_rows),
        }
    }

    /// Check if basis is compatible with model dimensions.
    #[inline]
    pub fn is_compatible(&self, num_cols: usize, num_rows: usize) -> bool {
        self.colstatus.len() == num_cols && self.rowstatus.len() == num_rows
    }

    /// Check if basis is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.colstatus.is_empty() && self.rowstatus.is_empty()
    }

    /// Clear basis data.
    #[inline]
    pub fn clear(&mut self) {
        self.colstatus.clear();
        self.rowstatus.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_conversion_function() {
        // Test the private c() conversion function (usize -> HighsInt/i32)
        assert_eq!(c(0), 0);
        assert_eq!(c(1), 1);
        assert_eq!(c(100), 100);
        assert_eq!(c(1000000), 1000000);
    }

    #[test]
    fn test_try_handle_status_ok() {
        // Test try_handle_status with OK status
        use highs_sys::STATUS_OK;
        let result = try_handle_status(STATUS_OK, "test_operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_handle_status_warning() {
        // Test try_handle_status with Warning status (should still be OK)
        use highs_sys::STATUS_WARNING;
        let result = try_handle_status(STATUS_WARNING, "test_operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_handle_status_error() {
        // Test try_handle_status with Error status
        use highs_sys::STATUS_ERROR;
        let result = try_handle_status(STATUS_ERROR, "test_operation");
        assert!(result.is_err());
    }

    #[test]
    fn test_problem_to_compressed_matrix_form_empty() {
        // Test conversion with empty problem
        let problem = Problem::new();
        let (astart, aindex, avalue) = problem.to_compressed_matrix_form();

        assert_eq!(astart, vec![0]);
        assert!(aindex.is_empty());
        assert!(avalue.is_empty());
    }

    #[test]
    fn test_problem_to_compressed_matrix_form_single_column() {
        // Test conversion with single column
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(1.0.., [(0, 2.5)]);

        let (astart, aindex, avalue) = problem.to_compressed_matrix_form();

        assert_eq!(astart, vec![0, 1]); // 0, then 1 element
        assert_eq!(aindex, vec![0]); // Row index 0
        assert_eq!(avalue, vec![2.5]); // Coefficient 2.5
    }

    #[test]
    fn test_problem_to_compressed_matrix_form_multiple_columns() {
        // Test conversion with multiple columns and rows
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..); // Column 0
        problem.add_column(2.0, 0.0..); // Column 1
        problem.add_row(1.0.., [(0, 1.0), (1, 2.0)]); // Row 0: x + 2y >= 1
        problem.add_row(3.0.., [(0, 3.0)]); // Row 1: 3x >= 3

        let (astart, aindex, avalue) = problem.to_compressed_matrix_form();

        // astart: cumulative starts [0, 2, 3] (col 0 has 2 entries, col 1 has 1)
        assert_eq!(astart, vec![0, 2, 3]);
        // aindex: row indices for each entry
        assert_eq!(aindex, vec![0, 1, 0]); // col0: rows 0,1; col1: row 0
                                           // avalue: coefficients
        assert_eq!(avalue, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_highs_ptr_clone_creates_independent_instance() {
        // Test that cloning HighsPtr creates independent HiGHS instance
        let ptr1 = HighsPtr::default();
        let ptr2 = ptr1.clone();

        // Pointers should be different (independent instances)
        assert_ne!(ptr1.ptr(), ptr2.ptr());
    }

    #[test]
    fn test_highs_model_status_try_from_valid() {
        // Test conversion of valid status codes using highs_sys constants
        use highs_sys::*;
        assert!(matches!(
            HighsModelStatus::try_from(MODEL_STATUS_NOTSET),
            Ok(HighsModelStatus::NotSet)
        ));
        assert!(matches!(
            HighsModelStatus::try_from(MODEL_STATUS_OPTIMAL),
            Ok(HighsModelStatus::Optimal)
        ));
        assert!(matches!(
            HighsModelStatus::try_from(MODEL_STATUS_INFEASIBLE),
            Ok(HighsModelStatus::Infeasible)
        ));
        assert!(matches!(
            HighsModelStatus::try_from(MODEL_STATUS_UNBOUNDED),
            Ok(HighsModelStatus::Unbounded)
        ));
    }

    #[test]
    fn test_highs_model_status_try_from_invalid() {
        // Test conversion of invalid status code
        let result = HighsModelStatus::try_from(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_basis_new() {
        // Test Basis::new creates empty basis
        let basis = Basis::new();
        assert_eq!(basis.columns().len(), 0);
        assert_eq!(basis.rows().len(), 0);
    }

    #[test]
    fn test_basis_with_capacity() {
        // Test Basis::with_capacity reserves space
        let basis = Basis::with_capacity(10, 5);
        assert_eq!(basis.columns().len(), 0);
        assert_eq!(basis.rows().len(), 0);
        // Capacity is set but length is 0 (we can't directly test capacity)
    }

    #[test]
    fn test_basis_default() {
        // Test Basis::default() uses new()
        let basis = Basis::default();
        assert_eq!(basis.columns().len(), 0);
        assert_eq!(basis.rows().len(), 0);
    }

    #[test]
    fn test_solution_with_capacity() {
        let sol = Solution::with_capacity(10, 5);
        assert_eq!(sol.colvalue.len(), 10);
        assert_eq!(sol.coldual.len(), 10);
        assert_eq!(sol.rowvalue.len(), 5);
        assert_eq!(sol.rowdual.len(), 5);
        // All values should be zero-initialized
        assert!(sol.colvalue.iter().all(|&x| x == 0.0));
        assert!(sol.rowdual.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_solution_ensure_capacity_no_resize() {
        let mut sol = Solution::with_capacity(10, 5);
        let resized = sol.ensure_capacity(5, 3);
        assert!(!resized);
        assert_eq!(sol.colvalue.len(), 10);
        assert_eq!(sol.rowvalue.len(), 5);
    }

    #[test]
    fn test_solution_ensure_capacity_resize() {
        let mut sol = Solution::with_capacity(5, 3);
        let resized = sol.ensure_capacity(10, 8);
        assert!(resized);
        assert_eq!(sol.colvalue.len(), 10);
        assert_eq!(sol.coldual.len(), 10);
        assert_eq!(sol.rowvalue.len(), 8);
        assert_eq!(sol.rowdual.len(), 8);
    }

    #[test]
    fn test_basis_with_size() {
        let basis = Basis::with_size(10, 5);
        assert_eq!(basis.columns().len(), 10);
        assert_eq!(basis.rows().len(), 5);
        // All values should be zero-initialized
        assert!(basis.columns().iter().all(|&x| x == 0));
        assert!(basis.rows().iter().all(|&x| x == 0));
    }

    #[test]
    fn test_basis_ensure_size_no_resize() {
        let mut basis = Basis::with_size(10, 5);
        basis.ensure_size(5, 3);
        assert_eq!(basis.columns().len(), 10);
        assert_eq!(basis.rows().len(), 5);
    }

    #[test]
    fn test_basis_ensure_size_resize() {
        let mut basis = Basis::with_size(5, 3);
        basis.ensure_size(10, 8);
        assert_eq!(basis.columns().len(), 10);
        assert_eq!(basis.rows().len(), 8);
    }

    #[test]
    fn test_change_rows_bounds_batch_empty() {
        // Test batch update with empty arrays (should be no-op)
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(0.0..=10.0, [(0, 1.0)]);

        let mut model = problem.try_optimise(Sense::Minimise).unwrap();

        // Empty batch should succeed
        let result = model.change_rows_bounds_batch(&[], &[], &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_change_rows_bounds_batch_single_row() {
        // Test batch update with single row
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(0.0..=10.0, [(0, 1.0)]);

        let mut model = problem.try_optimise(Sense::Minimise).unwrap();

        // Update single row
        let result = model.change_rows_bounds_batch(&[0], &[5.0], &[15.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_change_rows_bounds_batch_multiple_rows() {
        // Test batch update with multiple rows
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(0.0..=10.0, [(0, 1.0)]); // Row 0
        problem.add_row(0.0..=20.0, [(0, 2.0)]); // Row 1
        problem.add_row(0.0..=30.0, [(0, 3.0)]); // Row 2

        let mut model = problem.try_optimise(Sense::Minimise).unwrap();

        // Update rows 0 and 2 in batch
        let result =
            model.change_rows_bounds_batch(&[0, 2], &[5.0, 15.0], &[8.0, 25.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_change_rows_bounds_batch_mismatched_lengths() {
        // Test error when array lengths don't match
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(0.0..=10.0, [(0, 1.0)]);

        let mut model = problem.try_optimise(Sense::Minimise).unwrap();

        // Mismatched lengths should fail
        let result = model.change_rows_bounds_batch(&[0, 1], &[5.0], &[15.0]);
        assert!(result.is_err());

        let result2 =
            model.change_rows_bounds_batch(&[0], &[5.0, 6.0], &[15.0]);
        assert!(result2.is_err());
    }

    #[test]
    fn test_change_rows_bounds_batch_preserves_solve() {
        // Test that batch update preserves model solvability
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);
        problem.add_row(5.0..=50.0, [(0, 1.0)]); // Row 0: 5 <= x <= 50

        let mut model = problem.try_optimise(Sense::Minimise).unwrap();

        // Update bounds to tighten constraint
        let result = model.change_rows_bounds_batch(&[0], &[10.0], &[40.0]);
        assert!(result.is_ok());

        // Model should still solve
        model.solve();
        assert_eq!(model.status(), HighsModelStatus::Optimal);

        // Optimal x should be 10 (tightened lower bound)
        let solution = model.get_solution();
        assert!((solution.colvalue[0] - 10.0).abs() < 1e-6);
    }

    // ========================================
    // T-110: Problem::create_model() tests
    // ========================================

    #[test]
    fn test_create_model_does_not_consume() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=10.0);
        problem.add_row(5.0.., [(0, 1.0)]);

        let _ = problem.create_model(Sense::Minimise).unwrap();
        // Problem is still accessible
        assert_eq!(problem.num_col, 1);
        assert_eq!(problem.num_row, 1);
    }

    #[test]
    fn test_create_model_independent() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);
        problem.add_row(5.0.., [(0, 1.0)]);

        let mut model1 = problem.create_model(Sense::Minimise).unwrap();
        model1.solve();
        assert!((model1.get_solution().colvalue[0] - 5.0).abs() < 1e-9);

        // Modify problem
        problem.change_row_bounds(0, 8.0, f64::INFINITY);

        // model1 should be unaffected (still uses original bounds)
        let sol1 = model1.get_solution();
        assert!((sol1.colvalue[0] - 5.0).abs() < 1e-9);

        // New model gets modification
        let mut model2 = problem.create_model(Sense::Minimise).unwrap();
        model2.solve();
        assert!((model2.get_solution().colvalue[0] - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_create_multiple_models() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);
        problem.add_row(5.0.., [(0, 1.0)]);

        // Create multiple models from same problem
        let model1 = problem.create_model(Sense::Minimise).unwrap();
        let model2 = problem.create_model(Sense::Minimise).unwrap();
        let model3 = problem.create_model(Sense::Minimise).unwrap();

        assert_eq!(model1.num_cols(), 1);
        assert_eq!(model2.num_cols(), 1);
        assert_eq!(model3.num_cols(), 1);
    }

    // ========================================
    // T-111: Problem modification tests
    // ========================================

    #[test]
    fn test_problem_change_row_bounds() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);
        problem.add_row(5.0..=10.0, [(0, 1.0)]);

        problem.change_row_bounds(0, 8.0, 15.0);

        assert_eq!(problem.row_lower[0], 8.0);
        assert_eq!(problem.row_upper[0], 15.0);
    }

    #[test]
    fn test_problem_change_col_bounds() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);

        problem.change_col_bounds(0, 5.0, 50.0);

        assert_eq!(problem.col_lower[0], 5.0);
        assert_eq!(problem.col_upper[0], 50.0);
    }

    #[test]
    fn test_problem_change_coefficient_update() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(5.0.., [(0, 1.0)]);

        assert_eq!(problem.num_nz, 1);

        // Update existing coefficient
        problem.change_coefficient(0, 0, 2.0).unwrap();
        assert_eq!(problem.num_nz, 1); // Still 1 non-zero

        // Verify via model creation
        let mut model = problem.create_model(Sense::Minimise).unwrap();
        model.solve();
        // With 2*x >= 5, x >= 2.5
        assert!((model.get_solution().colvalue[0] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_problem_change_coefficient_remove() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(5.0.., [(0, 1.0)]);

        assert_eq!(problem.num_nz, 1);

        // Remove coefficient by setting to 0
        problem.change_coefficient(0, 0, 0.0).unwrap();
        assert_eq!(problem.num_nz, 0);
    }

    #[test]
    fn test_problem_change_coefficient_add() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_column(2.0, 0.0..);
        problem.add_row(5.0.., [(0, 1.0)]); // Only col 0 initially

        assert_eq!(problem.num_nz, 1);

        // Add coefficient for col 1
        problem.change_coefficient(0, 1, 3.0).unwrap();
        assert_eq!(problem.num_nz, 2);
    }

    #[test]
    fn test_problem_change_coefficient_out_of_bounds() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(5.0.., [(0, 1.0)]);

        assert!(problem.change_coefficient(0, 5, 1.0).is_err());
        assert!(problem.change_coefficient(5, 0, 1.0).is_err());
    }

    #[test]
    fn test_problem_changes_reflected_in_model() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);
        problem.add_row(5.0.., [(0, 1.0)]);

        // Change bounds
        problem.change_row_bounds(0, 10.0, f64::INFINITY);

        // Create model and verify changes are reflected
        let mut model = problem.create_model(Sense::Minimise).unwrap();
        model.solve();
        assert!((model.get_solution().colvalue[0] - 10.0).abs() < 1e-9);
    }

    // ========================================
    // T-112: StoredBasis tests
    // ========================================

    #[test]
    fn test_stored_basis_new() {
        let basis = StoredBasis::new();
        assert!(basis.is_empty());
    }

    #[test]
    fn test_stored_basis_with_capacity() {
        let basis = StoredBasis::with_capacity(10, 5);
        assert!(basis.is_empty());
        // Has capacity but no data
    }

    #[test]
    fn test_stored_basis_compatibility() {
        let mut basis = StoredBasis::new();
        basis.colstatus = vec![0; 10];
        basis.rowstatus = vec![0; 5];

        assert!(basis.is_compatible(10, 5));
        assert!(!basis.is_compatible(10, 6));
        assert!(!basis.is_compatible(11, 5));
    }

    #[test]
    fn test_stored_basis_clear() {
        let mut basis = StoredBasis::new();
        basis.colstatus = vec![0; 10];
        basis.rowstatus = vec![0; 5];

        assert!(!basis.is_empty());

        basis.clear();
        assert!(basis.is_empty());
    }

    #[test]
    fn test_warm_start_across_models() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);
        problem.add_row(5.0.., [(0, 1.0)]);

        // Solve model 1 and extract basis
        let mut model1 = problem.create_model(Sense::Minimise).unwrap();
        model1.solve();
        let basis = model1.get_stored_basis();
        drop(model1);

        // Modify problem
        problem.change_row_bounds(0, 6.0, f64::INFINITY);

        // Create new model and apply basis
        let mut model2 = problem.create_model(Sense::Minimise).unwrap();
        model2.apply_stored_basis(&basis).unwrap();
        model2.solve();

        assert_eq!(model2.status(), HighsModelStatus::Optimal);
        assert!((model2.get_solution().colvalue[0] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_basis_dimension_mismatch() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(0.0.., [(0, 1.0)]);

        let mut model = problem.create_model(Sense::Minimise).unwrap();

        let bad_basis = StoredBasis {
            colstatus: vec![0; 5], // Wrong size
            rowstatus: vec![0; 1],
        };

        assert!(model.apply_stored_basis(&bad_basis).is_err());
    }

    #[test]
    fn test_get_stored_basis_into() {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..=100.0);
        problem.add_row(5.0.., [(0, 1.0)]);

        let mut model = problem.create_model(Sense::Minimise).unwrap();
        model.solve();

        let mut stored = StoredBasis::new();
        model.get_stored_basis_into(&mut stored);

        assert!(!stored.is_empty());
        assert!(stored.is_compatible(model.num_cols(), model.num_rows()));
    }
}
