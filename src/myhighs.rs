//! Simplifies the implementation from "highs" crate, only supporting
//! the "RowProblem" variant, while adds new operations that are
//! relevant for performance gains in the context of SDDP.

use std::borrow::Borrow;
use std::convert::TryFrom;
use std::ffi::{c_void, CStr, CString};
use std::fmt::{Debug, Formatter};
use std::num::TryFromIntError;
use std::ops::{Bound, RangeBounds};
use std::os::raw::{c_char, c_int};
use std::ptr::null;

use highs_sys::*;

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

/// The kinds of results of an optimization
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsBasisStatus {
    Lower = 0 as isize,
    Basic = 1 as isize,
    Upper = 2 as isize,
    Zero = 3 as isize,
    NonBasic = 4 as isize,
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

impl<'a> HighsOptionValue for &'a CStr {
    unsafe fn apply_to_highs(
        self,
        highs: *mut c_void,
        option: *const c_char,
    ) -> c_int {
        highs_sys::Highs_setStringOptionValue(highs, option, self.as_ptr())
    }
}

impl<'a> HighsOptionValue for &'a [u8] {
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

impl<'a> HighsOptionValue for &'a str {
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

    fn to_compressed_matrix_form(
        &mut self,
    ) -> (Vec<c_int>, Vec<c_int>, Vec<f64>) {
        let mut astart = Vec::with_capacity(self.num_col);
        astart.push(0);
        let size: usize = self.num_nz;
        let mut aindex = Vec::with_capacity(size);
        let mut avalue = Vec::with_capacity(size);
        for (row_indices, factors) in self.columns.as_slice() {
            aindex.extend_from_slice(&row_indices);
            avalue.extend_from_slice(&factors);
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

impl HighsPtr {
    // To be used instead of unsafe_mut_ptr wherever possible
    #[allow(dead_code)]
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
    msg: &str,
) -> Result<HighsStatus, HighsStatus> {
    let status_enum = HighsStatus::try_from(status)
        .expect("HiGHS returned an unexpected status value. Please report it as a bug to https://github.com/rust-or/highs/issues");
    match status_enum {
        status @ HighsStatus::OK => Ok(status),
        status @ HighsStatus::Warning => {
            println!("HiGHS emitted a warning: {}", msg);
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
#[derive(Debug)]
pub struct Model {
    highs: HighsPtr,
}

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
    /// Panics if the problem is incoherent
    pub fn new(problem: Problem) -> Self {
        Self::try_new(problem).expect("incoherent problem")
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    /// If the given problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// Returns an error if the problem is incoherent
    pub fn try_new(problem: Problem) -> Result<Self, HighsStatus> {
        let mut highs = HighsPtr::default();
        highs.make_quiet();
        let mut problem: Problem = problem.into();
        let offset = 0.0;
        let (astart, aindex, avalue) = problem.to_compressed_matrix_form();
        unsafe {
            highs_call!(Highs_passLp(
                highs.mut_ptr(),
                c(problem.num_col),
                c(problem.num_row),
                c(problem.num_nz),
                MATRIX_FORMAT_COLUMN_WISE,
                OBJECTIVE_SENSE_MINIMIZE,
                offset,
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

    pub fn make_quiet(&mut self) {
        self.highs.make_quiet()
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
    /// Returns the added row index, or the error status value if HIGHS returned an error status.
    pub fn try_add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (usize, f64)>,
    ) -> Result<usize, HighsStatus> {
        let (cols, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();

        unsafe {
            highs_call!(Highs_addRow(
                self.highs.mut_ptr(),
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                cols.len().try_into().unwrap(),
                cols.into_iter()
                    .map(|c| c.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                factors.as_ptr()
            ))
        }?;

        Ok(self.highs.num_rows()? - 1)
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

    // // /// Gets a row from the built model
    // pub fn get_row(&self, row_index: usize) {
    //     let set: Vec<HighsInt> = vec![row_index as HighsInt];
    //     let mut num_row: Vec<HighsInt> = vec![0];
    //     let mut lower: Vec<f64> = vec![0.; 1];
    //     let mut upper: Vec<f64> = vec![0.; 1];
    //     let mut num_nz: Vec<HighsInt> = vec![0; 1];
    //     let mut matrix_start: Vec<HighsInt> = vec![0; 5];
    //     let mut matrix_index: Vec<HighsInt> = vec![0; 5];
    //     let mut matrix_value: Vec<f64> = vec![0.; 5];

    //     unsafe {
    //         Highs_getRowsBySet(
    //             self.highs.unsafe_mut_ptr(),
    //             c(1),
    //             set.as_ptr(),
    //             num_row.as_mut_ptr(),
    //             lower.as_mut_ptr(),
    //             upper.as_mut_ptr(),
    //             num_nz.as_mut_ptr(),
    //             matrix_start.as_mut_ptr(),
    //             matrix_index.as_mut_ptr(),
    //             matrix_value.as_mut_ptr(),
    //         );
    //     }

    //     println!(
    //         "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
    //         num_row,
    //         num_nz,
    //         matrix_start,
    //         matrix_index,
    //         matrix_value,
    //         lower,
    //         upper
    //     )
    // }

    /// Hot-starts at the initial guess. See HIGHS documentation for further details.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    ///
    /// If the data passed in do not have the correct lengths.
    /// `cols` and `col_duals` should have the lengths of `num_cols`.
    /// `rows` and `row_duals` should have the lengths of `num_rows`.
    pub fn set_solution(
        &mut self,
        cols: Option<&[f64]>,
        rows: Option<&[f64]>,
        col_duals: Option<&[f64]>,
        row_duals: Option<&[f64]>,
    ) {
        self.try_set_solution(cols, rows, col_duals, row_duals)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to hot-start using an initial guess by passing the column and row primal and dual solution values.
    /// See highs_c_api.h for further details.
    ///
    /// If the data passed in do not have the correct lengths, an `Err` is returned.
    /// `cols` and `col_duals` should have the lengths of `num_cols`.
    /// `rows` and `row_duals` should have the lengths of `num_rows`.
    pub fn try_set_solution(
        &mut self,
        cols: Option<&[f64]>,
        rows: Option<&[f64]>,
        col_duals: Option<&[f64]>,
        row_duals: Option<&[f64]>,
    ) -> Result<(), HighsStatus> {
        let num_cols = self.highs.num_cols()?;
        let num_rows = self.highs.num_rows()?;
        if let Some(cols) = cols {
            if cols.len() != num_cols {
                return Err(HighsStatus::Error);
            }
        }
        if let Some(rows) = rows {
            if rows.len() != num_rows {
                return Err(HighsStatus::Error);
            }
        }
        if let Some(col_duals) = col_duals {
            if col_duals.len() != num_cols {
                return Err(HighsStatus::Error);
            }
        }
        if let Some(row_duals) = row_duals {
            if row_duals.len() != num_rows {
                return Err(HighsStatus::Error);
            }
        }
        unsafe {
            highs_call!(Highs_setSolution(
                self.highs.mut_ptr(),
                cols.map(|x| { x.as_ptr() }).unwrap_or(null()),
                rows.map(|x| { x.as_ptr() }).unwrap_or(null()),
                col_duals.map(|x| { x.as_ptr() }).unwrap_or(null()),
                row_duals.map(|x| { x.as_ptr() }).unwrap_or(null())
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
}

/// Concrete values of the solution
#[derive(Clone, Debug)]
pub struct Solution {
    pub colvalue: Vec<f64>,
    pub coldual: Vec<f64>,
    pub rowvalue: Vec<f64>,
    pub rowdual: Vec<f64>,
}

// impl Solution {
//     /// The optimal values for each variables (in the order they were added)
//     pub fn columns(&self) -> &Vec<f64> {
//         &self.colvalue
//     }
//     /// The optimal values for each variables in the dual problem (in the order they were added)
//     pub fn dual_columns(&self) -> &Vec<f64> {
//         &self.coldual
//     }
//     /// The value of the constraint functions
//     pub fn rows(&self) -> &Vec<f64> {
//         &self.rowvalue
//     }
//     /// The value of the constraint functions in the dual problem
//     pub fn dual_rows(&self) -> &Vec<f64> {
//         &self.rowdual
//     }
// }

/// Basis statuses for a problem with concrete solution
#[derive(Clone, Debug)]
pub struct Basis {
    colstatus: Vec<usize>,
    rowstatus: Vec<usize>,
}

impl Basis {
    /// The basis status for each of the columns
    pub fn columns(&self) -> &[usize] {
        &self.colstatus
    }

    /// The basis status for each of the rows
    pub fn rows(&self) -> &[usize] {
        &self.rowstatus
    }
}
