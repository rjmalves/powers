use highs;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
enum MPSRowSense {
    Free,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
}

impl MPSRowSense {
    fn build(content: &str) -> Result<MPSRowSense, &'static str> {
        match content {
            "N" => Ok(MPSRowSense::Free),
            "G" => Ok(MPSRowSense::GreaterThanOrEqual),
            "L" => Ok(MPSRowSense::LessThanOrEqual),
            "E" => Ok(MPSRowSense::Equal),
            _ => Err("Failed to parse row sense"),
        }
    }
}
#[derive(Debug, PartialEq)]
struct MPSRow {
    sense: MPSRowSense,
    name: String,
}

impl MPSRow {
    fn build(content: &str) -> Result<MPSRow, &'static str> {
        let mut fields = content.split_ascii_whitespace();
        let sense = match fields.next() {
            Some(field) => MPSRowSense::build(field).unwrap(),
            None => return Err("Error parsing row content"),
        };
        let name = match fields.next() {
            Some(field) => field.to_string(),
            None => return Err("Error parsing row content"),
        };
        Ok(MPSRow { sense, name })
    }
}
#[derive(Debug, PartialEq)]
struct MPSColumn {
    column_name: String,
    row_name: String,
    value: f64,
}

impl MPSColumn {
    fn build(content: &str) -> Result<MPSColumn, &'static str> {
        let mut fields = content.split_ascii_whitespace();
        let column_name = match fields.next() {
            Some(field) => field.to_string(),
            None => return Err("Error parsing column content"),
        };
        let row_name = match fields.next() {
            Some(field) => field.to_string(),
            None => return Err("Error parsing column content"),
        };
        let value = match fields.next() {
            Some(field) => field.parse::<f64>().unwrap(),
            None => return Err("Error parsing column content"),
        };
        Ok(MPSColumn {
            column_name,
            row_name,
            value,
        })
    }
}
#[derive(Debug, PartialEq)]
struct MPSRHS {
    row_index: usize,
    value: f64,
}

impl MPSRHS {
    fn build(
        content: &str,
        row_name_map: &HashMap<String, usize>,
    ) -> Result<MPSRHS, &'static str> {
        let mut fields = content.split_ascii_whitespace();
        fields.next();
        let row_name = match fields.next() {
            Some(field) => field.to_string(),
            None => return Err("Error parsing RHS content"),
        };
        let row_index = row_name_map.get(&row_name).unwrap();
        let value = match fields.next() {
            Some(field) => field.parse::<f64>().unwrap(),
            None => return Err("Error parsing RHS content"),
        };
        Ok(MPSRHS {
            row_index: *row_index,
            value,
        })
    }
}

enum MPSBoundType {
    Lower,
    Upper,
}

impl MPSBoundType {
    fn build(content: &str) -> Result<MPSBoundType, &'static str> {
        match content {
            "UP" => Ok(MPSBoundType::Upper),
            "LO" => Ok(MPSBoundType::Lower),
            _ => Err("Failed to parse bound type"),
        }
    }
}

struct MPSBound {
    kind: MPSBoundType,
    column_name: String,
    value: f64,
}

impl MPSBound {
    fn build(content: &str) -> Result<MPSBound, &'static str> {
        let mut fields = content.split_ascii_whitespace();
        let kind = match fields.next() {
            Some(field) => MPSBoundType::build(field).unwrap(),
            None => return Err("Error parsing bound content"),
        };
        fields.next();
        let column_name = match fields.next() {
            Some(field) => field.to_string(),
            None => return Err("Error parsing bound content"),
        };
        let value = match fields.next() {
            Some(field) => field.parse::<f64>().unwrap(),
            None => return Err("Error parsing bound content"),
        };
        Ok(MPSBound {
            kind,
            column_name,
            value,
        })
    }
}

fn wait_for_pattern<'a, P>(
    lines: impl Iterator<Item = &'a str>,
    mut predicate: P,
) where
    P: FnMut(&str) -> bool,
{
    for line in lines {
        if predicate(line) {
            return;
        }
    }
}

fn parse_sense<'a>(
    mut lines: impl Iterator<Item = &'a str>,
) -> Result<highs::Sense, &'static str> {
    match lines.next() {
        Some(line) => match line.trim() {
            "MIN" => Ok(highs::Sense::Minimise),
            "MAX" => Ok(highs::Sense::Maximise),
            _ => Err("Failed to parse optimization sense"),
        },
        None => Err("Error reading file"),
    }
}

#[derive(Debug)]
struct LPRow {
    pub sense: MPSRowSense,
    pub rhs: f64,
}

impl LPRow {
    fn new() -> Self {
        Self {
            sense: MPSRowSense::Free,
            rhs: 0.0,
        }
    }
}

fn parse_rows<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> Result<(HashMap<String, usize>, Vec<LPRow>, String), &'static str> {
    let mut row_name_map = HashMap::<String, usize>::new();
    let mut row_factors = Vec::<LPRow>::new();
    let mut current_row_index: usize = 0;
    let mut free_row_name = vec![];

    for line in lines {
        if line.contains("COLUMNS") {
            break;
        }
        let mps_row = MPSRow::build(line).unwrap();
        if mps_row.sense == MPSRowSense::Free {
            free_row_name.push(mps_row.name);
        } else {
            row_name_map.insert(mps_row.name, current_row_index);
            let mut row = LPRow::new();
            row.sense = mps_row.sense;
            row_factors.push(row);
            current_row_index += 1;
        }
    }
    let name = free_row_name.get(0).unwrap();
    Ok((row_name_map, row_factors, name.clone()))
}

#[derive(Debug)]
struct LPColumn {
    pub objective_factor: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub constraints_factors: Vec<(usize, f64)>,
}

impl LPColumn {
    fn new() -> Self {
        Self {
            objective_factor: 0.0,
            lower_bound: 0.0,
            upper_bound: std::f64::INFINITY,
            constraints_factors: Vec::<(usize, f64)>::new(),
        }
    }
}

fn update_lp_column_with_factors<'a>(
    lp_column: &mut LPColumn,
    mps_column: &MPSColumn,
    free_row_name: &'a str,
    row_name_map: &HashMap<String, usize>,
) {
    if mps_column.row_name == free_row_name {
        lp_column.objective_factor = mps_column.value;
    } else {
        let row_index = row_name_map.get(&mps_column.row_name).unwrap();
        lp_column
            .constraints_factors
            .push((*row_index, mps_column.value));
    }
}

fn parse_columns<'a>(
    lines: impl Iterator<Item = &'a str>,
    row_name_map: &HashMap<String, usize>,
    free_row_name: String,
) -> Result<(HashMap<String, usize>, Vec<LPColumn>), &'static str> {
    let mut column_name_map = HashMap::<String, usize>::new();
    let mut column_factors = Vec::<LPColumn>::new();
    let mut current_column_index: usize = 0;
    for line in lines {
        if line.contains("RHS") {
            break;
        }
        let mps_column = MPSColumn::build(line).unwrap();
        let column_name = &mps_column.column_name;
        let mut lp_column = match column_name_map.get(column_name) {
            Some(index) => column_factors.get_mut(*index),
            None => {
                column_name_map
                    .insert(column_name.to_string(), current_column_index);
                current_column_index += 1;
                column_factors.push(LPColumn::new());
                column_factors.get_mut(current_column_index - 1)
            }
        }
        .unwrap();
        update_lp_column_with_factors(
            &mut lp_column,
            &mps_column,
            &free_row_name,
            row_name_map,
        );
    }
    Ok((column_name_map, column_factors))
}

fn parse_rhs<'a>(
    lines: impl Iterator<Item = &'a str>,
    row_name_map: &HashMap<String, usize>,
    row_factors: &mut Vec<LPRow>,
) {
    for line in lines {
        if line.contains("RANGES") {
            break;
        }
        let mps_rhs = MPSRHS::build(line, row_name_map).unwrap();
        let sense = &row_factors.get(mps_rhs.row_index).unwrap().sense;
        row_factors.get_mut(mps_rhs.row_index).unwrap().rhs = match sense {
            MPSRowSense::GreaterThanOrEqual => -mps_rhs.value,
            _ => mps_rhs.value,
        };
    }
}

fn update_lp_column_with_bounds(
    lp_column: &mut LPColumn,
    mps_bound: &MPSBound,
) {
    match mps_bound.kind {
        MPSBoundType::Lower => lp_column.lower_bound = mps_bound.value,
        MPSBoundType::Upper => lp_column.upper_bound = mps_bound.value,
    };
}

fn parse_bounds<'a>(
    lines: impl Iterator<Item = &'a str>,
    column_name_map: &mut HashMap<String, usize>,
    column_factors: &mut Vec<LPColumn>,
) {
    for line in lines {
        if line.contains("ENDATA") {
            break;
        }
        let mps_bound = MPSBound::build(line).unwrap();
        let index = column_name_map.get(&mps_bound.column_name).unwrap();
        update_lp_column_with_bounds(
            column_factors.get_mut(*index).unwrap(),
            &mps_bound,
        );
    }
}

fn add_rows<'a>(
    row_factors: &Vec<LPRow>,
    pb: &'a mut highs::ColProblem,
) -> Vec<highs::Row> {
    let mut rows = Vec::<highs::Row>::new();
    println!("Adding rows...");
    for (_, row_factor) in row_factors.iter().enumerate() {
        // println!("{:?}", row_factor);
        rows.push(pb.add_row(..row_factor.rhs));
    }
    rows
}

fn replace_indices_with_rows(
    rows: &Vec<highs::Row>,
    row_factors: &Vec<LPRow>,
    factors: &Vec<(usize, f64)>,
) -> Vec<(highs::Row, f64)> {
    factors
        .iter()
        .map(|f| {
            (
                rows[f.0],
                match row_factors[f.0].sense {
                    MPSRowSense::GreaterThanOrEqual => -f.1,
                    _ => f.1,
                },
            )
        })
        .collect()
}

fn add_columns<'a>(
    rows: &Vec<highs::Row>,
    row_factors: &Vec<LPRow>,
    column_factors: &Vec<LPColumn>,
    pb: &'a mut highs::ColProblem,
) {
    println!("Adding columns...");
    for (_, column) in column_factors.iter().enumerate() {
        let highs_factors = replace_indices_with_rows(
            rows,
            row_factors,
            &column.constraints_factors,
        );
        // println!("{}, {:?}", index, column);
        pb.add_column(
            column.objective_factor,
            column.lower_bound..column.upper_bound,
            highs_factors,
        );
    }
}

pub fn mps2highs<'a>(
    mut lines: impl Iterator<Item = &'a str>,
) -> Result<highs::Model, &'static str> {
    let mut pb = highs::ColProblem::default();
    println!("Parsing MPS file...");
    wait_for_pattern(&mut lines, |line| line.contains("OBJSENSE"));
    let sense = parse_sense(&mut lines)?;
    wait_for_pattern(&mut lines, |line| line.contains("ROWS"));
    let (row_name_map, mut row_factors, free_row_name) =
        parse_rows(&mut lines)?;
    let (mut column_name_map, mut column_factors) =
        parse_columns(&mut lines, &row_name_map, free_row_name)?;

    parse_rhs(&mut lines, &row_name_map, &mut row_factors);
    wait_for_pattern(&mut lines, |line| line.contains("BOUNDS"));
    parse_bounds(&mut lines, &mut column_name_map, &mut column_factors);
    let rows = add_rows(&row_factors, &mut pb);
    add_columns(&rows, &row_factors, &column_factors, &mut pb);
    println!("Total number of columns: {}", pb.num_cols());
    println!("Total number of rows: {}", pb.num_rows());

    Ok(pb.optimise(sense))
}

#[cfg(test)]
mod tests {

    #[test]
    fn read_mps() {}
}
