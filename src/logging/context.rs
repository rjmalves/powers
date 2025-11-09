//! Thread-local logging context for enriching log messages

use std::cell::RefCell;
use std::time::Duration;

/// Context information that enriches log messages
#[derive(Debug, Clone, Default)]
pub struct LogContext {
    pub iteration: Option<usize>,
    pub lower_bound: Option<f64>,
    pub simulation_cost: Option<f64>,
    pub forward_time: Option<Duration>,
    pub backward_time: Option<Duration>,
    pub total_time: Option<Duration>,
}

thread_local! {
    static CONTEXT: RefCell<LogContext> = RefCell::new(LogContext::default());
}

impl LogContext {
    /// Get the current thread-local context
    pub fn current() -> Self {
        CONTEXT.with(|ctx| ctx.borrow().clone())
    }

    /// Set the current thread-local context
    pub fn set(context: LogContext) {
        CONTEXT.with(|ctx| *ctx.borrow_mut() = context);
    }

    /// Clear the current thread-local context
    pub fn clear() {
        CONTEXT.with(|ctx| *ctx.borrow_mut() = LogContext::default());
    }

    /// Execute a closure with a specific context, restoring the previous context afterwards
    pub fn with_iteration<F, R>(iteration: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let prev = Self::current();
        let mut new_context = prev.clone();
        new_context.iteration = Some(iteration);
        Self::set(new_context);
        let result = f();
        Self::set(prev);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_default() {
        LogContext::clear();
        let ctx = LogContext::current();
        assert!(ctx.iteration.is_none());
    }

    #[test]
    fn test_context_set_and_get() {
        let ctx = LogContext {
            iteration: Some(42),
            lower_bound: Some(1234.56),
            ..Default::default()
        };
        LogContext::set(ctx.clone());
        let retrieved = LogContext::current();
        assert_eq!(retrieved.iteration, Some(42));
        assert_eq!(retrieved.lower_bound, Some(1234.56));
        LogContext::clear();
    }

    #[test]
    fn test_context_clear() {
        LogContext::set(LogContext {
            iteration: Some(10),
            ..Default::default()
        });
        LogContext::clear();
        let ctx = LogContext::current();
        assert!(ctx.iteration.is_none());
    }

    #[test]
    fn test_with_iteration() {
        LogContext::clear();
        let result = LogContext::with_iteration(99, || {
            let ctx = LogContext::current();
            ctx.iteration.unwrap()
        });
        assert_eq!(result, 99);
        // Context should be restored
        let ctx = LogContext::current();
        assert!(ctx.iteration.is_none());
    }
}
