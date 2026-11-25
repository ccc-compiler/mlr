//! The library form of the compiler, so that the integration tests can reach
//! the IR and the passes directly.

pub mod ast;
pub mod build;
pub mod check;
pub mod ir;
pub mod lexer;
pub mod parser;
pub mod passes;
pub mod pt;
