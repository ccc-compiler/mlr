//! Unit tests of the IR verifier: hand-built modules that must be rejected.

use mlr::ir::{self, Attr, Block, FuncDecl, FuncOp, Loc, Op, Region, Type, Value};

/// A builder helper: a fresh value of a type.
fn val(ty: Type) -> Value {
    ir::Builder::new().fresh_value(ty)
}

fn main_fn(ops: Vec<Op>) -> ir::Module {
    let mut m = ir::Module::default();
    m.funcs.push(FuncOp {
        name: "main".into(),
        params: vec![],
        results: vec![Type::scalar("i32").unwrap()],
        body: Region {
            blocks: vec![Block { args: vec![], ops }],
        },
        attrs: vec![],
        loc: Loc::default(),
    });
    m
}

#[test]
fn rejects_a_use_of_a_value_with_a_mismatched_type() {
    // A constant i32 used where the same id carries a memref: the ids are
    // reused by the lowering passes, so a stale reference is possible.
    let c = val(Type::Scalar(ir::ScalarKind::I32));
    let mut cst = Op::new("arith.constant");
    cst.attrs.push(("value".into(), Attr::Int(1)));
    cst.results.push(c.clone());
    // The same value again, but typed as a float: a type mismatch.
    let mut cst2 = Op::new("arith.constant");
    cst2.attrs.push(("value".into(), Attr::Int(2)));
    cst2.results.push(Value {
        id: c.id,
        ty: Type::Scalar(ir::ScalarKind::F32),
    });
    let mut ret = Op::new("func.return");
    ret.operands = vec![c.clone()];
    let err = ir::verify(&main_fn(vec![cst, cst2, ret])).unwrap_err();
    assert!(err.contains("defined twice"), "{}", err);
}

#[test]
fn rejects_a_call_with_the_wrong_operand_count() {
    let mut m = main_fn(vec![]);
    m.decls.push(FuncDecl {
        name: "mlr_dummy".into(),
        params: vec![Type::scalar("f32").unwrap(), Type::scalar("f32").unwrap()],
        results: vec![],
    });
    let a = val(Type::scalar("f32").unwrap());
    let mut cst = Op::new("arith.constant");
    cst.attrs.push(("value".into(), Attr::Float(1.0)));
    cst.results.push(a.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_dummy".into())));
    call.operands = vec![a.clone()]; // one operand, two declared
    let mut zero = Op::new("arith.constant");
    zero.attrs.push(("value".into(), Attr::Int(0)));
    let z = val(Type::scalar("i32").unwrap());
    zero.results.push(z.clone());
    let mut ret = Op::new("func.return");
    ret.operands = vec![z];
    let ops = vec![cst, call, zero, ret];
    m.funcs[0].body.blocks[0].ops = ops;
    let err = ir::verify(&m).unwrap_err();
    assert!(err.contains("the call of `mlr_dummy` has 1 operand(s)"), "{}", err);
}

#[test]
fn rejects_a_return_that_does_not_match_the_signature() {
    let a = val(Type::scalar("f32").unwrap());
    let mut cst = Op::new("arith.constant");
    cst.attrs.push(("value".into(), Attr::Float(1.0)));
    cst.results.push(a.clone());
    let mut ret = Op::new("func.return");
    ret.operands = vec![a.clone()]; // the signature expects i32
    let m = main_fn(vec![cst, ret]);
    let err = ir::verify(&m).unwrap_err();
    assert!(err.contains("the 0-th returned value has type f32"), "{}", err);
}
