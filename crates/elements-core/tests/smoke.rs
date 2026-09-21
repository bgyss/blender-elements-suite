#[test]
fn crate_version_is_exposed() {
    assert_eq!(elements_core::VERSION, env!("CARGO_PKG_VERSION"));
}
