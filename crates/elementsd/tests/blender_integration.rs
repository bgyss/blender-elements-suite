//! Runs only when BLENDER_BIN points at a Blender executable. CI sets it on
//! the pre-release job; local runs skip it.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elementsd is two levels below the root")
        .to_path_buf()
}

#[test]
fn blender_loads_the_addon_and_receives_a_frame() {
    let Ok(blender) = std::env::var("BLENDER_BIN") else {
        eprintln!("BLENDER_BIN is unset; skipping the Blender integration test");
        return;
    };

    let root = repo_root();
    let build = Command::new(if cfg!(windows) { "python" } else { "python3" })
        .arg(root.join("scripts/build_addon.py"))
        .current_dir(&root)
        .output()
        .expect("build_addon.py must run");
    assert!(
        build.status.success(),
        "build_addon.py failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let zip = root.join("dist/blender_elements-0.1.0.zip");
    assert!(zip.exists(), "{} was not built", zip.display());

    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(root.join("tests/blender/test_roundtrip.py"))
        .arg("--")
        .arg(&zip)
        .arg(env!("CARGO_BIN_EXE_elementsd"))
        .output()
        .expect("blender must run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("blender roundtrip ok"),
        "stdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
