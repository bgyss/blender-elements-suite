{
  description = "Elements Suite: real-time GPU field simulation for Blender";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          # Rust comes from rustup so rust-toolchain.toml stays authoritative.
          packages = with pkgs; [
            rustup
            python311
            ruff
            uv
            just
            cargo-nextest
            pkg-config
          ] ++ lib.optionals stdenv.isLinux [
            # Software Vulkan, so `cargo test` works in a Nix shell and in CI.
            mesa
            vulkan-loader
            vulkan-tools
          ];

          shellHook = ''
            export RUSTUP_TOOLCHAIN=$(sed -n 's/^[[:space:]]*channel = "\(.*\)"/\1/p' rust-toolchain.toml)
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export VK_ICD_FILENAMES=${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json
            export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
          '';
        };
      });
}
