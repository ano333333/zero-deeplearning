{
  description = ''
    Rust development environment for sample implementations of "ゼロから作るDeep Learning(Deep Learning from Scratch)" in Rust'';

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            clippy
            rustc
            rustfmt
            rust-analyzer
            pkg-config
            fontconfig
            freetype
          ];

          RUST_BACKTRACE = "1";
        };

        formatter = pkgs.nixfmt;
      });
}
